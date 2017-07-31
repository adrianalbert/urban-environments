# numeric packages
import numpy as np
import pandas as pd

# filesystem and OS
import sys, os, time
import glob

# plotting
from matplotlib import pyplot as plt
import matplotlib

# compression
import gzip
import cPickle as pickle
import copy

# geo stuff
import geopandas as gpd
from shapely.geometry import Point, Polygon

import gis_utils as gu

class UAShapeFile():
	'''
	Class that encapsulates functionality for analyzing GIS vector data from the Urban Atlas dataset, published as shapefiles. The Urban Atlas is a land use dataset of 300 cities in Europe. 
	'''

	def __init__(self, shapefile, prjfile=None, class_col='ITEM', **kwargs):
		'''
		Initialize a GeoPandas GeoDataFrame with settings specific to the published Urban Atlas shape files.
		'''
		self._shapefile = shapefile
		self._class_col = class_col
		for k,v in kwargs.iteritems():
			setattr(self, "_%s"%k, v)

		# read in shape file
		self._gdf = load_shapefile(self._shapefile)
		self._classes = self._gdf[self._class_col].unique()
		print "%d polygons | %d land use classes" % (len(self._gdf), len(self._classes))

		# read in projection file associated with shapefile, if available
		self._prjfile = shapefile.replace(".shp", ".prj") if prjfile is None else prjfile
		try:
			self._prj = gu.read_prj_file(self._prjfile)  
		except:
			print "Error: cannot find projection file %s" % self._prjfile
			self._prj = ""

		# compute bounds for current shapefile for easier access later
		lonmin, latmin, lonmax, latmax = get_bounds(self._gdf)
		self._bounds = (lonmin, latmin, lonmax, latmax)
		xmin, ymin = gu.lonlat2xy((lonmin, latmin), prj=prj)
		xmax, ymax = gu.lonlat2xy((lonmax, latmax), prj=prj)
		self._bounds_meters = (xmin, ymin, xmax, ymax)


	def compute_spatial_extent(self):
		xmin, ymin, xmax, ymax = self._bounds_meters
		L = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2) / 1.0e3 / np.sqrt(2)
		return L

	def compute_classified_area(self):
		xmin, ymin, xmax, ymax = self._bounds_meters
		box_area =  (xmax-xmin) / 1.0e3 * (ymax-ymin) / 1.0e3
		classified_area = self._gdf\
							.groupby(self._class_col)\
							.apply(lambda x: x[self._class_col].sum())
		frac_classified = classified_area/box_area
		return frac_classified
		

def load_shapefile(shapefile, class_col="ITEM"):
    # read in shapefile
    try:
        gdf = gpd.GeoDataFrame.from_file(shapefile)
    except:
        print "--> %s: error reading file!"%shapefile
        return None, None

    gdf.columns = [c.upper() if c != "geometry" else c for c in gdf.columns]
    if 'SHAPE_AREA' not in gdf.columns:
        gdf['SHAPE_AREA'] = gdf['geometry'].apply(lambda p: p.area)
    if 'SHAPE_LEN' not in gdf.columns:
        gdf['SHAPE_LEN'] = gdf['geometry'].apply(lambda p: p.length)
        
    # convert area & length to km
    gdf['SHAPE_AREA'] = gdf['SHAPE_AREA'] / 1.0e6 # convert to km^2
    gdf['SHAPE_LEN']  = gdf['SHAPE_LEN'] / 1.0e3 # convert to km
    
    # change coordinate system from northing/easting to lonlat
    targetcrs = {u'ellps': u'WGS84', u'datum': u'WGS84', u'proj': u'longlat'}
    gdf.to_crs(crs=targetcrs, inplace=True)

    return gdf


def get_city_center(shapefile):
    geolocator = Nominatim()
    country_code = shapefile.split("/")[-1].split("_")[0][:2]
    city = " ".join(shapefile.split("/")[-1].split("_")[1:]).split(".")[0]
    location = geolocator.geocode(city + "," + country_code)
    if location is None:
        return None, None
    latlon = (location.latitude, location.longitude)
    return latlon, country_code


def construct_class_raster(gdf, bbox, grid_size=(100,100)):
    grid_size_lon, grid_size_lat = grid_size
    latmin_grid, lonmin_grid, latmax_grid, lonmax_grid = bbox
    latv = np.linspace(latmin_grid, latmax_grid, grid_size_lat+1)
    lonv = np.linspace(lonmin_grid, lonmax_grid, grid_size_lon+1)
    
    raster = np.zeros((grid_size_lon, grid_size_lat, len(classes)))
    locations = []
    for i in range(len(lonv)-1):
        clear_output(wait=True)
        print "%d / %d"%(i, len(lonv)-1)
        for j in range(len(latv)-1):
            cell_poly = Polygon([(lonv[i],latv[j]), (lonv[i+1],latv[j]), \
                                 (lonv[i+1],latv[j+1]), (lonv[i],latv[j+1])])
            gdf_frame = filter_gdf_by_polygon(gdf, cell_poly)
            if len(gdf_frame) == 0:
                continue
            areas_per_class = gdf_frame.groupby("ITEM")\
                                .apply(lambda x: x.intersection(cell_poly)\
                                       .apply(lambda y: y.area*(6400**2)).sum())
            classified_area = areas_per_class.sum()
            if classified_area > 0:
                areas_per_class = areas_per_class / float(classified_area) 
                raster[i,j,:] = [areas_per_class[label2class[k]] if label2class[k] in areas_per_class\
                                 else 0 for k in range(len(classes))]  
                # also save sampling locations
                # only if we can get ground truth label for the cell
                cell_class = areas_per_class.argmax()
                loc = (i, j, 
                       cell_poly.centroid.xy[0][0], 
                       cell_poly.centroid.xy[1][0], 
                       cell_class)
                locations.append(loc)
    
    locations = pd.DataFrame(locations, \
                    columns=["grid-i", "grid-j", "lon", "lat", "class"])
    return raster, locations