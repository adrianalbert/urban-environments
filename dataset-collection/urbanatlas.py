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

import pysatml
from pysatml.utils import gis_utils as gu
from pysatml.utils import vector_utils as vu
from pysatml import satimage as satimg

N_SAMPLES_PER_CITY  = 25000
N_SAMPLES_PER_CLASS = 1250
MAX_SAMPLES_PER_POLY= 50

class UAShapeFile():
	'''
	Class that encapsulates functionality for analyzing GIS vector data from the Urban Atlas dataset, published as shapefiles. The Urban Atlas is a land use dataset of 300 cities in Europe. 
	'''

	def __init__(self, shapefile, prjfile=None, class_col='ITEM', consolidate_classes=False, **kwargs):
		'''
		Initialize a GeoPandas GeoDataFrame with settings specific to the published Urban Atlas shape files.
		'''
		self._shapefile = shapefile
		self._class_col = class_col
		for k,v in kwargs.iteritems():
			setattr(self, "_%s"%k, v)

		# read in shape file
		self._gdf = load_shapefile(self._shapefile)
		if self._gdf is None:
			return 
		if consolidate_classes:
			self._gdf = consolidate_UA_classes(self._gdf, self._class_col)

		self._classes = self._gdf[self._class_col].unique()
		print "%d polygons | %d land use classes" % (len(self._gdf), len(self._classes))

		# read in projection file associated with shapefile, if available
		self._prjfile = shapefile.replace(".shp", ".prj") if prjfile is None else prjfile
		try:
			self._prj = gu.read_prj_file(self._prjfile)  
		except:
			print "Error: cannot find projection file %s" % self._prjfile
			self._prj = ""

		self.compute_bounds()


	def compute_bounds(self):
		# compute bounds for current shapefile for easier access later
		lonmin, latmin, lonmax, latmax = vu.compute_gdf_bounds(self._gdf)
		self._bounds = (lonmin, latmin, lonmax, latmax)
		xmin, ymin = gu.lonlat2xy((lonmin, latmin), prj=self._prj)
		xmax, ymax = gu.lonlat2xy((lonmax, latmax), prj=self._prj)
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
							.apply(lambda x: x['SHAPE_AREA'].sum())
		frac_classified = classified_area/box_area
		return frac_classified

	def filter_by_polygon(self, poly):
		return vu.filter_gdf_by_polygon(self._gdf, poly)

	def crop_centered_window(self, center, window):
		'''
		Returns a UAShapeFile object obtained from original one by cropping a window of (W, H) (in kilometers) around a center (lon, lat).
		'''
		new_self = copy.deepcopy(self) 
		new_self._gdf = vu.filter_gdf_by_centered_window(new_self._gdf, center, window)
		new_self.compute_bounds()
		return new_self

	def extract_class_raster(self,center=None,window=None,grid_size=(100,100)):
		if center is None:
			lonmin, latmin, lonmax, latmax = self._bounds
			center = ((latmin+latmax)/2.0, (lonmin+lonmax)/2.0)
		if window is not None:
			bbox = gu.bounding_box_at_location(center, window)
		else:
			bbox = self._bounds
		return construct_class_raster(self._gdf, bbox, class_col=self._class_col, grid_size=grid_size)

	def generate_sampling_locations(self, n_samples_per_class=N_SAMPLES_PER_CLASS,thresh_area=0.25,max_samples=MAX_SAMPLES_PER_POLY):
		gdf_sel = self._gdf[self._gdf.SHAPE_AREA>=thresh_area]
		return generate_sampling_locations(gdf_sel, n_samples_per_class=n_samples_per_class, class_col=self._class_col, max_samples=max_samples)


def consolidate_UA_classes(gdf, class_col='ITEM'):
	consolidate_classes = {
	    "Continuous Urban Fabric (S.L. > 80%)":"High Density Urban Fabric",
	     "Discontinuous Dense Urban Fabric (S.L. : 50% -  80%)":"High Density Urban Fabric",
	     "Discontinuous Medium Density Urban Fabric (S.L. : 30% - 50%)":"Medium Density Urban Fabric",
	     "Discontinuous Low Density Urban Fabric (S.L. : 10% - 30%)":"Low Density Urban Fabric",
	     "Discontinuous Very Low Density Urban Fabric (S.L. < 10%)":"Low Density Urban Fabric"
	}
	gdf[class_col] = gdf[class_col].apply(
	    lambda x: consolidate_classes[x] if x in consolidate_classes else x)

	include_classes = ["Green urban areas", 
	                   "Airports",
	                   "Forests",
	                   "Agricultural + Semi-natural areas + Wetlands",
	                   # "Railways and associated land",
	                   "High Density Urban Fabric", 
	                    #"Mineral extraction and dump sites",
	                   "Medium Density Urban Fabric", 
	                   "Low Density Urban Fabric",
	                   "Water bodies",
	                   "Sports and leisure facilities",
	                   "Industrial, commercial, public, military and private units"]
	gdf = gdf[gdf[class_col].isin(include_classes)]
	return gdf


def load_shapefile(shapefile, class_col="ITEM"):
	# read in shapefile
	try:
		gdf = gpd.GeoDataFrame.from_file(shapefile)
	except:
		print "--> %s: error reading file!"%shapefile
		return None

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




def construct_class_raster(gdf, bbox, class_col="ITEM", label2class=None, grid_size=(100,100)):
	grid_size_lon, grid_size_lat = grid_size
	lonmin_grid, latmin_grid, lonmax_grid, latmax_grid = bbox
	latv = np.linspace(latmin_grid, latmax_grid, grid_size_lat+1)
	lonv = np.linspace(lonmin_grid, lonmax_grid, grid_size_lon+1)
	classes = gdf[class_col].unique()
	label2class = dict(zip(range(len(classes)), classes)) if label2class is None else label2class
	
	raster = np.zeros((grid_size_lon, grid_size_lat, len(classes)))
	locations = []
	for i in range(len(lonv)-1):
		for j in range(len(latv)-1):
			cell_poly = Polygon([(lonv[i],latv[j]), (lonv[i+1],latv[j]), \
								 (lonv[i+1],latv[j+1]), (lonv[i],latv[j+1])])
			gdf_frame = vu.filter_gdf_by_polygon(gdf, cell_poly)
			if len(gdf_frame) == 0:
				continue
			areas_per_class = gdf_frame.groupby(class_col)\
								.apply(lambda x: x.intersection(cell_poly)\
									   .apply(lambda y:y.area*(6400**2)).sum())
			classified_area = areas_per_class.sum()
			if classified_area > 0:
				areas_per_class = areas_per_class / float(classified_area) 
				raster[i,j,:] = [areas_per_class[label2class[k]] if label2class[k] in areas_per_class else 0 for k in range(len(classes))]  
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
	return raster, locations, classes


def generate_sampling_locations(gdf_sel, n_samples_per_class=N_SAMPLES_PER_CLASS,class_col="ITEM", max_samples=MAX_SAMPLES_PER_POLY):

	# select polygons to sample
	select_polygons = gdf_sel.groupby(class_col)\
						.apply(lambda x: sample_polygons(x, 
									n_samples=n_samples_per_class, 
									max_samples=max_samples))
	if class_col not in select_polygons.columns:
		select_polygons.reset_index(inplace=True)
	
	# make sure all polygons are ok
	# some polygons have their geometries messed up in the previous step??
	select_polygons['geometry'] = select_polygons['geometry'].apply(lambda p: p.buffer(0) if not p.is_valid else p)
	
	# sample locations from each polygon
	locations = select_polygons.groupby(class_col)\
				.apply(lambda x: sample_locations_from_polygon(x,
					sample_on_boundary = 'road' in x[class_col].iloc[0].lower() or 'railway' in x[class_col].iloc[0].lower()))
	return locations


def sample_polygons(df, n_samples=1000, max_samples=None):  
	'''
	A stratified sampling of polygons in the DataFrame gdf.
	'''  
	samples_per_poly = (df.SHAPE_AREA/float(df.SHAPE_AREA.min()))\
							.astype(int)
	# print df.ITEM.iloc[0]
	if samples_per_poly.sum() > n_samples:
		pvec = np.array([0.0, 0.2, 0.5, 0.7, 0.9, 0.95, 1])
		bins = np.percentile(samples_per_poly, pvec*100)
		cnts, _ = np.histogram(samples_per_poly, bins)

		ret = []
		x = samples_per_poly
		for i in range(len(bins)-1):
			if cnts[i] == 0:
				continue
			y = x[(x>=bins[i]) & (x<bins[i+1])] if i<len(bins)-2 \
					else x[(x>=bins[i]) & (x<=bins[i+1])]
			# print i, (bins[i], bins[i+1]), cnts[i], pvec[i+1], len(x[(x>=bins[i]) & (x<=bins[i+1])])
			y = y.sample(frac=pvec[i+1])
			ret.append(y)
		ret = pd.concat(ret)
		ret_scaled = (ret.astype(float) / ret.sum() * n_samples)\
						.apply(np.ceil).astype(int)
		ret_df = df.ix[ret_scaled.index]
		ret_df['samples'] = ret_scaled.values
	else:
		ret_df = df
		ret_df['samples'] = samples_per_poly.values
	
	# clamp # samples per polygon if specified
	if max_samples is not None:
		ret_df['samples'] = ret_df['samples'].apply(\
									lambda x: min([x, max_samples]))
	ret_df['samples'] = ret_df['samples'].astype(int)
	return ret_df


def sample_locations_from_polygon(df, sample_on_boundary=False):
	'''
	Given a list of polygons of the same type, generate locations for sampling images
	'''
	polygons = df['geometry']
	nsamples = df['samples']
	
	if not sample_on_boundary:
		centroids = np.array([(p.centroid.coords.xy[0][0], p.centroid.coords.xy[1][0]) \
					  for p in polygons])    
		idx = nsamples > 1
		if idx.sum()>0:
			polygons = polygons[idx]
			nsamples = nsamples[idx]
			locs = [satimg.generate_locations_within_polygon(p, nSamples=m-1, strict=True) \
					for p,m in zip(polygons, nsamples)]
			locs = np.vstack(locs).squeeze()
			locs = np.vstack([locs, centroids])
		else:
			locs = centroids
	else:
		boundaries= [zip(p.exterior.coords.xy[0], p.exterior.coords.xy[1]) \
					 for p in polygons]
		locs = np.array([b[l] for b,m in zip(boundaries,nsamples) \
						 for l in np.random.choice(np.arange(0,len(b)), min([len(b),m]))])
	ret = pd.DataFrame(locs, columns=["lon", "lat"])
	return ret

