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
from geopy.geocoders import Nominatim
from osgeo import gdal, osr


def read_prj_file(prj_file):
    '''
    Read ESRI projection file into string.
    '''
    prj_text = open(prj_file, 'r').read()
    srs = osr.SpatialReference()
    if srs.ImportFromWkt(prj_text):
        raise ValueError("Error importing PRJ information from: %s" % prj_file)
    prj = srs.ExportToProj4()
    if prj == "":
        return '+proj=merc +lon_0=0 +lat_ts=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs '
    else:
        return prj


def xy2lonlat(xy, prj=""):
    '''
    Convert northing/easting coordinates to commonly-used lat/lon.
    '''
    x, y = xy
    inProj = Proj(prj) 
    outProj = Proj(init='epsg:4326')
    lon, lat = transform(inProj,outProj,x,y) 
    return lon, lat
    

def lonlat2xy(lonlat, prj=""):
    '''
    Convert commonly-used lat/lon to northing/easting coordinates.
    '''
    lon, lat = lonlat
    inProj = Proj(init='epsg:4326')
    outProj = Proj(prj) 
    x, y = transform(inProj,outProj,lon,lat) 
    return x, y
    

def polygon_xy2lonlat(p, prj=""):
    '''
    Convert polygon coordinates from meter to lon/lat.
    '''
    inProj = Proj(prj) 
    outProj = Proj(init='epsg:4326')
    x, y = p.exterior.coords.xy
    locs_meter = zip(x, y)
    locs_lonlat= [transform(inProj,outProj,x1,y1) for x1,y1 in locs_meter]
    return Polygon(locs_lonlat)


def polygon_lonlat2xy(p, prj=""):
    '''
    Convert polygon coordinates from lon/lat to meter.
    '''
    inProj = Proj(init='epsg:4326')
    outProj = Proj(prj) 
    lon, lat = p.exterior.coords.xy
    locs_lonlat = zip(lon, lat)
    locs_meter = [transform(inProj,outProj,x,y) for x,y in locs_lonlat]
    return Polygon(locs_meter)


def get_bounds(gdf):
    bounds = np.array(gdf['geometry'].apply(lambda p: list(p.bounds)).values.tolist())
    xmin = bounds[:,[0,2]].min()
    xmax = bounds[:,[0,2]].max()
    ymin = bounds[:,[1,3]].min()
    ymax = bounds[:,[1,3]].max()
    return xmin, ymin, xmax, ymax


def filter_gdf_by_polygon(gdf, polygon):
    spatial_index = gdf.sindex
    possible_matches_index = list(spatial_index.intersection(polygon.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(polygon)]
    return precise_matches


def filter_gdf_by_centered_window(gdf0, center=None, window=None):
    if window is None:
        return gdf0
    else:
        latmin, lonmin, latmax, lonmax = satimg.bounding_box_at_location(center, window)
        pbox = Polygon([(lonmin,latmin), (lonmax,latmin), (lonmax,latmax), (lonmin,latmax)])
        return filter_gdf_by_polygon(gdf0, pbox)
    
    