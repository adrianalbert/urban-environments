# Comparing urban environments using satellite imagery and convolutional neural networks

This repository contains code related to the paper [Using convolutional networks and satellite imagery to identify patterns in urban environments at a large scale](https://arxiv.org/abs/1704.02965). A slightly modified version of the paper appears in the [proceedings](http://www.kdd.org/kdd2017/papers/view/using-convolutional-networks-and-satellite-imagery-to-identify-patterns-in-) of the [ACM KDD 2017](http://www.kdd.org/kdd2017/) conference.

If you use the code, data, or analysis results in this paper, we kindly ask that you cite the paper above as:

> _Using convolutional networks and satellite imagery to identify patterns in urban environments at a large scale_. A. Toni Albert, J. Kaur, M.C. Gonzalez, 2017. In Proceedings of the ACM SigKDD 2017 Conference, Halifax, Nova Scotia, Canada.

This repository contains the `Python` implementation of the data processing, model training, and analysis presented in the paper:

* code to construct training and evaluation datasets for land use classification of urban environments is in the [dataset-collection](./dataset-collection) folder
* Keras implementations of the convolutional neural networks classifiers used for this paper are in [keras-models](./keras-models)
* utilities for data ingestion and multi-GPU training in Keras (with a TensorFlow backend) are in [keras-utils](./keras-utils)
* code to train and validate the models, and to produce the analysis and figures in the paper is in the notebooks in the [land-use-classification](./land-use-classification) folder.

## Creating the _Urban Environments_ dataset

To build a consistent satellite and GIS data ingestion pipeline, we have developed, and made extensive use of, the `pysatml`  module in `Python`. The development page for that package can be found [here](https://github.com/adrianalbert/pysatml). That page offers more details and tutorials on how to use that module to streamline the processing and ingestion of typical raster and vector data formats that are common in remote-sensing and GIS data analysis. You can simply install the `pysatml` module via `pip`:

```bash
pip install pysatml
```

To construct training and validation datasets, we combine ground-truth labels from the [Urban Atlas](http://www.eea.europa.eu/data-and-maps/data/urban-atlas.) large-scale land use classification survey with satellite imagery obtained from [Google Maps](https://developers.google.com/maps/documentation/javascript/get-api-key). Our process consists of the following three main steps:
* Downloading land use survey files from Urban Atlas at [this link](http://www.eea.europa.eu/data-and-maps/data/urban-atlas). Unfortunately, this first step is a manual process. 
* Generating an appropriate list of locations - as (lat,lon) tuples - where to sample satellite imagery
* Downloading satellite imagery from Google Maps Static API ([API key needed](https://developers.google.com/maps/documentation/javascript/get-api-key))

We plan to make the actual dataset available after further curation and ensuring that this complies with all applicable data licenses of the dataset used. In the meantime, we describe below the detailed procedure used to construct this dataset. 

#### Obtaining shape files for ground truth labels
First, manually download GIS polygon data for ground truth, available as shapefiles at [the Urban Atlas website](http://www.eea.europa.eu/data-and-maps/data/urban-atlas). Unfortunately there is no way to automate this because of the confirmation web forms used on the Urban Atlas website. The vector data integration and sampling peline is detailed in a [notebook](./dataset-collection/Urban&nbspAtlas-process&nbspshapefiles&nbspto&nbspcompute&nbspstats&nbspand&nbspextract&nbspsampling&nbsplocations.ipynb).

You can save these shapefiles (along with their respective projection files) to `/home/data/urban-atlas/shapefiles/`, or to a folder of your choosing, and manually edit the paths in the [notebook](./dataset-collection/Urban&nbspAtlas-process&nbspshapefiles&nbspto&nbspcompute&nbspstats&nbspand&nbspextract&nbspsampling&nbsplocations.ipynb).

The paper uses the shapefiles for several cities, some of which are expected to be more "similar" to each other than others. We have experimented with data for several other cities, however we decided to only include the following six cities in the analysis in the paper:
* Athens
* Barcelona
* Berlin
* Madrid
* Budapest 
* Roma 

#### Ingesting and processing _Urban Atlas_ shapefiles
To process the vector data (shapefiles), we have developed the `UAShapeFile` class that encapsulates much of the functionality needed for shapefile data ingestion, sampling, etc. See the code in the [urbanatlas](./urbanatlas) folder. This is a lighweight wrapper around a ```GeoDataFrame``` object from the ```geopandas``` Python module. A sample usage is as follows:

```Python
>>> from urbanatlas import UAShapeFile
>>> myshapefile = "ro001l_bucharest.shp"
>>> mycity = UAShapeFile(myshapefile)
```

#### Creating ground truth validation raster grids

First, let's crop a window of width $W \times W$ (in km) centered at the city center:

```Python
>>> W = 25 # in Km
>>> window = (W, W)
>>> mycity_crop = mycity.crop_centered_window(city_center, window)
```
                    
The `UAShapeFile` class allows to compute a ground raster of a given grid size:
```Python
>>> grid_cell = 100
>>> grid_size = (grid_cell, grid_cell)
>>> raster, locations_grid, cur_classes = mycity_crop.extract_class_raster(grid_size=grid_size)
```
    
This step can be skipped in the case of the six cities above, for which the data (files ```sample_locations_raster.csv``` and ```ground_truth_class_raster.npz```) can be found in this repository under [processed-data](./processed-data).

#### Selecting appropriate samples

Now, let's generate locations to download imagery for from the shapefile, using a stratified sampling procedure that takes into account the imbalances in the classes.

```Python
>>> N_SAMPLES_PER_CLASS = 1250
>>> MAX_SAMPLES_PER_POLY = 50
>>> locations_train = mycity.generate_sampling_locations(thresh_area=thresh_area, \
                                                     n_samples_per_class=N_SAMPLES_PER_CLASS,\
                                                     max_samples=MAX_SAMPLES_PER_POLY)
```

This step can be skipped in the case of the six cities above, for which the data (files ```additional_sample_locations.csv```) can be found in this repository under [processed-data](./processed-data).

#### Downloading satellite imagery
This step is outlined in ./dataset-collection/Urban Atlas - extract images.ipynb

