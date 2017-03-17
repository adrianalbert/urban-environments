# urban-environments

This repository contains code related to the paper "Using convolutional networks and satellite imagery to identifypaerns in urban environments at a large scale".
* code to acquire and process the data is in the ./dataset-collection folder
* Keras implementations of the convolutional neural networks used for this paper are in ./keras-models
* utilities for loading data, preprocessing etc. are in ./keras-utils
* code to train and validate the models, and to produce the analysis and figures in the paper is in the notebooks in the ./land-use-classification folder.

## To download the ground truth GIS polygon data, manually download the shapefiles at http://www.eea.europa.eu/data-and-maps/data/urban-atlas