# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Installation:
### Prerequisites:
Prerequisite dependencies are stored in `deploy/conda/linux_cpu/py39.yml`. To setup the conda environment:

`$ conda env create --file deploy/conda/linux_cpu_py39.yml`

`$ conda activate mle-dev`

### Setup:
For editable install:
`$ pip install -e .`

For normal install:
`$ pip install .`

## Run code:
### To download and process data:
`$ python src/housing_price/ingest_data.py -r data/raw/ -p data/processed/`
### To train the models:
`$ python src/housing_price/train.py -d data/processed/housing_train.csv -m artifacts/`
### To score trained models:
`$ python src/housing_price/score.py -d data/processed/housing_test.csv -m artifacts/`
### Note:
You can get information on command line arguments for each of the above scripts using `-h` or `--help`. For example: 

`$ python src/housing_price/train.py --help`
## Steps performed:
 - We prepared and cleaned the data. 
 - We checked and imputed missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.


