#!/bin/bash
conda run -n housing python -m housing_price.ingest_data -r ./temp/raw/ -p ./temp/processed/
conda run -n housing python -m housing_price.train -d ./temp/processed/housing_train.csv -m ./temp/models/
conda run -n housing python -m housing_price.score -d ./temp/processed/housing_test.csv -m ./temp/models/ --rmse --mae

rm -rf temp/