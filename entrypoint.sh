#!/usr/bin/env bash

python -m housing_price.ingest_data -r data/raw/ -p data/processed/ --log-path logs/log.txt
python -m housing_price.train -d data/processed/housing_train.csv -m artifacts/ --log-path logs/log.txt
python -m housing_price.score -d data/processed/housing_test.csv -m artifacts/ --log-path logs/log.txt
python -m pytest tests