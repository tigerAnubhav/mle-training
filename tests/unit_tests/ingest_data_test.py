"""
This module contains unit test for src/housing_price/ingest_data.py
"""
import os

import housing_price.ingest_data as data
import pandas as pd
from housing_price.logger import configure_logger

args = data.parse_args()
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = args.raw
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def test_parse_args():
    """
    Tests parse_args function.
    """
    assert args.raw == "data/raw/"
    assert args.processed == "data/processed/"
    assert args.log_level == "DEBUG"
    assert not args.no_console_log
    assert args.log_path == ""


def test_fetch_data():
    """
    Tests fetch_housing_data function.
    """
    data.fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    assert not os.path.isfile(f"{args.raw}/housing.tgz")
    assert os.path.isfile(f"{args.raw}/housing.csv")


def test_split():
    """
    Tests stratified_shuffle_split function.
    """
    housing_df = pd.read_csv(f"{args.raw}/housing.csv")
    train_set, test_set = data.stratified_shuffle_split(housing_df)
    assert len(train_set) == len(housing_df) * 0.8
    assert len(test_set) == len(housing_df) * 0.2
    assert "income_cat" not in train_set.columns
    assert "income_cat" not in test_set.columns


def test_preprocess():
    """
    Tests pre_process_data function.
    """
    housing_df = pd.read_csv(f"{args.raw}/housing.csv")
    train_set, test_set = data.stratified_shuffle_split(housing_df)
    train_set, imputer = data.pre_process_data(train_set)
    test_set, _ = data.pre_process_data(test_set)
    cats = housing_df["ocean_proximity"].unique()

    assert not train_set.isna().sum().sum()
    assert "ocean_proximity" not in train_set.columns
    assert "ocean_proximity" not in test_set.columns
    assert "rooms_per_household" in train_set.columns
    assert "rooms_per_household" in test_set.columns
    assert "population_per_household" in train_set.columns
    assert "population_per_household" in test_set.columns
    assert "bedrooms_per_room" in train_set.columns
    assert "bedrooms_per_room" in test_set.columns
    for cat in cats:
        assert f"ocean_proximity_{cat}" in train_set.columns
        assert f"ocean_proximity_{cat}" in test_set.columns
