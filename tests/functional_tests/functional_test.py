"""Functional test for the whole housing_price package.
"""
import os
from glob import glob

import pytest


def test_ingest():
    """Tests ingest_data.py module."""
    raw = "tests/temp/data/raw/"
    processed = "tests/temp/data/processed/"
    os.system(
        f"python src/housing_price/ingest_data.py --raw {raw} --processed {processed}"
    )
    assert os.path.isfile(f"{raw}/housing.csv")
    assert os.path.isfile(f"{processed}/housing_train.csv")
    assert os.path.isfile(f"{processed}/housing_test.csv")


@pytest.mark.skipif(
    os.path.isfile("tests/temp/artifacts/LinearRegression.pkl"),
    reason="no need to retest if last test results still there",
)
def test_train():
    """Tests train.py module."""
    models = "tests/temp/artifacts/"
    dataset = "tests/temp/data/processed/housing_train.csv"
    os.system(f"python src/housing_price/train.py -d {dataset} -m {models}")
    assert os.path.isfile(f"{models}/LinearRegression.pkl")
    assert os.path.isfile(f"{models}/RandomForestRegressor.pkl")
    assert os.path.isfile(f"{models}/DecisionTreeRegressor.pkl")


def test_score(cleanup):
    """Tests score.py module."""
    models = "tests/temp/artifacts/"
    dataset = "tests/temp/data/processed/housing_test.csv"
    log_file = "tests/temp/log_file.txt"

    os.system(
        f"python src/housing_price/score.py -d {dataset} -m {models} --mae --rmse --log-path {log_file}"
    )

    with open(log_file, "r") as f:
        lines = f.readlines()
    pkls = glob(f"{models}/*.pkl")

    assert len(lines) == len(pkls) * 4
    assert "DecisionTreeRegressor" in lines[0]
    assert "LinearRegression" in lines[4]
    assert "RandomForestRegressor" in lines[8]
    assert lines[1].startswith("R2 score")
    assert lines[5].startswith("R2 score")
    assert lines[9].startswith("R2 score")
    assert lines[2].startswith("RMSE")
    assert lines[6].startswith("RMSE")
    assert lines[10].startswith("RMSE")
    assert lines[3].startswith("MAE")
    assert lines[7].startswith("MAE")
    assert lines[11].startswith("MAE")


@pytest.fixture()
def cleanup():
    yield
    os.system("rm -rf tests/temp/")
