"""
This module contains helper functions for ingestion of data.
Running this standalone downloads the housing data and stores preprocessed copies of it in the specified folders.
"""
import os
import tarfile
from argparse import ArgumentParser, Namespace
from logging import Logger

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

from housing_price.logger import configure_logger


def parse_args() -> Namespace:
    """Commandline argument parser for standalone run.

    Returns
    -------
    arparse.Namespace
        Commandline arguments. Contains keys: ["raw": str,
         "processed": str,
         "log_level": str,
         "no_console_log": bool,
         "log_path": str]
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--raw",
        type=str,
        default="data/raw/",
        help="Path to raw dataset.",
    )
    parser.add_argument(
        "-p",
        "--processed",
        type=str,
        default="data/processed/",
        help="Path to processed dataset.",
    )
    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument("--log-path", type=str, default="")
    return parser.parse_args()


def fetch_housing_data(housing_url: str, housing_path: str) -> None:
    """Function to download and extract housing data.

    Parameters
    ----------
    housing_url : str
        Url to download the housing data from.
    housing_path : str
        Path to store the raw csv files after extraction.
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    os.remove(tgz_path)


def stratified_shuffle_split(
    base_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Does stratified shuffle split on "income_cat" attribute of housing data.

    Parameters
    ----------
    base_df : pd.DataFrame
        The dataframe to be split.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        [train_dataset, test_dataset]
    """
    base_df["income_cat"] = pd.cut(
        base_df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(base_df, base_df["income_cat"]):
        strat_train_set = base_df.loc[train_index]
        strat_test_set = base_df.loc[test_index]

    for set_ in (strat_test_set, strat_train_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return (strat_train_set, strat_test_set)


def pre_process_data(
    df: pd.DataFrame, imputer: SimpleImputer = None
) -> tuple[pd.DataFrame, SimpleImputer]:
    """Preprocesses the given dataframe. Imputes missing values with median.
    Replaces categorical column "ocean_proximity" with onehot dummy variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to preprocess.
    imputer : SimpleImputer, optional
        Imputer that imputes missing values, by default None.
        If None, new imputer is created and fit to the given dataframe.

    Returns
    -------
    tuple[pd.DataFrame, SimpleImputer]
        Index 0 is the preprocessed dataframe.
        Index 1 is the SimpleImputer passed or fit on the dataframe if None is passed.
    """
    df = pd.get_dummies(df, columns=["ocean_proximity"])

    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        imputer.fit(df)

    data = imputer.transform(df)
    df = pd.DataFrame(data, columns=df.columns, index=df.index)

    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    return (df, imputer)


def run(args: Namespace, logger: Logger) -> None:
    """Does all the ingesting work (fetching, splitting, preprocessing).
    Gets called if this module is run standalone.

    Parameters
    ----------
    args : Namespace
        Commandline arguments from parse_args.
    logger : Logger
        Logger to log the state while running.
    """
    DOWNLOAD_ROOT = (
        "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    )
    HOUSING_PATH = args.raw
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    logger.debug("Fetched housing data.")

    housing_df = pd.read_csv(os.path.join(args.raw, "housing.csv"))
    train_set, test_set = stratified_shuffle_split(housing_df)

    logger.debug("Preprocessing...")
    train_set, imputer = pre_process_data(train_set)
    test_set, _ = pre_process_data(test_set, imputer)
    logger.debug("Preprocessing finished.")

    logger.debug("Saving datasets.")
    os.makedirs(args.processed, exist_ok=True)

    train_path = os.path.join(args.processed, "housing_train.csv")
    train_set.to_csv(train_path)
    logger.debug(f"Preprocessed train datasets stored at {train_path}.")

    test_path = os.path.join(args.processed, "housing_test.csv")
    test_set.to_csv(test_path)
    logger.debug(f"Preprocessed test datasets stored at {test_path}.")


if __name__ == "__main__":
    args = parse_args()
    logger = configure_logger(
        log_level=args.log_level,
        log_file=args.log_path,
        console=not args.no_console_log,
    )

    run(args, logger)
