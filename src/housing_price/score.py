"""
This module contains helper functions to score the models.
Can be run standalone with commandline arguments for models and the datasets to score them on.
"""
import os
import pickle
from argparse import ArgumentParser, Namespace
from glob import glob
from logging import Logger

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from housing_price.logger import configure_logger


def parse_args() -> Namespace:
    """Commandline argument parser for standalone run.

    Returns
    -------
    arparse.Namespace
        Commandline arguments. Contains keys: ["models": str,
         "dataset": str,
         "rmse": bool,
         "mae": bool,
         "log_level": str,
         "no_console_log": bool,
         "log_path": str]
    """
    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--models",
        type=str,
        default="artifacts/",
        help="Directory where the models are stored.",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="data/processed/housing_test.csv",
        help="Path to test dataset csv file.",
    )

    parser.add_argument("--rmse", action="store_true", help="Show RMSE.")
    parser.add_argument("--mae", action="store_true", help="Show MAE.")

    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument("--log-path", type=str, default="")

    return parser.parse_args()


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Loads dataset and splits features and labels.

    Parameters
    ----------
    path : str
        Path to training dataset csv file.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Index 0 is the testing features dataframe.
        Index 1 is the testing labels series.
    """
    df = pd.read_csv(path)
    y = df["median_house_value"].copy(deep=True)
    X = df.drop(["median_house_value"], axis=1)
    return (X, y)


def load_models(path: str) -> list[sklearn.base.BaseEstimator]:
    """Loads models from given directory path.

    Parameters
    ----------
    path : str
        Path to directory with model pkl files.

    Returns
    -------
    list[sklearn.base.BaseEstimator]
        List of models loaded from pkl files in directory.
    """
    paths = glob(f"{path}/*.pkl")
    paths = sorted(paths)
    models = []

    for path in paths:
        if os.path.isfile(path):
            model = pickle.load(open(path, "rb"))
            models.append(model)

    return models


def score_model(
    model: sklearn.base.BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    args: Namespace,
) -> dict:
    """Scores given model on given data.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Estimator to score.
    X : pd.DataFrame
        Input features dataframe.
    y : pd.Series
        Ground truth labels.
    args : Namespace
        Command line arguments. Used to determine which scores to calculate.

    Returns
    -------
    dict
        Contains calculated scores.

    """
    scores = {}
    scores["R2 score"] = model.score(X, y)
    y_hat = model.predict(X)

    if args.rmse:
        rmse = np.sqrt(mean_squared_error(y, y_hat))
        scores["RMSE"] = rmse

    if args.mae:
        mae = mean_absolute_error(y, y_hat)
        scores["MAE"] = mae

    return scores


def run(args: Namespace, logger: Logger) -> None:
    """Runs the whole scoring process according to the given commandline arguments.

    Parameters
    ----------
    args : Namespace
        Commandline arguments from parse_args.
    logger : Logger
        Logs the outputs.
    """
    X, y = load_data(args.dataset)

    models = load_models(args.models)

    for model in models:
        model_name = type(model).__name__
        scores = score_model(model, X, y, args)
        logger.debug(f"Model: {model_name}")
        for k, v in scores.items():
            logger.debug(f"{k}: {v}")


if __name__ == "__main__":
    args = parse_args()
    logger = configure_logger(
        log_file=args.log_path,
        log_level=args.log_level,
        console=not args.no_console_log,
    )

    run(args, logger)
