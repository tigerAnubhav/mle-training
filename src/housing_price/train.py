"""
This module contains helper functions to train models.
Can be run standalone with commandline arguments for dataset path and models directory.
    """
import os
import pickle
from argparse import ArgumentParser, Namespace
from logging import Logger

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from housing_price.logger import configure_logger


def parse_args() -> Namespace:
    """Commandline argument parser for standalone run.

    Returns
    -------
    arparse.Namespace
        Commandline arguments. Contains keys: ["models": str,
         "dataset": str,
         "log_level": str,
         "no_console_log": bool,
         "log_path": str]
    """
    parser = ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="data/processed/housing_train.csv",
        help="Path to training dataset csv file.",
    )

    parser.add_argument(
        "-m",
        "--models",
        type=str,
        default="artifacts/",
        help="Directory to store model pickles.",
    )

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
        Index 0 is the training features dataframe.
        Index 1 is the training labels series.
    """
    df = pd.read_csv(path)
    y = df["median_house_value"].copy(deep=True)
    X = df.drop(["median_house_value"], axis=1)
    return (X, y)


def save_model(
    model: sklearn.base.BaseEstimator, dir: str
) -> tuple[str, str]:
    """Saves the given model in given directory as pickle file.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Estimator to save.
    dir : str
        Directory to save in.

    Returns
    -------
    tuple[str, str]
        Index 0 is the name of the model.
        Index 1 is the path it is saved in.
    """
    os.makedirs(dir, exist_ok=True)
    model_name = type(model).__name__

    path = os.path.join(dir, f"{model_name}.pkl")
    with open(path, "wb") as file:
        pickle.dump(model, file)
    return (model_name, path)


def run(args: Namespace, logger: Logger) -> None:
    """Runs the whole training process according to given commandline arguments.

    Parameters
    ----------
    args : Namespace
        Commandline arguments from parse_args.
    logger : Logger
        Logs the outputs.
    """
    logger.info("Started training.")

    X, y = load_data(args.dataset)

    lr = LinearRegression()
    lr.fit(X, y)
    model_name, path = save_model(lr, args.models)
    logger.debug(f"{model_name} model saved in {path}.")

    dtree = DecisionTreeRegressor(random_state=42)
    dtree.fit(X, y)
    model_name, path = save_model(dtree, args.models)
    logger.debug(f"{model_name} model saved in {path}.")

    random_forest = RandomForestRegressor()
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]
    grid_search = GridSearchCV(
        random_forest,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        return_train_score=True,
    )
    grid_search.fit(X, y)
    model_name, path = save_model(grid_search.best_estimator_, args.models)
    logger.debug(f"{model_name} model saved in {path}.")

    logger.info("Done training.")


if __name__ == "__main__":
    args = parse_args()
    logger = configure_logger(
        log_level=args.log_level,
        console=not args.no_console_log,
        log_file=args.log_path,
    )

    run(args, logger)
