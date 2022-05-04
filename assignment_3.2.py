import os
import tarfile
import warnings
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from six.moves import urllib


# mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 127.0.0.2 --port 5000
remote_server_uri = "http://127.0.0.2:5000" # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env
mlflow.tracking.get_tracking_uri()


exp_name = "Housing_Price_Prediction_Experiment"
mlflow.set_experiment(exp_name)


with mlflow.start_run(run_name='PARENT_RUN') as parent_run:
    mlflow.log_param("parent", "yes")
    with mlflow.start_run(run_name='MATRIX_EVALUATION', nested=True) as child_run:
        mlflow.log_param("child", "yes")
        def eval_metrics(actual, pred):
            # compute relevant metrics
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            return rmse, mae, r2
        
    
    with mlflow.start_run(run_name='LOADING_DATA', nested=True) as child_run:
        mlflow.log_param("child", "yes")
        def load_data():
    
            DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
            HOUSING_PATH = os.path.join("datasets", "housing")
            HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

            def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
                if not os.path.isdir(housing_path):
                    os.makedirs(housing_path)
                tgz_path = os.path.join(housing_path, "housing.tgz")
                urllib.request.urlretrieve(housing_url, tgz_path)
                housing_tgz = tarfile.open(tgz_path)
                housing_tgz.extractall(path=housing_path)
                housing_tgz.close()

            def load_housing_data(housing_path=HOUSING_PATH):
                csv_path = os.path.join(housing_path, "housing.csv")
                return pd.read_csv(csv_path)
            fetch_housing_data()
            housing = load_housing_data()
            return housing
        
    with mlflow.start_run(run_name='SAMPLING_DATA', nested=True) as child_run:
        mlflow.log_param("child", "yes")
        #categorizing median income to perform stratified sampling.
        housing = load_data()
        housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6.,np.inf],
                                   labels=[1, 2, 3, 4, 5])

        #Performing stratified sampling.

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        housing = strat_train_set.drop("median_house_value", axis=1)
        housing_labels = strat_train_set["median_house_value"].copy()

    with mlflow.start_run(run_name='CLEANING_DATA', nested=True) as child_run:
        mlflow.log_param("child", "yes")

        # Data cleaning

        imputer = SimpleImputer(strategy="median")
        housing_num = housing.drop("ocean_proximity", axis=1) # Dropped Ocean_Proximity as it is a non-numeric column.
        imputer.fit(housing_num)
        X = imputer.transform(housing_num)
        housing_tr = pd.DataFrame(X, columns=housing_num.columns)
        housing_cat = housing[["ocean_proximity"]]
        from sklearn.preprocessing import OneHotEncoder
        cat_encoder = OneHotEncoder()
        housing_cat_1hot = cat_encoder.fit_transform(housing_cat) # Creating Dummy clomns for non-numeric data

    with mlflow.start_run(run_name='CUSTOM_TRANSFORMER', nested=True) as child_run:
        mlflow.log_param("child", "yes")
        from sklearn.base import BaseEstimator, TransformerMixin
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
            def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
                self.add_bedrooms_per_room = add_bedrooms_per_room
            def fit(self, X, y=None):
                return self # nothing else to do
            def transform(self, X, y=None):
                rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
                population_per_household = X[:, population_ix] / X[:, households_ix]
                if self.add_bedrooms_per_room:
                    bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                    return np.c_[X, rooms_per_household, population_per_household,
                    bedrooms_per_room]
                else:
                    return np.c_[X, rooms_per_household, population_per_household]

        attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
        housing_extra_attribs = attr_adder.transform(housing.values)

    with mlflow.start_run(run_name='PIPELINE', nested=True) as child_run:
        mlflow.log_param("child", "yes")

        #Transformation Pipelines

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        num_pipeline = Pipeline([
         ('imputer', SimpleImputer(strategy="median")), ## Imputing missing values
         ('attribs_adder', CombinedAttributesAdder()),  ## combining attributes to make them logical
         ('std_scaler', StandardScaler()),              ## Standardising features
         ])
        housing_num_tr = num_pipeline.fit_transform(housing_num)

        #Full pipeline for both categorical and numerical data columns 

        from sklearn.compose import ColumnTransformer
        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
         ("num", num_pipeline, num_attribs),
         ("cat", OneHotEncoder(), cat_attribs),
         ])
        housing_prepared = full_pipeline.fit_transform(housing)
    with mlflow.start_run(run_name='TRAINING_MODEL', nested=True) as child_run:
        mlflow.log_param("child", "yes")
        # train a model with given parameters

        warnings.filterwarnings("ignore")
        np.random.seed(40) 

        # Making a Random Forest Model


        with mlflow.start_run(run_name='RANDOM_FOREST_MODEL', nested=True) as child_run:
            mlflow.log_param("child", "yes")
            from sklearn.ensemble import RandomForestRegressor
            forest_reg = RandomForestRegressor()
            forest_reg.fit(housing_prepared, housing_labels)
            housing_predictions = forest_reg.predict(housing_prepared)

            #Performing Grid Search

            from sklearn.model_selection import GridSearchCV
            param_grid = [
                {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
             ]
            forest_reg = RandomForestRegressor()
            grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                                       scoring='neg_mean_squared_error',
            return_train_score=True)
            grid_search.fit(housing_prepared, housing_labels)
            cvres = grid_search.cv_results_
            feature_importances = grid_search.best_estimator_.feature_importances_
            feature_importances

            extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
            cat_encoder = full_pipeline.named_transformers_["cat"]
            cat_one_hot_attribs = list(cat_encoder.categories_[0])
            attributes = num_attribs + extra_attribs + cat_one_hot_attribs
            sorted(zip(feature_importances, attributes), reverse=True)

            final_model = grid_search.best_estimator_
            X_test = strat_test_set.drop("median_house_value", axis=1)
            y_test = strat_test_set["median_house_value"].copy()
            X_test_prepared = full_pipeline.transform(X_test)
            final_predictions = final_model.predict(X_test_prepared)

            # Evaluate Metrics

            (rmse, mae, r2) = eval_metrics(y_test, final_predictions)

            print("\n-- Random Forest Model --\n")
            print("  RMSE: %s" % rmse)
            print("  MAE: %s" % mae)
            print("  R2: %s" % r2)

            # metrics, and model to MLflow
            mlflow.log_param("Model", "Random Forest Model")
            mlflow.log_metric(key="rmse", value=rmse)
            mlflow.log_metrics({"mae": mae, "r2": r2})
            print("Save to: {}".format(mlflow.get_artifact_uri()))

            mlflow.sklearn.log_model(forest_reg, "model")

        with mlflow.start_run(run_name='LINEAR_REGRESSION_MODEL', nested=True) as child_run:
            mlflow.log_param("child", "yes")
            from sklearn.linear_model import LinearRegression
            lin_reg = LinearRegression()
            lin_reg.fit(housing_prepared, housing_labels)
            from sklearn.metrics import mean_squared_error
            housing_predictions = lin_reg.predict(housing_prepared)

            # Evaluate Metrics

            (rmse, mae, r2) = eval_metrics(housing_labels, housing_predictions)

            print("\n-- Linear Regression Model --\n")
            print("  RMSE: %s" % rmse)
            print("  MAE: %s" % mae)
            print("  R2: %s" % r2)

            # metrics, and model to MLflow
            mlflow.log_param("Model", "Linear Regression Model")
            mlflow.log_metric(key="rmse", value=rmse)
            mlflow.log_metrics({"mae": mae, "r2": r2})
            print("Save to: {}".format(mlflow.get_artifact_uri()))

            mlflow.sklearn.log_model(lin_reg, "model")
