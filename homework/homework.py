import os
import json
import pickle
import gzip
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.compose import ColumnTransformer


def load_train_test_data():
    train_df = pd.read_csv("../files/input/train_data.csv.zip", compression="zip")
    test_df = pd.read_csv("../files/input/test_data.csv.zip", compression="zip")
    return train_df, test_df


def clean_vehicle_data(df):
    df = df.copy()
    df['Age'] = 2021 - df['Year']
    df.drop(columns=['Car_Name', 'Year'], inplace=True)
    return df


def split_features_target(df):
    return df.drop(columns=['Selling_Price']), df['Selling_Price']


def build_regression_pipeline():
    categorical = ['Fuel_Type', 'Selling_type', 'Transmission']
    numerical = ['Present_Price', 'Driven_kms', 'Owner', 'Age']

    preprocessing = ColumnTransformer(transformers=[
        ('categorical_encoder', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('scaler', MinMaxScaler(), numerical),
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessing),
        ('feature_selector', SelectKBest(score_func=f_regression)),
        ('regressor', LinearRegression())
    ])

    return pipeline


def perform_grid_search(pipeline, x_train, y_train):
    param_grid = {
        'feature_selector__k': [5, 7]
    }

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    cv = KFold(n_splits=10, shuffle=True, random_state=123)

    grid_search = GridSearchCV(pipeline, param_grid, scoring=scorer, cv=cv, n_jobs=-1)
    grid_search.fit(x_train, y_train)
    return grid_search


def calculate_regression_metrics(dataset_name, y_true, y_pred):
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mad": mean_absolute_error(y_true, y_pred)
    }


def save_model_to_gzip(estimator, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wb") as f:
        pickle.dump(estimator, f)


def write_metrics_to_json(metrics_list, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for metrics in metrics_list:
            f.write(json.dumps(metrics) + "\n")


def run_regression_pipeline():
    train_df, test_df = load_train_test_data()
    train_df = clean_vehicle_data(train_df)
    test_df = clean_vehicle_data(test_df)

    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    pipeline = build_regression_pipeline()
    model = perform_grid_search(pipeline, x_train, y_train)

    save_model_to_gzip(model, "../files/models/model.pkl.gz")

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics = [
        calculate_regression_metrics("train", y_train, y_train_pred),
        calculate_regression_metrics("test", y_test, y_test_pred)
    ]

    write_metrics_to_json(metrics, "../files/output/metrics.json")


if __name__ == "__main__":
    run_regression_pipeline()
