import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params_path = os.path.join(BASE_DIR, "params.yaml")

params = yaml.safe_load(open("params.yaml"))

df = pd.read_csv("data/processed/train.csv")

target = params["train"]["target"]
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["train"]["test_size"]
)

with mlflow.start_run():

    model = RandomForestRegressor(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"]
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
#   rmse = mean_squared_error(y_test, preds, squared=False) not compatible with older versions of sklearn
    rmse = mean_squared_error(y_test, preds) ** 0.5

    mlflow.log_params(params["model"])
    mlflow.log_metric("rmse", rmse)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    mlflow.log_artifact("models/model.pkl")