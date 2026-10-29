"""Train the P07 delivery-time model, log it to MLflow, register it,
and give version 1 the alias 'Production'.

Run from the folder where you want mlflow.db to live:
    python register_model.py
"""
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "delivery-time-model"
FEATURES = ["distance_km", "prep_time_min", "traffic_level", "rain"]

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("delivery-time")

# Same 600-row data as the notebook (same seed and formula).
rng = np.random.default_rng(42)
n = 600
distance_km = np.round(rng.uniform(0.5, 12.0, n), 2)
prep_time_min = np.round(rng.uniform(5, 30, n), 0)
traffic_level = rng.integers(1, 4, n)
rain = rng.binomial(1, 0.25, n)
delivery_min = np.round(
    6.0 + 3.1 * distance_km + 0.65 * prep_time_min + 4.2 * traffic_level
    + 5.5 * rain + rng.normal(0, 2.5, n), 1)
df = pd.DataFrame({"distance_km": distance_km, "prep_time_min": prep_time_min,
                   "traffic_level": traffic_level, "rain": rain,
                   "delivery_min": delivery_min})
df[FEATURES] = df[FEATURES].astype("float64")  # avoids MLflow int-schema warnings

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df["delivery_min"], test_size=0.2, random_state=42)

with mlflow.start_run(run_name="linreg-p07") as run:
    model = LinearRegression().fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(
        model, name="model",
        input_example=X_train.head(3),
        registered_model_name=MODEL_NAME,
    )
    run_id = run.info.run_id

client = MlflowClient()
version = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
client.set_registered_model_alias(MODEL_NAME, "Production", version)

print("\n==== write these down for your PDF ====")
print("Model name :", MODEL_NAME)
print("Version    :", version)
print("Alias      : Production")
print("Run ID     :", run_id)
print("MAE        :", round(mae, 2))
