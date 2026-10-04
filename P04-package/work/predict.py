"""Predict one delivery, from the command line."""

import pandas as pd
from pathlib import Path

from delivery import load_model


def main():
    model = load_model(Path(__file__).parent / "model.joblib")

    X = pd.DataFrame([{
        "distance_km": 7,
        "prep_time_min": 25,
        "traffic_level": 3,
        "rain": 0
    }])

    prediction = model.predict(X)[0]

    print(f"PREDICTION: {prediction:.1f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
