from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train.csv"
MODEL_PATH = BASE_DIR / "house_price_model.joblib"
TARGET = "SalePrice"
RANDOM_STATE = 1


def prepare_training_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    if TARGET not in data.columns:
        raise ValueError(f"Dataset harus memiliki kolom target {TARGET!r}.")

    missing_counts = data.isna().sum()
    dropped_columns = missing_counts[missing_counts >= 1000].index.tolist()
    retained = data.drop(columns=dropped_columns).copy()
    numeric_columns = retained.select_dtypes(include="number").columns.tolist()
    categorical_columns = retained.select_dtypes(exclude="number").columns.tolist()
    if TARGET not in numeric_columns:
        raise ValueError(f"Kolom target {TARGET!r} harus berupa numerik.")

    numeric_medians = retained[numeric_columns].median()
    retained[numeric_columns] = retained[numeric_columns].fillna(numeric_medians)
    categorical_modes: dict[str, Any] = {}
    for column in categorical_columns:
        modes = retained[column].mode(dropna=True)
        if modes.empty:
            raise ValueError(f"Kolom kategori {column!r} tidak memiliki nilai.")
        categorical_modes[column] = modes.iloc[0]
        retained[column] = retained[column].fillna(categorical_modes[column])

    quartiles_1 = retained[numeric_columns].quantile(0.25)
    quartiles_3 = retained[numeric_columns].quantile(0.75)
    iqr = quartiles_3 - quartiles_1
    inlier_mask = ~(
        (retained[numeric_columns] < quartiles_1 - 1.5 * iqr)
        | (retained[numeric_columns] > quartiles_3 + 1.5 * iqr)
    ).any(axis=1)
    filtered = retained.loc[inlier_mask].copy()
    if filtered.empty:
        raise ValueError("Tidak ada baris tersisa setelah penanganan outlier.")

    scaler = StandardScaler().fit(filtered[numeric_columns])
    numeric_mean = {
        column: float(scaler.mean_[index])
        for index, column in enumerate(numeric_columns)
    }
    numeric_scale = {
        column: float(scaler.scale_[index])
        for index, column in enumerate(numeric_columns)
    }
    standardized = pd.DataFrame(
        scaler.transform(filtered[numeric_columns]),
        columns=numeric_columns,
        index=filtered.index,
    )

    category_values: dict[str, list[str]] = {}
    for column in categorical_columns:
        encoder = LabelEncoder()
        standardized[column] = encoder.fit_transform(filtered[column].astype(str))
        category_values[column] = encoder.classes_.tolist()

    feature_columns = [
        column for column in standardized.columns if column != TARGET
    ]
    features = standardized[feature_columns].astype(float)
    target = standardized[TARGET]
    codec = {
        "target": TARGET,
        "numeric_columns": [
            column for column in numeric_columns if column != TARGET
        ],
        "categorical_columns": categorical_columns,
        "feature_columns": feature_columns,
        "numeric_mean": numeric_mean,
        "numeric_scale": numeric_scale,
        "categorical_values": category_values,
        "dropped_columns": dropped_columns,
        "numeric_medians": numeric_medians.to_dict(),
        "categorical_modes": categorical_modes,
        "target_mean": numeric_mean[TARGET],
        "target_scale": numeric_scale[TARGET],
        "filtered_row_indices": filtered.index.tolist(),
    }
    return features, target, codec


def train_and_evaluate(
    train_path: Path = TRAIN_PATH,
    model_path: Path = MODEL_PATH,
) -> dict[str, Any]:
    data = pd.read_csv(train_path)
    features, target, codec = prepare_training_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )

    candidates = {
        "Baseline (median)": DummyRegressor(strategy="median"),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
    }

    cv_results: dict[str, float] = {}
    for name, estimator in candidates.items():
        negative_mse = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )
        cv_results[name] = float(np.sqrt(-negative_mse).mean())

    selected_name = min(cv_results, key=cv_results.get)
    final_model = candidates[selected_name]
    final_model.fit(X_train, y_train)
    scaled_predictions = final_model.predict(X_test)
    predictions = scaled_predictions * codec["target_scale"] + codec["target_mean"]
    actual_values = y_test * codec["target_scale"] + codec["target_mean"]
    holdout_metrics = {
        "mae": float(mean_absolute_error(actual_values, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predictions))),
        "r2": float(r2_score(actual_values, predictions)),
        "scaled_rmse": float(np.sqrt(mean_squared_error(y_test, scaled_predictions))),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "model_name": selected_name,
            **codec,
        },
        model_path,
    )
    return {
        "cv_rmse_scaled_saleprice": cv_results,
        "selected_model": selected_name,
        "holdout_metrics": holdout_metrics,
        "model_path": str(model_path),
        "training_rows": len(X_train),
        "feature_count": len(codec["feature_columns"]),
        "deployment_vector_length": len(codec["feature_columns"]),
    }


if __name__ == "__main__":
    results = train_and_evaluate()
    print(f"CV RMSE (SalePrice terstandardisasi): {results['cv_rmse_scaled_saleprice']}")
    print(f"Selected model: {results['selected_model']}")
    print(f"Holdout metrics: {results['holdout_metrics']}")
    print(f"Deployment vector length: {results['deployment_vector_length']}")
    print(f"Saved model: {results['model_path']}")
