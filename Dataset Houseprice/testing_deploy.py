from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.joblib"


def decode_legacy_vectors(
    vectors: list[list[Any]],
    bundle: dict[str, Any],
) -> pd.DataFrame:
    numeric_columns = bundle["numeric_columns"]
    categorical_columns = bundle["categorical_columns"]
    feature_columns = bundle["feature_columns"]
    expected_length = len(feature_columns)
    rows: list[dict[str, Any]] = []

    for row_index, vector in enumerate(vectors):
        if not isinstance(vector, list) or len(vector) != expected_length:
            raise ValueError(
                f"Baris {row_index} harus berupa array dengan "
                f"{expected_length} fitur."
            )

        try:
            values = np.asarray(vector, dtype=float)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"Baris {row_index} berisi nilai non-numerik.") from error
        if not np.isfinite(values).all():
            raise ValueError(f"Baris {row_index} berisi nilai tidak valid.")

        decoded: dict[str, float] = {}
        numeric_length = len(numeric_columns)
        numeric_values = values[:numeric_length]
        for column, value in zip(numeric_columns, numeric_values, strict=True):
            decoded[column] = float(value)

        category_codes = values[numeric_length:]
        for column, code in zip(categorical_columns, category_codes, strict=True):
            labels = bundle["categorical_values"][column]
            if not float(code).is_integer():
                raise ValueError(
                    f"Kode kategori {column!r} pada baris {row_index} harus bilangan bulat."
                )
            category_index = int(code)
            if category_index < 0 or category_index >= len(labels):
                raise ValueError(
                    f"Kode kategori {column!r} pada baris {row_index} "
                    "tidak tersedia pada data training."
                )
            decoded[column] = float(category_index)
        rows.append(decoded)

    return pd.DataFrame.from_records(rows, columns=feature_columns)


def create_app(model_path: Path = MODEL_PATH) -> Flask:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    app = Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def predict():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or "data" not in body:
            return jsonify({"error": "Body JSON harus berisi kunci 'data'."}), 400

        vectors = body["data"]
        if not isinstance(vectors, list) or not vectors:
            return jsonify({"error": "'data' harus berupa array berisi minimal satu baris."}), 400

        try:
            features = decode_legacy_vectors(vectors, bundle)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        predictions = model.predict(features)
        return jsonify({"prediction": np.asarray(predictions, dtype=float).tolist()})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=False)
