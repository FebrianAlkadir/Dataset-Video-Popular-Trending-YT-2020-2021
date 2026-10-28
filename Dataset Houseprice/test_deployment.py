import json
import unittest
from pathlib import Path

from testing_deploy import app


BASE_DIR = Path(__file__).resolve().parent


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        with (BASE_DIR / "data.json").open(encoding="utf-8") as file:
            self.payload = json.load(file)

    def test_predicts_json_examples(self):
        response = self.client.post("/predict", json=self.payload)

        self.assertEqual(response.status_code, 200)
        predictions = response.get_json()["prediction"]
        self.assertEqual(len(predictions), len(self.payload["data"]))
        self.assertTrue(all(isinstance(value, float) for value in predictions))
        self.assertAlmostEqual(predictions[0], 0.5415485115928796, places=5)

    def test_rejects_missing_feature(self):
        payload = {"data": [self.payload["data"][0][:-1]]}

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_rejects_out_of_range_category_code(self):
        vector = self.payload["data"][0].copy()
        vector[-1] = 999
        response = self.client.post("/predict", json={"data": [vector]})

        self.assertEqual(response.status_code, 400)

    def test_rejects_non_numeric_values(self):
        vector = self.payload["data"][0].copy()
        vector[0] = "not-a-number"
        response = self.client.post("/predict", json={"data": [vector]})

        self.assertEqual(response.status_code, 400)

    def test_rejects_empty_data(self):
        response = self.client.post("/predict", json={"data": []})

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
