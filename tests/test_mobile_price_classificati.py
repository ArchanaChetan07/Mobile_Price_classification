import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class TestMobileDataProcessing:

    def test_feature_columns_present(self):
        df = pd.DataFrame({
            "battery_power": [1000], "ram": [2048],
            "px_height": [1280], "px_width": [720],
            "price_range": [1]
        })
        required = ["battery_power", "ram", "px_height", "px_width"]
        for col in required:
            assert col in df.columns

    def test_price_range_valid(self):
        prices = pd.Series([0, 1, 2, 3])
        assert prices.min() == 0
        assert prices.max() == 3
        assert set(prices.unique()) == {0, 1, 2, 3}

    def test_ram_values_positive(self):
        df = pd.DataFrame({"ram": [256, 512, 1024, 2048, 4096]})
        assert (df["ram"] > 0).all()

    def test_feature_scaling(self):
        X = np.array([[1000, 2048], [2000, 4096], [500, 1024]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert abs(X_scaled[:, 0].mean()) < 1e-10

    def test_no_nulls_in_features(self):
        df = pd.DataFrame({"battery_power": [1000, None, 2000], "ram": [2048, 4096, 1024]})
        df_clean = df.dropna()
        assert df_clean.isnull().sum().sum() == 0


class TestPriceClassification:

    def test_multiclass_predictions_valid(self):
        np.random.seed(42)
        X = np.random.rand(200, 8)
        y = np.random.randint(0, 4, 200)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert set(preds).issubset({0, 1, 2, 3})

    def test_accuracy_above_random(self):
        np.random.seed(42)
        X = np.random.rand(400, 10)
        y = (X[:, 0] * 4).astype(int).clip(0, 3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        assert acc > 0.25
