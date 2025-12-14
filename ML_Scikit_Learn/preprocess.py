# preprocess.py

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer


# kolumny zbedne dla modelu
def drop_unused_columns(df):
    df = df.copy()
    cols_to_drop = ["id", "name", "host_id", "host_name", "last_review"]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    return df


# Nowe cechy, ktore dodajemy do zbioru
def add_new_features(df):
    df = df.copy()

    # "Zajetosc"
    if "availability_365" in df.columns:
        df["busy_ratio"] = 365 - df["availability_365"]

    # Wynajem dlugoterminowy
    if "minimum_nights" in df.columns:
        df["long_term"] = (df["minimum_nights"] > 30).astype(int)

    return df


# log transformacja dla cech skosnych(tzn. zeby zmniejszyc skosnosc rozkladu)
def log_transform(df):
    df = df.copy()

    skewed_cols = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365"
    ]

    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    return df


def build_preprocess_pipeline(df):
    df = df.copy()
    df = drop_unused_columns(df)

    # target
    y = df["price"]

    # wszystko oprocz ceny
    X = df.drop("price", axis=1)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([  #stosujemy najczestsza wartosc do wartosci kategorycznych
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    full_pipeline = Pipeline([
        ("add_features", FunctionTransformer(add_new_features, validate=False)),
        ("log_transform", FunctionTransformer(log_transform, validate=False)),
        ("preprocess", preprocessor)
    ])

    return full_pipeline, X, y


# czyli na koncu zwracamy pipeline, oraz X i y do trenowania modelu 