import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from preprocess import build_preprocess_pipeline



os.makedirs("results", exist_ok=True)



df = pd.read_csv("data/AB_NYC_2019.csv")

print("Dataset wczytany. Liczba rekordow:", len(df))


# przed normalizacja
num_cols = ["price", "minimum_nights", "number_of_reviews"]

plt.figure(figsize=(12, 4))


for i, col in enumerate(num_cols):
    
    plt.subplot(1, 3, i + 1)
    df[col].hist(bins=40)
    plt.title(f"Przed normalizacją: {col}")

plt.tight_layout()

plt.savefig("results/hist_before.png")

plt.close()

print("Zapisano histogramy PRZED transformacja : results/hist_before.png")


# po normalizacji (log + scaler)
df_log = df.copy()


for c in num_cols:
    df_log[c] = np.log1p(df_log[c])

scaler = StandardScaler()

scaled = scaler.fit_transform(df_log[num_cols])

plt.figure(figsize=(12, 4))

for i, col in enumerate(num_cols):
    
    
    plt.subplot(1, 3, i + 1)
    plt.hist(scaled[:, i], bins=40)
    plt.title(f"Po normalizacji: {col}")

plt.tight_layout()

plt.savefig("results/hist_after.png")

plt.close()

print("Zapisano histogramy PO transformacji: results/hist_after.png")


# analizujemy cechy slabo skorelowane z targetem
target = "price"


corr = df.corr(numeric_only=True)[target].abs().sort_values()


weak_features = corr[corr < 0.05].index.tolist()

print("\nSlabo skorelowane cechy:")

print(weak_features)

with open("results/weak_features.txt", "w") as f:
    f.write("Słabo skorelowane cechy:\n")
    for w in weak_features:
        f.write(w + "\n")


# usuwamy slabo skorelowane cechy
df_reduced = df.drop(columns=weak_features, errors="ignore")

train_df, test_df = train_test_split(df_reduced, test_size=0.2, random_state=42)

full_pipeline, X_train, y_train = build_preprocess_pipeline(train_df)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf_pipeline = Pipeline([
    ("preprocess", full_pipeline),
    ("model", rf)
])

print("\nTrenowanie bez slabych cech\n")

start = time.time()
rf_pipeline.fit(X_train, y_train)
end = time.time()

train_time = round(end - start, 2)

# Test-set
_, X_test, y_test = build_preprocess_pipeline(test_df)
y_pred = rf_pipeline.predict(X_test)

mae_reduced = mean_absolute_error(y_test, y_pred)

print(f"MAE po usunieciu slabych cech: {mae_reduced:.3f}")
print(f"Czas trenowania: {train_time} s\n")





with open("results/reduced_features_results.txt", "w") as f:
    
    f.write(f"SSlabe cechy: {weak_features}\n")
    
    f.write(f"MAE: {mae_reduced:.3f}\n")
    
    f.write(f"Czas trenowania: {train_time} s\n")

print("Wyniki zapisane w results/reduced_features_results.txt")

