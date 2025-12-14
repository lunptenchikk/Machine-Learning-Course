# Tuning hiperparametrow najlepszego modelu (RandomForestRegressor)

import pandas as pd
import numpy as np
import time
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


from preprocess import build_preprocess_pipeline



df = pd.read_csv("data/AB_NYC_2019.csv")



train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

full_pipeline, X_train, y_train = build_preprocess_pipeline(train_df)



rf = RandomForestRegressor(random_state=42)


# n_estimators: liczba drzew w lesie
# max_depth: maksymalna glebokosc drzewa
# max_features: liczba cech do rozwazenia przy kazdym podziale



param_distributions = {
    "model__n_estimators": [50, 100, 150, 200, 300],
    "model__max_depth": [5, 10, 20, 30, 40, None],
    "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7]
}



rf_pipeline = Pipeline([
    
    ("preprocess", full_pipeline),
    ("model", rf)
])



print("Rozpoczynamy tuning naszego modelu\n")

start = time.time()

search = RandomizedSearchCV(
    
    rf_pipeline,
    param_distributions=param_distributions,
    n_iter=15,  # sprawdzamy 15 kombinacji          
    
    scoring="neg_mean_absolute_error",
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

end = time.time()

print(f"\nTuning zakonczyl sie. Czas: {end - start:.2f} sekund\n")


# wyniki tuningu
best_params = search.best_params_

best_score = -search.best_score_

print("Wypisujemy najlepsze parametry:\n")
print(best_params)
print(f"Najlepsze MAE: {best_score:.3f}\n")



with open("results/tuning_results.txt", "w") as f:
    
    f.write("Najlepsze parametry:\n")
    for k, v in best_params.items():
        
        f.write(f"{k}: {v}\n")
        
    f.write(f"\nNajlepsze MAE: {best_score:.3f}\n")


# zapisujemy najlepszy model dla final_testu


joblib.dump(search.best_estimator_, "results/best_model.pkl")

print("Najlepszy model zapisany jako: results/best_model.pkl")
