
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.pipeline import Pipeline


from preprocess import build_preprocess_pipeline



df = pd.read_csv("data/AB_NYC_2019.csv")



train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)



full_pipeline, X_train, y_train = build_preprocess_pipeline(train_df)

# Pipeline zwraca X i y gotowe do trenowania modelu


models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=50, random_state=42),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5)
}



print("Porownanie modeli 5-fold CV\n")

results = {}

for name, model in models.items():
    
    
    print(f"Testowanie modelu: {name}")

    # preprocessing + model
    model_pipeline = Pipeline([
        
        ("preprocess", full_pipeline),
        ("model", model)
        
    ])

    start = time.time()

    # MAE = mean absolute error(sredni blad wzgledny)
    
    scores = cross_val_score(
        model_pipeline,
        X_train,
        y_train,
        scoring="neg_mean_absolute_error",
        cv=3
    )

    end = time.time()

    mean_score = -scores.mean()  # odwracamy znak, zeby MAE bylo dodatnie
    std_score = scores.std()

    results[name] = mean_score

    print(f"Sredni MAE: {mean_score:.3f}")
    print(f"Odchylenie: {std_score:.3f}")
    print(f"Czas: {end - start:.2f} s\n")


# Wybieramy najlepszy model
    
    
best_model_name = min(results, key=results.get)
best_mae = results[best_model_name]

print("=== Najlepszy model ===")
print(f"{best_model_name}  (MAE = {best_mae:.3f})")


# Zapisujemy wyniki do pliku
with open("results/model_comparison.txt", "w") as f:
    
    for name, score in results.items():
        
        f.write(f"{name}: MAE = {score:.3f}\n")
        
        
    f.write(f"\nNajlepszy model: {best_model_name} (MAE = {best_mae:.3f})\n")
