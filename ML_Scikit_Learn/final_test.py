import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocess import build_preprocess_pipeline



df = pd.read_csv("data/AB_NYC_2019.csv")


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


_, X_test, y_test = build_preprocess_pipeline(test_df)



best_model = joblib.load("results/best_model.pkl")




y_pred = best_model.predict(X_test)


#metryki koncowe
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Wyniki na zbiorze testowym:\n")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}\n") 



with open("results/final_test_score.txt", "w") as f:
    
    f.write(f"MAE:  {mae:.3f}\n")
    f.write(f"RMSE: {rmse:.3f}\n")

print("Wyniki zapisane w results/final_test_score.txt")
