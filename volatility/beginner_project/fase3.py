# In this snippet, we load the dataset we saved in the previous phase, 
# apply standardization using the StandardScaler from scikit-learn, 
# and save the normalized dataset to a new CSV file. 
# The StandardScaler scales each feature to have a mean of 0 and a standard deviation of 1. 
# This is a common preprocessing step in machine learning pipelines to ensure that all features have the same scale and are centered around zero.

from sklearn.preprocessing import StandardScaler
import pandas as pd

# Carica i dati
df = pd.read_csv("stock_data.csv", index_col=0)

# Creiamo lo scaler e applichiamo la standardizzazione
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Stampiamo le prime righe per vedere la differenza
print(df_scaled.head())

# Salviamo il dataset normalizzato
df_scaled.to_csv("stock_data_normalized.csv")
