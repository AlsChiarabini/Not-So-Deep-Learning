import pandas as pd

# Carica i dati normalizzati
df = pd.read_csv("stock_data_normalized.csv", index_col=0)

# Creiamo la variabile target: 1 se sopra la mediana, 0 altrimenti
median_return = df["Return_1y"].median()
df["Target"] = (df["Return_1y"] > median_return).astype(int)

# Stampiamo per controllare
print(df[["Return_1y", "Target"]].head())

# Salviamo il dataset finale con la variabile target
df.to_csv("stock_data_final.csv")
