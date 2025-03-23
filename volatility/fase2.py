import pandas as pd

# Carica il dataset
df = pd.read_csv("stock_data.csv", index_col=0)

# Visualizza le prime righe
print(df.head())

# Controlla se ci sono valori mancanti
print("\nValori mancanti per colonna:\n", df.isnull().sum())


# Rimuovi righe con valori mancanti
# df = df.dropna() # Rimuove tutte le righe con valori mancanti

# Come sostituire i valori mancanti con la media
# df = df.fillna(df.mean())  # Sost
# df = df.fillna(df.median())  # Sost

# Come sostituire i valori mancanti con valore precedente o successivo
# df = df.fillna(method="ffill")  # Sostituisce con il valore precedente
# df = df.fillna(method="bfill")  # Sostituisce con il valore successivo