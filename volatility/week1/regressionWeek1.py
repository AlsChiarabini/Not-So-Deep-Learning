import yfinance as yf
import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from io import StringIO
import requests
import os

filename = "sp500_data.csv"
if os.path.exists(filename):
    print(f"File {filename} trovato, non lo scarico di nuovo.")
else:
    print("🔄 File non trovato, scarico i dati...")

    # 📌 Scarichiamo la lista delle aziende S&P 500
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(StringIO(requests.get(url).text))
    sp500 = tables[0]  # Prima tabella con i simboli delle aziende

    tickers = [symbol.replace(".", "-") for symbol in sp500["Symbol"].head(200).tolist()]
    valid_tickers = []
    data = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')  # Dati storici di un anno
            if hist.empty:
                print(f"⚠️ No data for {ticker}, skipping...")
                continue
            info = stock.info

            # Creazione delle feature --> feature engineering
            data[ticker] = {
                "PE": info.get("trailingPE", np.nan),
                "PB": info.get("priceToBook", np.nan),
                "ROE": info.get("returnOnEquity", np.nan),
                "ROA": info.get("returnOnAssets", np.nan),
                "DebtToEquity": info.get("debtToEquity", np.nan),
                "Beta": info.get("beta", np.nan),
                "MarketCap": info.get("marketCap", np.nan),
                "DividendYield": info.get("dividendYield", np.nan),
                "52WeekChange": info.get("52WeekChange", np.nan),
                "Momentum_6m": hist["Close"].pct_change(min(len(hist), 126)).iloc[-1] if len(hist) > 126 else np.nan,
                "Volatility": hist["Close"].pct_change().std() * (252 ** 0.5),
                "Return_6m": hist["Close"].pct_change(min(len(hist), 126)).iloc[-1] if len(hist) > 126 else np.nan,
            }
            valid_tickers.append(ticker)
        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}, skipping...")
            continue

    df = pd.DataFrame.from_dict(data, orient="index")
    df.dropna(inplace=True)
    df.to_csv(filename)
    print(f"✅ Dati salvati in '{filename}' per {len(valid_tickers)} aziende su {len(tickers)} disponibili.")

# 2️⃣ Preprocessing dei dati (=preparo i dati per le tecniche di ML)
df = pd.read_csv("sp500_data.csv", index_col=0) # index_col=0 vuol dire che la prima colonna è l'indice
X = df.drop(columns=["Return_6m"])  # Tolgo il target ed il resto lo uso come features
y = df["Return_6m"]  # Nuovo target --> variabile dipendente, obiettivo da prevedere nel modello. I dati del ritorno a 6 mesi non entrano nel training, ma vengono usati per testare il modello
                    # modelli migliori prevedono il futuro senza dati da confrontare

scaler = StandardScaler() # Creo l'oggetto (da sklearn) che standardizza i dati
X_scaled = scaler.fit_transform(X) # Fit calcola la media e la deviazione standard per ogni feature e poi li usa per standardizzare i dati
                                   # X_scaled ora contiene la versione standardizzata di X
                                   # X' = (X - media) / stdev
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # 80% train, 20% test

''' PRIMO MODELLO DI RANDOMFOREST, SENZA OTTIMIIZZAZIONI
# 3️⃣ Creazione del modello (uso un Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42) # 100 alberi, modello pronto per l'addestramento 
model.fit(X_train, y_train) # addestro sui dati di train, alberi imparano a relazionare le feature con il target
y_pred = model.predict(X_test) # genera y_pred per gli x_test'
'''

# Uso randomizedSearch per stringere il campo di ricerca, poi GridSearchCV per ottimizzare i parametri
# 3️⃣ Creazione del modello (uso un Random Forest)
model = RandomForestRegressor(random_state=42)
param_dist = {
    'n_estimators': [50, 100, 200], # numero di alberi
    'max_depth': [None, 10, 20, 30], # profondità massima dell'albero
    'min_samples_split': [2, 5, 10], # numero minimo di campioni richiesti per dividere un nodo
    'min_samples_leaf': [1, 2, 4], # numero minimo di campioni richiesti per essere una foglia
}
random_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv = 5, scoring='r2', n_jobs = 1, random_state=42)
random_search.fit(X_train, y_train) # addestro sui dati di train, alberi imparano a relazionare le feature con il target
print("Migliori parametri trovati: ", random_search.best_params_)

best_params = random_search.best_params_ # prendo i migliori parametri trovati
param_grid = {
    'n_estimators': [best_params['n_estimators'] - 50, best_params['n_estimators'], best_params['n_estimators'] + 50],
    'max_depth': [best_params['max_depth'] - 10, best_params['max_depth'], best_params['max_depth'] + 10] if best_params['max_depth'] is not None else [None],
    'min_samples_split': [best_params['min_samples_split'], best_params['min_samples_split'] + 1, best_params['min_samples_split'] + 2],
    'min_samples_leaf': [best_params['min_samples_leaf'] - 1, best_params['min_samples_leaf'], best_params['min_samples_leaf'] + 1],
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=1)
grid_search.fit(X_train, y_train) # addestro sui dati di train, alberi imparano a relazionare le feature con il target
print("Migliori parametri trovati: ", grid_search.best_params_)

best_model = grid_search.best_estimator_ # prendo il miglior modello trovato
y_pred = best_model.predict(X_test) # genera y_pred per gli x_test'

# 4️⃣ Valutazione del modello
mae = mean_absolute_error(y_test, y_pred) # MAE, 1/n * Σ|y_i - y_pred_i|
r2 = r2_score(y_test, y_pred) # R^2, 1 - (Σ(y_i - y_pred_i)^2) / Σ(y_i - media)^2
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")

