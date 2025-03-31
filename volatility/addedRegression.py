import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error, r2_score
import requests
from io import StringIO

# Funzioni per calcolare RSI e MACD
def compute_rsi(series, period=14): # RSI = Relative Strength Index
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, slow=26, fast=12, signal=9): # MACD = Moving Average Convergence Divergence
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, min_periods=signal).mean()
    return macd.iloc[-1] - signal_line.iloc[-1]

# Ottenere la lista aggiornata delle aziende S&P 500 da Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(StringIO(requests.get(url).text))
sp500_df = tables[0]  # La prima tabella contiene i titoli

# Prendiamo i primi 200 tickers
tickers = [symbol.replace(".", "-") for symbol in sp500_df["Symbol"].head(200).tolist()]
valid_tickers = []  # Lista per tenere solo i ticker con dati validi
data = {}

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y") #hist = history

        # Se non ci sono dati, salta l'azienda
        if hist.empty:
            print(f"⚠️ No data for {ticker}, skipping...")
            continue

        info = stock.info # Informazioni sull'azienda

        # Creazione delle feature
        data[ticker] = {
            "PE": info.get("trailingPE", np.nan),
            "PB": info.get("priceToBook", np.nan),
            "ROE": info.get("returnOnEquity", np.nan),
            "ROA": info.get("returnOnAssets", np.nan),
            "DebtToEquity": info.get("debtToEquity", np.nan),
            "Beta": info.get("beta", np.nan),
            "MarketCap": info.get("marketCap", np.nan),
            "Momentum_6m": hist["Close"].pct_change(min(len(hist), 126)).iloc[-1] if len(hist) > 126 else np.nan,
            "Volatility": hist["Close"].pct_change().std() * (252 ** 0.5),
            "RSI": compute_rsi(hist["Close"]).iloc[-1] if len(hist) > 14 else np.nan,
            "MACD": compute_macd(hist["Close"]) if len(hist) > 26 else np.nan,
            "SMA_50": hist["Close"].rolling(window=50).mean().iloc[-1] if len(hist) > 50 else np.nan,
            "SMA_200": hist["Close"].rolling(window=200).mean().iloc[-1] if len(hist) > 200 else np.nan,
            "Volume_Price_Ratio": hist["Volume"].iloc[-1] / hist["Close"].iloc[-1] if not hist.empty else np.nan,
            "Return_6m": hist["Close"].pct_change(min(len(hist), 126)).iloc[-1] if len(hist) > 126 else np.nan,
        }

        valid_tickers.append(ticker)  # Aggiungi ai validi solo se ha dati

    except Exception as e:
        print(f"⚠️ Error processing {ticker}: {e}, skipping...")

df = pd.DataFrame.from_dict(data, orient="index")
df.dropna(inplace=True)  # Rimuove eventuali righe con NaN
df.to_csv("stock_data.csv")

print(f"✅ Dati raccolti per {len(valid_tickers)} aziende su {len(tickers)}")

# 2️⃣ Preprocessing dei dati (=preparo i dati per le tecniche di ML)
df = pd.read_csv("stock_data.csv", index_col=0)
X = df.drop(columns=["Return_6m"])  # Features
y = df["Return_6m"]  # Nuovo target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # 80% train, 20% test

# 3️⃣ Modelli di Regressione
## 3.1 Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
print(f"Random Forest MAE: {rf_mae:.4f}")

## 3.2 XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
print(f"XGBoost MAE: {xgb_mae:.4f}")

## 3.3 Rete Neurale
nn_model = Sequential([                         # Mi permette di avere piu layer
    keras.Input(shape=(X_train.shape[1],)),     # Input layer, shape = numero di features
    Dense(32, activation='relu'),               # Hidden layer, 32 neuroni con funzione di attivazione ReLU, aggiunge non linearità
    Dropout(0.2),                               # Dropout layer, 20% dei neuroni disattivati per evitare overfitting
    Dense(16, activation='relu'),               # Hidden layer, 16 neuroni con funzione di attivazione
    Dense(1)                                    # Output layer, 1 neurone, tipicamente per la regressione
])
nn_model.compile(optimizer='adam', loss='mse')  # Compilazione del modello, ottimizzatore Adam, loss function Mean Squared Error
nn_model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0, validation_data=(X_test, y_test)) # Addestramento del modello con 50 epoche e batch size 8, Verboso = 0, non stampa nulla, validation_data = dati di test
y_pred_nn = nn_model.predict(X_test).flatten()
nn_mae = mean_absolute_error(y_test, y_pred_nn)
print(f"Neural Network MAE: {nn_mae:.4f}")

# 4️⃣ Confronto dei Modelli
print(f"R2 Score Random Forest: {r2_score(y_test, y_pred_rf):.4f}")
print(f"R2 Score XGBoost: {r2_score(y_test, y_pred_xgb):.4f}")
print(f"R2 Score Neural Network: {r2_score(y_test, y_pred_nn):.4f}")