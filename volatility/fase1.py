import yfinance as yf
import pandas as pd

# Lista di 20 aziende di settori diversi
tickers = [
    "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN",  # Tecnologia
    "JPM", "BAC", "WFC", "C", "GS",  # Banche
    "XOM", "CVX", "COP", "BP", "SHEL",  # Energia
    "PG", "KO", "PEP", "MCD", "WMT"  # Consumi
]

# Scarica i dati fondamentali e di prezzo
data = {}

for ticker in tickers:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")  # Prezzi ultimi 12 mesi
    info = stock.info  # Info fondamentali
    
    # Costruiamo un dizionario con le informazioni chiave
    data[ticker] = {
        "PE": info.get("trailingPE", None),  # P/E Ratio
        "PB": info.get("priceToBook", None),  # P/B Ratio
        "Momentum_6m": hist["Close"].pct_change(min(len(hist), 126)).iloc[-1],  # Rendimento ultimi 6 mesi
        "Volatility": hist["Close"].pct_change().std() * (252 ** 0.5),  # Volatilità annualizzata
        "Return_1y": hist["Close"].pct_change(min(len(hist), 249)).iloc[-1],  # Rendimento ultimi 12 mesi
    }

# Convertiamo il dizionario in DataFrame
df = pd.DataFrame.from_dict(data, orient="index")

# Mostriamo i dati
print(df)

# Salviamo il dataset
df.to_csv("stock_data.csv", index=True)
