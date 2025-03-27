import yfinance as yf
import pandas as pd

# Lista di aziende (aggiungine altre se vuoi arrivare a 100)
tickers = [
    "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "JPM", "BAC", "WFC", "C", "GS", 
    "XOM", "CVX", "COP", "BP", "SHEL", "PG", "KO", "PEP", "MCD", "WMT",
    "DIS", "V", "MA", "NVDA", "NFLX", "INTC", "AMD", "BA", "GE", "HON",
    "IBM", "T", "VZ", "PFE", "MRNA", "JNJ", "LLY", "NKE", "ADBE", "ORCL",
    "CSCO", "CAT", "MMM", "UPS", "FDX", "MDT", "UNH", "CVS", "TGT", "LOW"
]


# Dizionario per salvare i dati
data = {}

for ticker in tickers:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")  # Prezzi ultimi 12 mesi
    info = stock.info  # Info fondamentali
    
    # Estrai gli indicatori finanziari chiave
    data[ticker] = {
        "PE": info.get("trailingPE", None),  # P/E Ratio
        "PB": info.get("priceToBook", None),  # P/B Ratio
        "ROE": info.get("returnOnEquity", None),  # ROE
        "ROA": info.get("returnOnAssets", None),  # ROA
        "DebtToEquity": info.get("debtToEquity", None),  # Debt-to-Equity
        "Momentum_6m": hist["Close"].pct_change(min(len(hist), 126)).iloc[-1],  # Rendimento ultimi 6 mesi
        "Volatility": hist["Close"].pct_change().std() * (252 ** 0.5),  # Volatilità annualizzata
        "Return_1y": hist["Close"].pct_change(min(len(hist), 249)).iloc[-1],  # Rendimento ultimi 12 mesi
    }

# Convertiamo il dizionario in DataFrame
df = pd.DataFrame.from_dict(data, orient="index")

# Mostriamo i dati
print(df.head())

# Salviamo il dataset
df.to_csv("stock_data.csv", index=True)
