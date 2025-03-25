import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class StockReturnPredictor:
    def __init__(self, tickers, start_date='2020-01-01', end_date='2024-03-25'):
        """
        Inizializzazione del predittore di rendimenti azionari
        
        Parameters:
        - tickers: lista di simboli azionari da analizzare
        - start_date: data di inizio per il download dei dati
        - end_date: data di fine per il download dei dati
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.features = None
        self.target = None
        
    def _calculate_rsi(self, prices, periods=14):
        """
        Calcolo dell'Indice di Forza Relativa (RSI)
        
        Parameters:
        - prices: serie dei prezzi
        - periods: periodo per il calcolo
        
        Returns:
        RSI medio
        """
        try:
            delta = prices.diff()
            
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not rsi.empty else np.nan
        except Exception as e:
            logging.warning(f"Errore nel calcolo RSI: {e}")
            return np.nan
    
    def _calculate_macd(self, prices, slow=26, fast=12, signal=9):
        """
        Calcolo dell'Indice MACD (Moving Average Convergence Divergence)
        
        Parameters:
        - prices: serie dei prezzi
        - slow/fast/signal: periodi per i calcoli
        
        Returns:
        Valore MACD
        """
        try:
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            
            return (macd - signal_line).iloc[-1] if not macd.empty else np.nan
        except Exception as e:
            logging.warning(f"Errore nel calcolo MACD: {e}")
            return np.nan
    
    def download_data(self):
        """
        Scarica dati storici e fondamentali per i ticker specificati
        
        Returns:
        DataFrame con dati finanziari
        """
        all_data = {}
        
        for ticker in self.tickers:
            try:
                # Download dati storici
                stock = yf.Ticker(ticker)
                hist = stock.history(start=self.start_date, end=self.end_date)
                
                # Verifica la disponibilità dei dati
                if hist.empty:
                    logging.warning(f"Nessun dato disponibile per {ticker}")
                    continue
                
                # Calcolo features
                features = self._extract_features(hist, stock)
                
                if features is not None:
                    all_data[ticker] = features
                
            except Exception as e:
                logging.error(f"Errore durante il download di {ticker}: {e}")
        
        # Conversione in DataFrame
        self.data = pd.DataFrame.from_dict(all_data, orient='index')
        
        # Pulizia dei dati
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data = self.data.dropna()
        
        logging.info(f"Dati scaricati per {len(self.data)} ticker")
        return self.data
    
    def _extract_features(self, hist, stock):
        """
        Estrazione delle features da dati storici e info ticker
        
        Parameters:
        - hist: DataFrame storico
        - stock: oggetto Ticker
        
        Returns:
        Dizionario di features
        """
        try:
            features = {
                # Metriche di prezzo
                'Close': hist['Close'].iloc[-1] if not hist.empty else np.nan,
                'PriceChange': hist['Close'].pct_change().mean(),
                'PriceVolatility': hist['Close'].pct_change().std(),
                
                # Metriche volumi
                'VolumeChange': hist['Volume'].pct_change().mean(),
                'VolumeVolatility': hist['Volume'].pct_change().std(),
                
                # Indicatori tecnici
                'RSI': self._calculate_rsi(hist['Close']),
                'MACD': self._calculate_macd(hist['Close']),
                
                # Metriche fondamentali (con gestione eccezioni)
                'PE': stock.info.get('trailingPE', np.nan),
                'PB': stock.info.get('priceToBook', np.nan),
                'MarketCap': stock.info.get('marketCap', np.nan),
                
                # Rendimento target (prossimo mese)
                'FutureReturn': hist['Close'].pct_change(periods=30).iloc[-1] 
                    if len(hist) >= 30 else np.nan
            }
            return features
        
        except Exception as e:
            logging.error(f"Errore nell'estrazione features: {e}")
            return None
    
    def prepare_features(self):
        """
        Preparazione delle features per il modello di regressione
        
        Returns:
        Features e target
        """
        if self.data is None or self.data.empty:
            raise ValueError("Nessun dato disponibile. Eseguire download_data() prima.")
        
        # Separazione features e target
        self.features = self.data.drop(columns=['FutureReturn'])
        self.target = self.data['FutureReturn']
        
        logging.info(f"Preparate {len(self.features.columns)} features")
        return self.features, self.target
    
    def build_model(self):
        """
        Costruzione del pipeline di regressione con feature selection
        
        Returns:
        Pipeline di regressione
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Standardizzazione
            ('selector', SelectKBest(score_func=f_regression, k=5)),  # Selezione migliori 5 feature
            ('regressor', GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=3, 
                random_state=42
            ))
        ])
        
        return pipeline
    
    def train_and_evaluate(self, X, y):
        """
        Addestramento e valutazione del modello
        
        Parameters:
        - X: features
        - y: target
        
        Returns:
        Metriche di performance
        """
        # Split dei dati
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Inizializzazione modello
        model = self.build_model()
        
        # Addestramento
        model.fit(X_train, y_train)
        
        # Predizioni
        y_pred = model.predict(X_test)
        
        # Metriche di valutazione
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation con Time Series Split
        cv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        
        return {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'CV Scores': -cv_scores.mean()
        }
    
    def visualize_results(self, X_test, y_test, y_pred):
        """
        Visualizzazione grafica dei risultati
        """
        plt.figure(figsize=(12, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Returns')
        plt.ylabel('Predicted Returns')
        plt.title('Actual vs Predicted Stock Returns')
        plt.show()

def main():
    # Lista di aziende più ampia e diversificata
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Tecnologia
        "JPM", "BAC", "GS", "C",  # Finanza
        "XOM", "CVX", "COP",  # Energia
        "MRNA", "PFE", "JNJ",  # Salute
        "WMT", "COST", "TGT",  # Consumo
        "F", "GM", "TSLA"  # Automotive
    ]
    
    # Istanza del predittore
    predictor = StockReturnPredictor(tickers)
    
    try:
        # Download e preparazione dati
        data = predictor.download_data()
        X, y = predictor.prepare_features()
        
        # Addestramento e valutazione
        results = predictor.train_and_evaluate(X, y)
        
        # Stampa risultati
        for metric, value in results.items():
            print(f"{metric}: {value}")
    
    except Exception as e:
        logging.error(f"Errore nel processo: {e}")

if __name__ == "__main__":
    main()