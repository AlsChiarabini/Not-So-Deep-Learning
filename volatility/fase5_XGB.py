import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carica il dataset finale
df = pd.read_csv("stock_data_final.csv", index_col=0)

# Separiamo le feature (X) e la variabile target (y)
X = df.drop(columns=["Return_1y", "Target"])  # Usiamo tutti i dati tranne il target e il rendimento
y = df["Target"]  # Target: 1 = buona azione, 0 = cattiva azione


# Dividiamo i dati in training (80%) e test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Standardizziamo i dati (nel caso in cui non fosse già fatto)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definiamo il modello XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# Definiamo i parametri da ottimizzare
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7]
}

# Grid Search per trovare i migliori parametri
grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Miglior modello trovato
best_model = grid_search.best_estimator_

# Testiamo il miglior modello
y_pred = best_model.predict(X_test)

# Valutiamo il modello ottimizzato
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Accuracy del modello XGBoost Ottimizzato: {accuracy:.2f}")
print("\n🔎 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🏆 Migliori parametri trovati:", grid_search.best_params_)