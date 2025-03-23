import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# Creiamo un modello di Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Alleniamo il modello
model.fit(X_train, y_train)

# Facciamo previsioni sul set di test
y_pred = model.predict(X_test)

# Valutiamo il modello
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Accuracy del modello Random Forest: {accuracy:.2f}")
print("\n🔎 Classification Report:\n", classification_report(y_test, y_pred))
