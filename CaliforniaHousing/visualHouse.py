import pandas as pd 
import matplotlib.pyplot as plt # Per creare grafici
import sklearn.datasets # Contiene i dataset di esempio

# Read the data from the dictionary
calHouse = sklearn.datasets.fetch_california_housing()

#converto dizionario in dataframe
df = pd.DataFrame(calHouse.data, columns=calHouse.feature_names) # Colonne = nomi features del dizionario

#aggiungo colonna target al DF
df['target'] = calHouse.target

#prime righe
print("Sono qua111!!!")
print(df.head())

#statistiche
print("Sono qua222!!!")
print(df.describe()) # statistichre base (media,deviazione standard, min, max, quantili)

#presenza valori mancanti
print("Sono qua333!!!")
print(df.isnull().sum()) # conto valori mancanti per colonna

#distribuzione prezzi delle case
plt.hist(df.target, bins=50) # bins = intervalli
plt.xlabel('Prezzo')
plt.ylabel('Numero di case')
plt.title('Distribuzione prezzi delle case')
plt.show()

#rimuovo righe con valori mancanti
df.dropna(inplace=True)

# Normalizzo i dati
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # rendo i dati con media=0 e stddev=1
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns) # calcolo trasformazione ed applico ai dati

# Creo un modello di regressione lineare
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = df_scaled.drop("target", axis=1)
y = df_scaled["target"]

# Divido i dati in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training, 20% test

# Creo il modello
model = LinearRegression() # modello di regressione lineare
model.fit(X_train, y_train) # addestramento sui dati di training

# Valuto il modello
y_pred = model.predict(X_test) # predizione sui dati di test

# Calcolo l'errore
error = mean_squared_error(y_test, y_pred)
print("Errore: ", error)

# visualizzo il modello
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Prezzo reale')
plt.ylabel('Prezzo predetto')
plt.title('Prezzo reale vs prezzo predetto')
plt.show()