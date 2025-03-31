import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Implementazione del metodo TOPSIS
def topsis(df, weights):
    # Il miglior asset è quello più vicino alla soluzione ideale e più lontano dalla peggiore.
    # S+ = ∑ (w_j * (x_j - x_j^+)^2)
    # S- = ∑ (w_j * (x_j - x_j^-)^2)
    # w_j^+ = max(w_j)
    # w_j^- = min(w_j)
    # w_j = pesi

    # Creare una matrice di decisione con gli asset e i criteri.
    # Normalizzare i dati per renderli confrontabili.
    # Pesare i criteri (es. più importanza a ROE rispetto al P/E).
    # Determinare la soluzione ideale e la soluzione peggiore.
    # Calcolare la distanza di ogni asset da queste soluzioni.
    # Ordinare gli asset in base alla distanza relativa.

    vectnorm = df/np.sqrt((df*2).sum()) #  1. Normalizzazione vettoriale

    weighted_matrix = vectnorm * weights # 2. Matrice pesata

    # 3. Soluzione ideale positiva e negativa
    ideal_positive = weighted_matrix.max()
    ideal_negative = weighted_matrix.min()

    # 4. Distanza euclidea
    distance_positive = np.sqrt(((weighted_matrix - ideal_positive) ** 2).sum(axis=1))
    distance_negative = np.sqrt(((weighted_matrix - ideal_negative) ** 2).sum(axis=1))

    # 5. Calcolo del punteggio di preferenza
    preference_score = distance_negative / (distance_positive + distance_negative)

    # 6. Ordinamento
    return preference_score

# Implementazione del metodo VIKOE
def vikor(df, weights, v=0.5):

    # Trova una soluzione di compromesso tra più criteri, considerando il grado di insoddisfazione per ciascuno.

    #Creare una matrice di decisione (simile a TOPSIS).
    #Normalizzare i dati e assegnare pesi ai criteri.
    #Determinare il valore migliore e peggiore per ogni criterio.
    #Calcolare gli indicatori SiSi​ e RiRi​ (misure di distanza dai valori ideali e peggiori).
    #Calcolare il punteggio VIKOR Qi​:
    #Qi​=v(Si​−Smin​)/(Smax​−Smin​)+(1−v)(Ri​−Rmin​)/(Rmax​−Rmin​)
    # dove v è un parametro di compromesso (0≤v≤1).
    # Si = somma pesata delle distanze
    # Ri = massimo scarto tra i criteri

    best = df.max()
    worst = df.min()

    # Calcolo dei valori di soddisfacimento S e di regret R
    S = ((best - df) * weights).sum(axis=1)
    R = ((df - worst) * weights).sum(axis=1)
    S_Best = S.min()
    S_Worst = S.max()
    R_Best = R.min()
    R_Worst = R.max()
    
    Q = v * (S - S_Best) / (S_Worst - S_Best) + (1 - v) * (R - R_Best) / (R_Worst - R_Best) # calcolo del punteggio Q

    return Q

# Implementazione del metodo ELECTRE
def electre(df, weights, threshold=0.6):

    #Confronta gli asset a coppie e le classifica sulla base di concordanza e discordanza.
    
    #Creare una matrice di decisione con i criteri e gli asset.
    #Normalizzare i dati e assegnare pesi.
    #Calcolare l'indice di concordanza (quanto un’alternativa è migliore di un’altra).
    #Calcolare l'indice di discordanza (quanto un’alternativa è peggiore di un’altra).
    #Eliminare le alternative dominate e classificare le rimanenti.

    n = len(df)
    concordance = np.zeros((n, n))
    discordance = np.zeros((n, n))

    # 1️⃣ Creiamo le matrici di concordanza e discordanza
    for i in range(n):
        for j in range(n):
            if i != j:
                concordance[i, j] = sum(weights[m] for m in range(len(weights)) if df.iloc[i, m] >= df.iloc[j, m])
                discordance[i, j] = max(df.iloc[j, :] - df.iloc[i, :]) / max(df.max() - df.min())

    # 2️⃣ Matrici normalizzate
    concordance /= concordance.max()
    discordance /= discordance.max()

    # 3️⃣ Determiniamo il dominio
    domination = (concordance >= threshold) & (discordance <= threshold)
    
    scores = domination.sum(axis=1)
    return scores

# Esempio di matrice di decisione
data = {
    "P/E": [25, 30, 60, 50, 28],                        # Piu basso è meglio
    "ROE": [0.15, 0.18, 0.12, 0.14, 0.17],              # Piu alto è meglio
    "Beta": [1.2, 1.1, 1.8, 1.3, 1.0],                  # Vicino a 1 è meglio
    "Volatility": [0.25, 0.20, 0.40, 0.30, 0.22],       # Piu basso è meglio
}

# Creazione del DataFrame
df = pd.DataFrame(data, index=["Stock A", "Stock B", "Stock C", "Stock D", "Stock E"])
print("Matrice di decisione:")
print(df)
print("\n")

# Normalizzazione min-max
df_normalized = (df - df.min()) / (df.max() - df.min())
print("Matrice normalizzata:")
print(df_normalized)
print("\n")

# Pesatura delle caratteristiche x TOPSIS
weights = np.array([0.25, 0.35, 0.20, 0.20])  # Pesi per P/E, ROE, Beta e Volatilità
topsis_scores = topsis(df_normalized, weights)
df["TOPSIS Score"] = topsis_scores
df_topsis = df.sort_values(by="TOPSIS Score")
print("Matrice di decisione con punteggio TOPSIS: \n", df_topsis)
print("\n")

# Pesatura delle caratteristiche x VIKOR
vikor_scores = vikor(df_normalized, weights)
df["VIKOR Score"] = vikor_scores
df_vikor = df.sort_values(by="VIKOR Score")
print("Matrice di decisione con punteggio VIKOR: \n", df_vikor)
print("\n")

# Pesatura delle caratteristiche x ELECTRE
electre_scores = electre(df_normalized, weights)
df["ELECTRE Score"] = electre_scores
df_electre = df.sort_values(by="ELECTRE Score")
print("Matrice di decisione con punteggio ELECTRE: \n", df_electre)
print("\n")

# Confronto dei punteggi
df_final = df[["TOPSIS Score", "VIKOR Score", "ELECTRE Score"]]
print("\n🔹 Confronto tra metodi:\n", df_final)
print("\n")

# Visualizzazione dei risultati
# 🔹 Impostiamo il tema dei grafici
sns.set_theme(style="whitegrid")

# 🔹 Creiamo il DataFrame dei risultati finali
df_final = df[["TOPSIS Score", "VIKOR Score", "ELECTRE Score"]]

# 🔹 Plot a barre per visualizzare i punteggi
plt.figure(figsize=(12, 5))
df_final.plot(kind="bar", figsize=(12, 6), colormap="coolwarm", edgecolor="black")

plt.title("Confronto tra TOPSIS, VIKOR e ELECTRE")
plt.xlabel("Aziende")
plt.ylabel("Punteggi Normalizzati")
plt.legend(loc="upper right")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.savefig("confronto_MCDM.png")
plt.close

plt.figure(figsize=(8,6))
sns.heatmap(df_final.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Correlazione tra i Metodi di Decisione")
plt.savefig("correlazione_MCDM.png")
plt.close
