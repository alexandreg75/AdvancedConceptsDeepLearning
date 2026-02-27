import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("--- GLASS-BOX : RÉGRESSION LOGISTIQUE (BREAST CANCER) ---")

# 1. Chargement des données
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0: Maligne, 1: Bénigne

# 2. Séparation des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Normalisation (Crucial pour l'interprétabilité des coefficients)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # ✅ fit_transform sur train
X_test_scaled = scaler.transform(X_test)         # ✅ transform sur test

# 4. Entraînement du modèle "Glass-box"
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)               # ✅ fit

# Évaluation rapide
y_pred = model.predict(X_test_scaled)
print(f"Accuracy de la Régression Logistique : {accuracy_score(y_test, y_pred):.4f}")

# 5. Extraction de l'explication (Intrinsèque)
coefficients = model.coef_[0]                    # ✅ coef_

# Création d'un DataFrame pour faciliter la manipulation
feature_importance = pd.DataFrame({
    "Feature": data.feature_names,
    "Coefficient": coefficients
})

# Tri par importance absolue
feature_importance["Abs_Coefficient"] = feature_importance["Coefficient"].abs()
feature_importance = feature_importance.sort_values(by="Abs_Coefficient", ascending=True)

# 6. Visualisation
plt.figure(figsize=(10, 8))
colors = ["red" if c < 0 else "blue" for c in feature_importance["Coefficient"]]

top15 = feature_importance.tail(15)
plt.barh(top15["Feature"], top15["Coefficient"], color=[("red" if c < 0 else "blue") for c in top15["Coefficient"]])

plt.xlabel("Valeur du Coefficient (β)")
plt.title("Top 15 - Importance des variables (Régression Logistique)")
plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
plt.tight_layout()

output_filename = "glassbox_coefficients.png"
plt.savefig(output_filename, dpi=200)
print(f"Graphique sauvegardé dans {output_filename}")

# Petite aide pour la question 3.c : feature la plus "maligne" (coef le plus négatif)
most_malignant = feature_importance.sort_values("Coefficient", ascending=True).iloc[0]
print(f"Feature la plus poussant vers MALIGNE (coef le plus négatif) : {most_malignant['Feature']} ({most_malignant['Coefficient']:.4f})")
