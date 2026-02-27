import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

print("--- RANDOM FOREST + SHAP ---")

# 1. Chargement et Séparation des données
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0: Maligne, 1: Bénigne

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Entraînement d'une "Boîte Noire"
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)   # ✅ fit

print(f"Accuracy du Random Forest : {model.score(X_test, y_test):.4f}")

# 3. Explicabilité Post-Hoc avec SHAP
explainer = shap.TreeExplainer(model)   # ✅ TreeExplainer

shap_values = explainer(X_test)         # ✅ appel direct sur X_test

# Classe 1 = Bénigne
shap_values_class1 = shap_values[:, :, 1]

# 4. Explicabilité Locale
patient_idx = 0
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values_class1[patient_idx], show=False)
plt.title(f"Explication Locale SHAP - Patient {patient_idx}")
plt.tight_layout()
output_local = "shap_waterfall.png"
plt.savefig(output_local, dpi=200)
plt.close()
print(f"Waterfall plot sauvegardé dans {output_local}")

# 5. Explicabilité Globale
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_class1, X_test, show=False)
plt.title("Importance globale et directionnelle des variables (SHAP)")
plt.tight_layout()
output_global = "shap_summary.png"
plt.savefig(output_global, dpi=200)
plt.close()
print(f"Summary plot sauvegardé dans {output_global}")

# Pour t'aider à répondre à la question locale :
print("\n--- Infos Patient 0 ---")
print(X_test.iloc[0])
