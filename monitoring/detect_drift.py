import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
import os

# Charge le dataset
print("Chargement des données...")
df = pd.read_csv("creditcard.csv")

# Données de référence — 80% du dataset
reference = df.sample(frac=0.8, random_state=42).drop(columns=["Class"])

# Données de production simulées — 20% restants avec drift artificiel
current = df.sample(frac=0.2, random_state=123).drop(columns=["Class"])
current["V14"] = current["V14"] * 1.5 + np.random.normal(0, 0.5, len(current))
current["Amount"] = current["Amount"] * 2.0

print("Génération du rapport Evidently...")
report = Report(metrics=[DataDriftPreset()])
my_eval = report.run(reference_data=reference, current_data=current)

# Sauvegarde le rapport HTML
os.makedirs("monitoring/reports", exist_ok=True)
my_eval.save_html("monitoring/reports/drift_report.html")

print("\n=== RAPPORT GÉNÉRÉ ===")
print("Rapport HTML sauvegardé dans monitoring/reports/drift_report.html")
print("Ouvre ce fichier dans ton navigateur pour voir les résultats visuels !")