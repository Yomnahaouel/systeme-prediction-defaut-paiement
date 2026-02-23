# Credit Default Risk — Prediction Pipeline

**Système de prédiction du défaut de paiement (Credit Scoring)**

Pipeline complet de Machine Learning : EDA → Preprocessing → Feature Engineering → Feature Selection → Training multi-modèle → Comparaison → Déploiement API.

**Auteur :** Yomna Haouel

---

## Contexte

Ce projet a pour objectif de développer un système de prédiction du défaut de paiement
à partir de données clients liées aux demandes de crédit (Home Credit).

**Problématique :** Comment prédire, à partir des informations financières et socio-démographiques d'un client, le risque de défaut de paiement ?

Le défaut de paiement représente un enjeu critique pour les institutions financières.
Ce pipeline applique une approche industrielle avec anti data-leakage, reproductibilité et tracking MLflow.

---

## Architecture du Projet

```
systeme-prediction-defaut-paiement/
│
├── config.py                    # Configuration centralisée (paths, params, grids)
├── requirements.txt             # Dépendances Python
├── README.md                    # Documentation complète
│
├── code/
│   ├── exploration_donnees.py   # 1. EDA — Analyse exploratoire
│   ├── preprocessing.py         # 2. Preprocessing — Nettoyage + Pipeline sklearn
│   ├── feature_engineering.py   # 3. Feature Engineering — Ratios financiers
│   ├── feature_selection.py     # 4. Feature Selection — Corrélation + MI + LightGBM
│   ├── training.py              # 5. Training — 5 modèles + SMOTE + MLflow
│   ├── model_comparison.py      # 6. Comparaison — Sélection du meilleur modèle
│   └── main.py                  # Orchestrateur principal
│
├── api/
│   └── app.py                   # FastAPI — API de prédiction
│
├── data/
│   └── feature_matrix.csv       # Dataset (Home Credit)
│
├── models/                      # Modèles sauvegardés (.joblib)
├── plots/                       # Graphiques générés
│   ├── eda/                     # Plots EDA
│   ├── training/                # Confusion matrices, ROC curves
│   └── comparison/              # Comparaison multi-modèle
│
└── mlruns/                      # MLflow tracking (auto-generated)
```

---

## Installation

```bash
# 1. Cloner le repo
git clone <repo-url>
cd systeme-prediction-defaut-paiement

# 2. Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## Exécution

### Pipeline complet (sans EDA)
```bash
python code/main.py
```

### Pipeline complet avec EDA
```bash
python code/main.py --eda
```

### EDA uniquement
```bash
python code/main.py --eda-only
```

### Lancer l'API de prédiction
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
Puis ouvrir : http://localhost:8000/docs

### Lancer MLflow UI
```bash
mlflow ui --port 5000
```

---

## Description Détaillée du Pipeline

### 1. EDA (`exploration_donnees.py`)
- Distribution de TARGET et taux de défaut (~8%)
- Analyse des valeurs manquantes (seuil configurable)
- Distributions numériques par classe (histogrammes)
- Détection d'outliers (méthode IQR + boxplots)
- Analyse catégorielle (taux de défaut par modalité)
- Corrélation avec TARGET (top 20 + heatmap top 12)
- Recommandations métier automatiques

### 2. Preprocessing (`preprocessing.py`)
- Chargement et filtrage (train only, TARGET valide)
- Nettoyage basique (suppression colonnes >60% manquantes)
- **Split stratifié AVANT toute transformation** (anti data-leakage)
- Pipeline sklearn : `SimpleImputer(median)` → `StandardScaler` pour le numérique
- Pipeline sklearn : `SimpleImputer(most_frequent)` → `OrdinalEncoder` pour le catégoriel
- Pipeline sauvegardé en `.joblib` pour le déploiement

### 3. Feature Engineering (`feature_engineering.py`)
| Feature | Formule | Justification Métier |
|---------|---------|----------------------|
| CREDIT_INCOME_RATIO | AMT_CREDIT / AMT_INCOME | Capacité d'endettement |
| ANNUITY_INCOME_RATIO | AMT_ANNUITY / AMT_INCOME | Effort financier mensuel |
| CREDIT_TERM | AMT_CREDIT / AMT_ANNUITY | Durée estimée du crédit |
| INCOME_PER_PERSON | AMT_INCOME / CNT_FAM_MEMBERS | Revenu par personne |
| AGE_YEARS | -DAYS_BIRTH / 365.25 | Âge en années |
| EMPLOYMENT_YEARS | -DAYS_EMPLOYED / 365.25 | Ancienneté emploi |
| EMPLOYED_ANOMALY | DAYS_EMPLOYED == 365243 | Détection retraité/chômeur |
| EXT_SOURCE_MEAN | mean(EXT_1, EXT_2, EXT_3) | Score de crédit agrégé |
| EXT_SOURCE_1x2 | EXT_1 × EXT_2 | Interaction non-linéaire |
| DOCUMENTS_PROVIDED | sum(FLAG_DOCUMENT_*) | Complétude du dossier |
| REGION_MISMATCH_SUM | sum(REG_*_NOT_*) | Stabilité géographique |

### 4. Feature Selection (`feature_selection.py`)
- **Étape 1 :** Filtre de corrélation (seuil 0.85) — supprime la redondance
- **Étape 2 :** Mutual Information (top 120) — pertinence statistique
- **Étape 3 :** Importance LightGBM (seuil 0.001) — pertinence non-linéaire
- **Combinaison :** Stratégie Union ou Intersection (configurable dans `config.py`)
- **Optimisation vitesse :** Sampling 50K/80K lignes pour les calculs coûteux

### 5. Training Multi-Modèle (`training.py`)
5 modèles entraînés avec gestion du déséquilibre :

| Modèle | Stratégie déséquilibre | Vitesse |
|--------|----------------------|---------|
| Logistic Regression | class_weight='balanced' | ⚡ Très rapide |
| Random Forest | class_weight='balanced' | 🔄 Moyen |
| XGBoost | scale_pos_weight=10 | ⚡ Rapide |
| LightGBM | is_unbalance=True | ⚡ Très rapide |
| CatBoost | auto_class_weights='Balanced' | 🔄 Moyen |

- SMOTE appliqué sur le train uniquement (aucune fuite)
- Métriques : ROC-AUC, Recall, Precision, F1
- Confusion matrix et ROC curve par modèle
- Sauvegarde `.joblib` de chaque modèle

### 6. MLflow (`training.py`)
- Tracking automatique des hyperparamètres par run
- Tracking des métriques (AUC, Recall, F1)
- Sauvegarde des modèles dans l'artifact store
- Interface web pour comparer les expériences

### 7. Model Comparison (`model_comparison.py`)
- Tableau comparatif (AUC, Recall, F1, temps d'entraînement)
- ROC curves combinées sur un même graphique
- Sélection pondérée : **50% AUC + 30% Recall + 20% F1**
- Analyse SHAP (feature importance globale)
- Justification formelle du choix du modèle

### 8. FastAPI (`api/app.py`)
- `GET /` → Health check
- `GET /info` → Infos modèle + features attendues
- `POST /predict` → Prédiction pour un client

Exemple de requête :
```json
POST /predict
{
  "features": {
    "AMT_CREDIT": 500000.0,
    "AMT_INCOME_TOTAL": 150000.0,
    "AMT_ANNUITY": 25000.0,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.4
  }
}
```

Réponse :
```json
{
  "client_id": "client_request",
  "default_probability": 0.1234,
  "prediction": 0,
  "risk_level": "LOW",
  "threshold": 0.5
}
```

---

## Bonnes Pratiques Appliquées

| Pratique | Implémentation |
|----------|---------------|
| **Anti data-leakage** | Split AVANT preprocessing ; SMOTE sur train uniquement |
| **Reproductibilité** | `RANDOM_STATE=42` dans toutes les opérations stochastiques |
| **Configuration centralisée** | Tout dans `config.py`, aucune valeur en dur dans le code |
| **Tracking** | MLflow pour paramètres, métriques, modèles |
| **Modularité** | 1 fichier = 1 responsabilité, imports clairs |
| **Scalabilité** | Sampling pour les opérations CPU-intensives |
| **Déploiement** | FastAPI + modèle sérialisé + documentation Swagger |

---

## Recommandations Avancées

### Gestion du déséquilibre
- **SMOTE** inside training folds (jamais sur le test)
- **class_weight='balanced'** dans les modèles linéaires
- **Threshold tuning** : ajuster le seuil de décision (par défaut 0.5) pour maximiser le Recall

### Interprétabilité (SHAP)
- `shap.TreeExplainer` pour les modèles à base d'arbres
- Summary plot pour importance globale des features
- Force plot pour expliquer une prédiction individuelle

### Optimisation Performance
- LightGBM > XGBoost pour les grands datasets
- Sampling pour MI et corrélation (50K-80K suffisent)
- `n_jobs=-1` pour le parallélisme partout

---

## Conseils pour la Soutenance

1. **Commencer par le business** : expliquer le coût d'un défaut non détecté vs faux positif
2. **Montrer l'EDA** : les plots dans `plots/eda/` illustrent le déséquilibre et les features clés
3. **Insister sur l'anti-leakage** : split avant transform, SMOTE dans les folds uniquement
4. **Présenter la comparaison** : le tableau et les ROC curves justifient le choix objectivement
5. **Montrer SHAP** : interprétabilité = confiance dans le modèle pour le métier
6. **Démo API live** : lancer FastAPI et montrer une prédiction en temps réel
7. **MLflow** : montrer le tracking des expériences et la reproductibilité

---

## Jeu de données

Le projet repose sur le dataset Home Credit (Kaggle), contenant ~307K clients avec 1698 features.
Le fichier `data/feature_matrix.csv` n'est pas inclus dans le dépôt (trop volumineux).

---

## Licence

Projet académique — Data Science.
