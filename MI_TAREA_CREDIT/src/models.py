import json
import pickle
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

SEED = 42

MODELOS_CONFIG = {
    "Random Forest": (
        RandomForestClassifier(random_state=SEED),
        {"n_estimators": [100, 200], "max_depth": [4, 6, None]},
    ),
    "XGBoost": (
        XGBClassifier(random_state=SEED, eval_metric="logloss", verbosity=0),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    ),
    "Logistic Regression": (
        LogisticRegression(random_state=SEED, max_iter=1000),
        {"C": [0.01, 0.1, 1, 10]},
    ),
}

def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Entrena modelos con GridSearchCV y retorna los mejores."""
    mejores_modelos = {}
    for nombre, (modelo, params) in MODELOS_CONFIG.items():
        gs = GridSearchCV(modelo, params, cv=5, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)
        mejores_modelos[nombre] = gs.best_estimator_
    return mejores_modelos

def evaluate_models(models: dict, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """Retorna tabla con AUC-ROC ordenado de mayor a menor."""
    resultados = []
    for nombre, modelo in models.items():
        y_prob = modelo.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        resultados.append({"Modelo": nombre, "AUC": auc})
    return pd.DataFrame(resultados).sort_values("AUC", ascending=False)

def save_model(model, path: str, metadata: dict) -> None:
    """Serializa el modelo y genera el metadata.json automático."""
    directorio = Path(path)
    directorio.mkdir(parents=True, exist_ok=True)
    
    # Guarda el archivo del modelo (.pkl)
    with open(directorio / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Agrega fecha automática y guarda metadata.json
    metadata["saved_at"] = date.today().isoformat()
    with open(directorio / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)