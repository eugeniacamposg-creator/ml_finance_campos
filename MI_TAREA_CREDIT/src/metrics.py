import numpy as np
from sklearn.metrics import roc_auc_score

def auc_roc(model, X: np.ndarray, y: np.ndarray) -> float:
    """Retorna el AUC-ROC del modelo sobre el set de datos."""
    y_prob = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_prob)

def costo_total(y_true: np.ndarray, y_prob: np.ndarray, umbral: float, c_fn: float = 500, c_fp: float = 100) -> float:
    """
    Calcula el costo operacional total.
    c_fn: costo de aprobar a alguien que no pagará (Falso Negativo).
    c_fp: costo de rechazar a alguien que sí pagaría (Falso Positivo).
    """
    y_pred = (y_prob >= umbral).astype(int)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(fn * c_fn + fp * c_fp)