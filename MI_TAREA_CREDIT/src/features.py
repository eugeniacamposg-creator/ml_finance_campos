import numpy as np
import pandas as pd

def compute_woe_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Calcula WoE e IV para una variable numérica continua."""
    df = df.copy()
    # Discretiza la variable en intervalos
    df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')
    
    # Agrupa por bin y calcula eventos (Default=1) y no eventos (Default=0)
    stats = df.groupby('bin', observed=True)[target].agg(['count', 'sum'])
    stats.columns = ['n_obs', 'n_events']
    stats['n_non_events'] = stats['n_obs'] - stats['n_events']
    
    # Proporciones (sumamos 0.0001 para evitar divisiones por cero)
    dist_events = stats['n_events'] / (stats['n_events'].sum() + 0.0001)
    dist_non_events = stats['n_non_events'] / (stats['n_non_events'].sum() + 0.0001)
    
    # Cálculo de WoE e IV por cada bin
    stats['woe'] = np.log(dist_events / dist_non_events)
    stats['iv_bin'] = (dist_events - dist_non_events) * stats['woe']
    
    return stats, stats['iv_bin'].sum()

def select_features_by_iv(df: pd.DataFrame, target: str, threshold: float = 0.1) -> list[str]:
    """Retorna la lista de variables con IV superior al umbral establecido."""
    selected = []
    for col in df.columns:
        if col != target:
            # Calculamos el IV para cada columna
            _, iv = compute_woe_iv(df, col, target)
            if iv >= threshold:
                selected.append(col)
    return selected