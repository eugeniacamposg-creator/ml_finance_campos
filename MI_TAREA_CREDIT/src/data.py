import pandas as pd

# Estas son las columnas que el profesor pide validar
REQUIRED_COLUMNS = [
    "Age", "Employ", "Address", "Income",
    "Creddebt", "OthDebt", "MonthlyLoad", "Default"
]

def load_raw(path: str) -> pd.DataFrame:
    """Carga el dataset crudo desde un CSV."""
    return pd.read_csv(path)

def validate_schema(df: pd.DataFrame) -> None:
    """Lanza ValueError si falta alguna columna requerida."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}")

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features derivadas y elimina filas con NaN."""
    df_mod = df.copy()
    # Crea la variable pedida: OthDebtRatio
    df_mod["OthDebtRatio"] = df_mod["OthDebt"] / df_mod["Income"]
    return df_mod.dropna()