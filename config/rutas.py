"""Rutas estáticas del proyecto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DATA = RAIZ / "data"
DATA_RAW = DATA / "data-raw"
DATA_PROCESSED = DATA / "data-processed"

ARCHIVO_RAW = DATA_RAW / "household_power_consumption.txt"
ARCHIVO_LIMPIO = DATA_PROCESSED / "consumo_limpio.parquet"
ARCHIVO_NORMALIZADO = DATA_PROCESSED / "consumo_normalizado.parquet"
PARAMETROS_ZSCORE = DATA_PROCESSED / "parametros_zscore.json"
ARCHIVO_FINAL = DATA_PROCESSED / "dataset_final.parquet"
ARCHIVO_HORARIO = DATA_PROCESSED / "consumo_horario.parquet"
ARCHIVO_PEARSON = DATA_PROCESSED / "pearson_entrenamiento.json"

FIGURAS = RAIZ / "reports" / "figures"
FIG_PEARSON = FIGURAS / "pearson_heatmap.png"
FIG_SCREE = FIGURAS / "pca_scree.png"
FIG_LOADINGS = FIGURAS / "pca_loadings.png"
FIG_PROYECCION = FIGURAS / "pca_proyeccion_2d.png"

PARAMETROS_PCA = DATA_PROCESSED / "parametros_pca.json"
