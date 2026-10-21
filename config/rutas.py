"""Rutas estáticas del proyecto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DATA = RAIZ / "data"
DATA_RAW = DATA / "data-raw"
DATA_PROCESSED = DATA / "data-processed"

ARCHIVO_RAW = DATA_RAW / "household_power_consumption.txt"
ARCHIVO_LIMPIO = DATA_PROCESSED / "consumo_limpio.parquet"
