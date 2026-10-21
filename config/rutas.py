"""Rutas estáticas del proyecto (único lugar donde se definen)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DATA = RAIZ / "data"
DATA_RAW = DATA / "data-raw"
DATA_PROCESSED = DATA / "data-processed"

ARCHIVO_RAW = DATA_RAW / "household_power_consumption.txt"
