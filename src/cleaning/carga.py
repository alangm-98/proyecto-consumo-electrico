"""Lectura del archivo crudo."""

import polars as pl

from config.rutas import ARCHIVO_RAW

SEPARADOR = ";"
MARCA_FALTANTE = "?"

COLUMNAS_NUMERICAS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def cargar_crudo_texto(ruta=ARCHIVO_RAW) -> pl.DataFrame:
    """Lee todas las columnas como texto."""
    return pl.read_csv(ruta, separator=SEPARADOR, infer_schema=False)


def cargar_crudo_inferido(ruta=ARCHIVO_RAW) -> pl.DataFrame:
    """Lee el archivo con los tipos que infiere polars."""
    return pl.read_csv(ruta, separator=SEPARADOR, infer_schema_length=10_000)
