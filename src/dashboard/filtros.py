"""Filtros de la barra lateral y su aplicación al resumen por hora."""

from dataclasses import dataclass
from datetime import date

import polars as pl
import streamlit as st

AGREGACIONES = {"Hora": "1h", "Día": "1d", "Semana": "1w", "Mes": "1mo"}
TIPOS_DIA = ["Todos", "Laborable", "Fin de semana"]
ESTACIONES = {
    "Invierno": "invierno",
    "Primavera": "primavera",
    "Verano": "verano",
    "Otoño": "otono",
}


@dataclass
class Filtros:
    inicio: date
    fin: date
    agregacion: str
    tipo_dia: str
    estaciones: list[str]


def barra_lateral(df: pl.DataFrame) -> Filtros:
    """Dibuja los filtros en la barra lateral y devuelve su valor actual."""
    primera = df["datetime"].min().date()
    ultima = df["datetime"].max().date()

    with st.sidebar:
        st.header(":violet[:material/filter_alt:] Filtros")
        rango = st.date_input(
            "Rango de fechas",
            value=(primera, ultima),
            min_value=primera,
            max_value=ultima,
        )
        agregacion = st.pills(
            "Agregación",
            list(AGREGACIONES),
            default="Mes",
            required=True,
        )
        tipo_dia = st.pills(
            "Tipo de día",
            TIPOS_DIA,
            default="Todos",
            required=True,
        )
        estaciones = st.pills(
            "Estación",
            list(ESTACIONES),
            selection_mode="multi",
            default=list(ESTACIONES),
        )
        st.caption("Los filtros aplican a la pestaña «Consumo del hogar».")
        st.divider()
        st.markdown(
            "**Acerca de los datos**  \n"
            "Individual Household Electric Power Consumption "
            "([UCI](https://archive.ics.uci.edu/dataset/235/"
            "individual+household+electric+power+consumption)). "
            "2,075,259 mediciones por minuto, resumidas por hora para el dashboard."
        )

    # Mientras se elige la segunda fecha, el control devuelve solo una
    inicio, fin = (rango[0], rango[-1]) if rango else (primera, ultima)
    return Filtros(inicio, fin, agregacion, tipo_dia, estaciones)


def aplicar(df: pl.DataFrame, f: Filtros) -> pl.DataFrame:
    """Horas que cumplen el rango de fechas, el tipo de día y las estaciones."""
    condicion = pl.col("datetime").dt.date().is_between(f.inicio, f.fin) & pl.col(
        "estacion"
    ).is_in([ESTACIONES[e] for e in f.estaciones])
    if f.tipo_dia == "Laborable":
        condicion &= ~pl.col("es_fin_semana")
    elif f.tipo_dia == "Fin de semana":
        condicion &= pl.col("es_fin_semana")
    return df.filter(condicion)
