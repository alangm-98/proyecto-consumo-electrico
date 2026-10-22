"""Dashboard de consumo eléctrico doméstico (Streamlit)."""

import sys
from pathlib import Path

# Streamlit agrega al path la carpeta de este archivo; se agrega la raíz para importar src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
import streamlit as st

from config.rutas import ARCHIVO_HORARIO
from src.dashboard import filtros, metricas
from src.dashboard.tema import ICONO

st.set_page_config(
    page_title="Consumo eléctrico · Práctica 5",
    page_icon=ICONO.read_text(encoding="utf-8"),
    layout="wide",
)


@st.cache_data
def cargar_horario() -> pl.DataFrame:
    return pl.read_parquet(ARCHIVO_HORARIO)


def numero(valor: float | None, decimales: int = 1, sufijo: str = "") -> str:
    return "—" if valor is None else f"{valor:,.{decimales}f}{sufijo}"


def seccion_kpis(df: pl.DataFrame) -> None:
    k = metricas.kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        ":blue[:material/bolt:] Consumo diario",
        numero(k["consumo_diario_media"], 1, " kWh"),
        delta=f"Mediana {numero(k['consumo_diario_mediana'], 1)} kWh",
        delta_color="blue",
        delta_arrow="off",
        chart_data=k["consumo_mensual"] or None,
        help=f"Promedio de {k['dias_completos']:,} días con al menos 23 h medidas.",
        border=True,
        height="stretch",
    )
    diferencia = k["diferencia_fin_semana"]
    c2.metric(
        ":orange[:material/speed:] Potencia media",
        numero(k["potencia_media"], 2, " kW"),
        delta=None if diferencia is None else f"{diferencia:+.0f} % fin de semana",
        help="Potencia activa promedio. La variación compara fines de semana con días laborables.",
        border=True,
        height="stretch",
    )
    fecha = k["fecha_maxima"]
    c3.metric(
        ":red[:material/trending_up:] Potencia máxima",
        numero(k["potencia_maxima"], 2, " kW"),
        delta=None if fecha is None else f"{fecha:%d/%m/%Y · %H} h",
        delta_color="red",
        delta_arrow="off",
        help="Máximo de un solo minuto en el periodo filtrado.",
        border=True,
        height="stretch",
    )
    c4.metric(
        ":green[:material/check_circle:] Cobertura de datos",
        numero(k["cobertura"], 1, " %"),
        delta=f"{k['minutos_faltantes']:,} min sin medición",
        delta_color="off",
        delta_arrow="off",
        help="Porcentaje de minutos con medición (originales o interpolados).",
        border=True,
        height="stretch",
    )


horario = cargar_horario()
seleccion = filtros.barra_lateral(horario)
filtrado = filtros.aplicar(horario, seleccion)

st.title(":yellow[:material/bolt:] Consumo eléctrico doméstico")
st.caption("Hogar en Sceaux, Francia · dic 2006 – nov 2010 · una medición por minuto")

consumo, preparacion = st.tabs(
    [":material/home: Consumo del hogar", ":material/science: Preparación de datos"]
)

with consumo:
    if filtrado.is_empty():
        st.warning(
            "Ninguna hora cumple los filtros elegidos. Amplía el rango de fechas "
            "o selecciona al menos una estación.",
            icon=":material/filter_alt_off:",
        )
    else:
        seccion_kpis(filtrado)

with preparacion:
    st.caption("Calidad de los datos, correlación de Pearson y PCA.")
