"""Dashboard de consumo eléctrico doméstico (Streamlit)."""

import sys
from pathlib import Path

# Streamlit agrega al path la carpeta de este archivo; se agrega la raíz para importar src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
import streamlit as st

from config.rutas import ARCHIVO_HORARIO
from src.dashboard import filtros, graficas, metricas
from src.dashboard.tema import ICONO, MOCHA

st.set_page_config(
    page_title="Consumo eléctrico · Práctica 5",
    page_icon=ICONO.read_text(encoding="utf-8"),
    layout="wide",
)


@st.cache_data
def cargar_horario() -> pl.DataFrame:
    return pl.read_parquet(ARCHIVO_HORARIO)


CONFIG_GRAFICA = {"displaylogo": False}


def numero(valor: float | None, decimales: int = 1, sufijo: str = "") -> str:
    return "—" if valor is None else f"{valor:,.{decimales}f}{sufijo}"


def cuadro(color: str) -> str:
    """Cuadro de color en texto Markdown (para leyendas en subtítulos)."""
    return f':color[■]{{foreground="{color}"}}'


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


def seccion_tendencia_y_usos(df: pl.DataFrame, agregacion: str) -> None:
    izquierda, derecha = st.columns([2, 1], gap="large")
    with izquierda:
        titulo = (
            "Consumo por hora" if agregacion == "Hora" else "Consumo promedio por día"
        )
        st.subheader(f"{titulo} :blue-badge[:material/schedule: Tiempo]")
        st.caption(
            f"Agrupado por {agregacion.lower()} · sombreado: "
            f"{cuadro(MOCHA['sapphire'])} invierno · {cuadro(MOCHA['peach'])} verano"
        )
        st.plotly_chart(graficas.tendencia(df, agregacion), config=CONFIG_GRAFICA)
    with derecha:
        st.subheader(
            "Energía por uso :orange-badge[:material/donut_small: Composición]"
        )
        st.caption("Porcentaje del total del periodo filtrado")
        st.plotly_chart(graficas.energia_por_uso(df), config=CONFIG_GRAFICA)
        st.caption(
            "No submedido: iluminación, electrónica y otros aparatos sin medidor propio."
        )


def seccion_perfil(df: pl.DataFrame) -> None:
    st.subheader("Perfil diario de potencia :blue-badge[:material/schedule: Tiempo]")
    st.caption(
        "Potencia media por hora del día · cambia con los filtros de estación y tipo de día"
    )
    st.plotly_chart(
        graficas.perfil_diario(metricas.perfil_diario(df)), config=CONFIG_GRAFICA
    )


def seccion_distribucion(df: pl.DataFrame) -> None:
    diario = metricas.consumo_diario(df)
    izquierda, derecha = st.columns([2, 1], gap="large")
    with izquierda:
        st.subheader(
            "Distribución del consumo diario :violet-badge[:material/bar_chart: Distribución]"
        )
        st.caption(
            "Número de días por rango de 4 kWh · solo días completos · "
            f"{cuadro(MOCHA['text'])} media · {cuadro(MOCHA['peach'])} mediana (punteada)"
        )
        st.plotly_chart(graficas.histograma_diario(diario), config=CONFIG_GRAFICA)
    with derecha:
        st.subheader(
            "Resumen estadístico :violet-badge[:material/bar_chart: Distribución]"
        )
        st.caption("Consumo diario (kWh)")
        st.dataframe(
            metricas.resumen_estadistico(diario),
            hide_index=True,
            column_config={"Valor": st.column_config.TextColumn(alignment="right")},
        )


FORMATO_PERIODO = {
    "Hora": "YYYY-MM-DD HH:00",
    "Día": "YYYY-MM-DD",
    "Semana": "[Sem.] YYYY-MM-DD",
    "Mes": "YYYY-MM",
}


def seccion_detalle(df: pl.DataFrame, agregacion: str) -> None:
    st.subheader("Detalle por periodo :gray-badge[:material/table: Tabla]")
    st.caption(
        f"Una fila por {agregacion.lower()} · ordenable por columna · "
        "descarga en CSV desde el menú de la tabla"
    )
    tabla = metricas.detalle_por_periodo(df, filtros.AGREGACIONES[agregacion])
    columnas = {
        "Periodo": st.column_config.DatetimeColumn(
            format=FORMATO_PERIODO[agregacion], pinned=True
        ),
        "Cobertura (%)": st.column_config.ProgressColumn(
            "Cobertura",
            help="Porcentaje de minutos con medición",
            format="%.1f %%",
            min_value=0,
            max_value=100,
            color="green",
            width="small",
        ),
    }
    # Etiqueta corta en pantalla; la unidad va en el tooltip y en el nombre del CSV
    for nombre, etiqueta, decimales in [
        ("Energía (kWh)", "Energía", 1),
        ("Potencia media (kW)", "Pot. media", 2),
        ("Pico (kW)", "Pico", 2),
        ("Cocina (kWh)", "Cocina", 1),
        ("Lavandería (kWh)", "Lavandería", 1),
        ("Calentador/AC (kWh)", "Calent./AC", 1),
        ("No submedido (kWh)", "No submedido", 1),
    ]:
        columnas[nombre] = st.column_config.NumberColumn(
            etiqueta, help=nombre, format=f"%.{decimales}f", width="small"
        )
    st.dataframe(tabla, hide_index=True, height=420, column_config=columnas)


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
        seccion_tendencia_y_usos(filtrado, seleccion.agregacion)
        seccion_perfil(filtrado)
        seccion_distribucion(filtrado)
        seccion_detalle(filtrado, seleccion.agregacion)

with preparacion:
    st.caption("Calidad de los datos, correlación de Pearson y PCA.")
