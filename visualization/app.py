"""Dashboard de consumo eléctrico doméstico (Streamlit)."""

import json
import sys
from pathlib import Path

# Streamlit agrega al path la carpeta de este archivo; se agrega la raíz para importar src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
import streamlit as st

from config.rutas import (
    ARCHIVO_CALIDAD,
    ARCHIVO_HORARIO,
    ARCHIVO_PEARSON,
    FIG_LOADINGS,
    FIG_PROYECCION,
    FIG_SCREE,
    PARAMETROS_PCA,
)
from src.dashboard import filtros, graficas, metricas
from src.dashboard.tema import ICONO, MOCHA
from src.features.correlacion import ETIQUETAS

st.set_page_config(
    page_title="Consumo eléctrico · Práctica 5",
    page_icon=ICONO.read_text(encoding="utf-8"),
    layout="wide",
)


@st.cache_data
def cargar_horario() -> pl.DataFrame:
    return pl.read_parquet(ARCHIVO_HORARIO)


@st.cache_data
def cargar_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


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


# ------------------------------------------------------------ preparación de datos
def seccion_calidad(calidad: dict, pearson: dict, pca: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    repetidas = calidad["registros"] - calidad["marcas_unicas"]
    c1.metric(
        ":blue[:material/database:] Registros",
        f"{calidad['registros']:,}",
        delta=f"{repetidas:,} minutos repetidos",
        delta_color="off",
        delta_arrow="off",
        help="Una fila por minuto, del 16/12/2006 al 26/11/2010.",
        border=True,
        height="stretch",
    )
    c2.metric(
        ":yellow[:material/auto_fix_high:] Minutos interpolados",
        f"{calidad['interpolados']:,}",
        delta=f"Huecos de hasta {calidad['max_minutos_interpolados']} min",
        delta_color="yellow",
        delta_arrow="off",
        help="Interpolación lineal solo en huecos cortos.",
        border=True,
        height="stretch",
    )
    c3.metric(
        ":red[:material/error:] Minutos faltantes",
        f"{calidad['faltantes']:,}",
        delta=f"{len(calidad['huecos'])} huecos largos, sin rellenar",
        delta_color="red",
        delta_arrow="off",
        help="Quedan como nulos; no se usan para ajustar el z-score, Pearson ni el PCA.",
        border=True,
        height="stretch",
    )
    medidas = len(pearson["variables"])
    modelo = len(pca["variables"])
    c4.metric(
        ":violet[:material/filter_list:] Variables para modelar",
        f"{medidas} → {modelo} → {pca['n_componentes']}",
        delta=f"Pearson descarta {medidas - modelo}",
        delta_color="violet",
        delta_arrow="off",
        help=(
            f"{medidas} medidas (incluido el consumo no submedido) → {modelo} tras "
            f"Pearson → {pca['n_componentes']} componentes principales."
        ),
        border=True,
        height="stretch",
    )


COLOR_PASO = ["blue", "blue", "blue", "yellow", "green", "orange", "violet"]


def pasos_preparacion(calidad: dict, pca: dict) -> list[str]:
    acumulada = 100 * sum(pca["varianza_explicada"][: pca["n_componentes"]])
    return [
        "Faltantes «?» y vacíos unificados como nulos",
        "Fecha y hora en una sola columna; mediciones como números",
        f"Deduplicación por minuto ({calidad['registros'] - calidad['marcas_unicas']} repetidas)",
        f"Interpolación solo en huecos de hasta {calidad['max_minutos_interpolados']} min",
        "Z-score ajustado con 2006–2009; 2010 como prueba",
        "Pearson: fuera la intensidad y el consumo no submedido",
        f"PCA: {pca['n_componentes']} componentes, {acumulada:.1f} % de la varianza",
    ]


def seccion_faltantes(horario: pl.DataFrame, calidad: dict, pca: dict) -> None:
    izquierda, derecha = st.columns([2, 1], gap="large")
    with izquierda:
        st.subheader(
            "Minutos sin medición por mes :red-badge[:material/report: Calidad]"
        )
        largos = [h["minutos"] for h in calidad["huecos"] if h["minutos"] > 24 * 60]
        st.caption(
            f"{len(largos)} huecos de más de un día suman el "
            f"{100 * sum(largos) / max(calidad['faltantes'], 1):.0f} % de los minutos faltantes"
        )
        st.plotly_chart(
            graficas.faltantes_mes(metricas.faltantes_por_mes(horario)),
            config=CONFIG_GRAFICA,
        )
        with st.expander(f"Ver los {len(calidad['huecos'])} huecos sin rellenar"):
            st.dataframe(
                metricas.tabla_huecos(calidad["huecos"]),
                hide_index=True,
                column_config={
                    "Inicio": st.column_config.DatetimeColumn(
                        format="YYYY-MM-DD HH:mm"
                    ),
                    "Fin": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                    "Duración (h)": st.column_config.NumberColumn(format="%.1f"),
                },
            )
    with derecha:
        st.subheader("Decisiones de preparación")
        st.caption("En el orden en que se aplicaron")
        for n, (color, texto) in enumerate(
            zip(COLOR_PASO, pasos_preparacion(calidad, pca)), start=1
        ):
            st.markdown(f":{color}-badge[{n}] {texto}")


def seccion_pearson(horario: pl.DataFrame, pearson: dict) -> None:
    st.subheader("Correlación de Pearson :green-badge[:material/hub: Relaciones]")
    st.caption(
        "Minutos de entrenamiento (2006–2009) sin faltantes · "
        "umbral de redundancia |r| > 0.8 · el par elegido aparece enmarcado"
    )
    variables = pearson["variables"]
    izquierda, derecha = st.columns(2, gap="large")
    with derecha:
        cx, cy = st.columns(2)
        x = cx.selectbox(
            "Variable X",
            variables,
            index=variables.index("Global_active_power"),
            format_func=ETIQUETAS.get,
        )
        opciones_y = [v for v in variables if v != x]
        y = cy.selectbox(
            "Variable Y",
            opciones_y,
            index=opciones_y.index("Voltage") if "Voltage" in opciones_y else 0,
            format_func=ETIQUETAS.get,
        )
        r = pearson["r"][variables.index(y)][variables.index(x)]
        columnas = [graficas.VARIABLES_PEARSON[v][1] for v in (x, y)]
        r_horas = horario.select(pl.corr(*columnas)).item()
        st.plotly_chart(graficas.dispersion(horario, x, y), config=CONFIG_GRAFICA)
        st.caption(
            f"Cada punto es una hora (promedio; suma en los submedidores) · "
            f"r por minuto = {r:.2f} · r por hora = {r_horas:.2f} · "
            "la correlación no implica causalidad"
        )
    with izquierda:
        st.plotly_chart(graficas.matriz_pearson(pearson, x, y), config=CONFIG_GRAFICA)


def seccion_pca(pca: dict) -> None:
    st.subheader(
        "Componentes principales (PCA) :yellow-badge[:material/compress: Reducción]"
    )
    st.info(
        "La varianza explicada resume cuánta variación de las entradas conserva cada "
        "componente; no mide capacidad predictiva. El PCA solo captura relaciones "
        "lineales y es sensible a valores atípicos.",
        icon=":material/info:",
    )
    izquierda, derecha = st.columns(2, gap="large")
    izquierda.image(str(FIG_SCREE))
    derecha.image(str(FIG_LOADINGS))
    st.image(str(FIG_PROYECCION))
    with st.expander("Ver loadings en tabla"):
        tabla = metricas.tabla_loadings(pca, ETIQUETAS)
        st.dataframe(
            tabla,
            hide_index=True,
            column_config={
                c: st.column_config.NumberColumn(format="%.3f")
                for c in tabla.columns[1:]
            },
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
        seccion_tendencia_y_usos(filtrado, seleccion.agregacion)
        seccion_perfil(filtrado)
        seccion_distribucion(filtrado)
        seccion_detalle(filtrado, seleccion.agregacion)

with preparacion:
    st.caption("Dataset completo, sin filtros · calidad, correlación de Pearson y PCA")
    calidad = cargar_json(ARCHIVO_CALIDAD)
    pearson = cargar_json(ARCHIVO_PEARSON)
    pca = cargar_json(PARAMETROS_PCA)
    seccion_calidad(calidad, pearson, pca)
    seccion_faltantes(horario, calidad, pca)
    seccion_pearson(horario, pearson)
    seccion_pca(pca)
