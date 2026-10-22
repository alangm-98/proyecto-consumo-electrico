"""Gráficas Plotly del dashboard.

Los colores de fondo, texto y rejilla los aplica el tema de Streamlit
(st.plotly_chart con theme="streamlit"); aquí solo se fijan los colores con
significado, tomados de la paleta Mocha.
"""

from datetime import datetime

import plotly.graph_objects as go
import polars as pl

from src.dashboard.filtros import AGREGACIONES
from src.dashboard.tema import MOCHA, mezclar

BANDA_INVIERNO = mezclar(MOCHA["sapphire"], MOCHA["base"], 0.14)
BANDA_VERANO = mezclar(MOCHA["peach"], MOCHA["base"], 0.16)

USOS = {
    "no_submedido_kwh": ("No submedido", MOCHA["lavender"]),
    "calentador_ac_kwh": ("Calentador/AC (S3)", MOCHA["peach"]),
    "lavanderia_kwh": ("Lavandería (S2)", MOCHA["teal"]),
    "cocina_kwh": ("Cocina (S1)", MOCHA["yellow"]),
}


MESES = [
    "ene",
    "feb",
    "mar",
    "abr",
    "may",
    "jun",
    "jul",
    "ago",
    "sep",
    "oct",
    "nov",
    "dic",
]


def fechas_es(fechas: pl.Series, formato: str) -> list[str | None]:
    """Fechas como texto con el mes abreviado en español (`%b` en `formato`)."""
    return [
        None if f is None else f.strftime(formato.replace("%b", MESES[f.month - 1]))
        for f in fechas
    ]


def _base(fig: go.Figure, alto: int) -> go.Figure:
    """Márgenes, alto y estilo de tooltip comunes."""
    return fig.update_layout(
        height=alto,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        hoverlabel={"bgcolor": MOCHA["mantle"], "bordercolor": MOCHA["surface1"]},
        showlegend=False,
    )


def serie_consumo(df: pl.DataFrame, agregacion: str) -> pl.DataFrame:
    """Consumo por periodo, con los periodos sin datos como nulos.

    Hora: kWh en la hora. Día, semana y mes: kWh por día promedio del periodo,
    calculado con la potencia media de los minutos medidos (no se subestima
    cuando faltan horas).
    """
    cada = AGREGACIONES[agregacion]
    agrupado = (
        df.sort("datetime")
        .group_by_dynamic("datetime", every=cada)
        .agg(
            pl.col("energia_kwh").sum(),
            pl.col("minutos_medidos").cast(pl.Int64).sum(),
        )
        .with_columns(
            pl.when(pl.col("minutos_medidos") > 0)
            .then(
                pl.col("energia_kwh")
                * 60
                / pl.col("minutos_medidos")
                * (1 if agregacion == "Hora" else 24)
            )
            .alias("consumo")
        )
    )
    # Rejilla completa de periodos para que los huecos de los filtros no se unan con líneas
    rejilla = pl.datetime_range(
        agrupado["datetime"].min(),
        agrupado["datetime"].max(),
        interval=cada,
        eager=True,
    ).alias("datetime")
    return rejilla.to_frame().join(agrupado, on="datetime", how="left")


def _bandas_estacion(fig: go.Figure, inicio: datetime, fin: datetime) -> None:
    """Sombrea inviernos (dic–feb) y veranos (jun–ago) dentro del rango."""
    for anio in range(inicio.year - 1, fin.year + 1):
        for x0, x1, color in [
            (datetime(anio, 12, 1), datetime(anio + 1, 3, 1), BANDA_INVIERNO),
            (datetime(anio + 1, 6, 1), datetime(anio + 1, 9, 1), BANDA_VERANO),
        ]:
            x0, x1 = max(x0, inicio), min(x1, fin)
            if x0 < x1:
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=color,
                    opacity=1,
                    line_width=0,
                    layer="below",
                )


def tendencia(df: pl.DataFrame, agregacion: str) -> go.Figure:
    serie = serie_consumo(df, agregacion)
    unidad = "kWh en la hora" if agregacion == "Hora" else "kWh por día"
    formato_x = {"Hora": "%d %b %Y %H h", "Día": "%d %b %Y", "Semana": "%d %b %Y"}
    traza = go.Scattergl if serie.height > 5000 else go.Scatter
    fig = go.Figure(
        traza(
            x=serie["datetime"],
            y=serie["consumo"],
            mode="lines",
            line={"color": MOCHA["blue"], "width": 1 if agregacion == "Hora" else 2},
            connectgaps=False,
            customdata=fechas_es(serie["datetime"], formato_x.get(agregacion, "%b %Y")),
            hovertemplate=f"%{{customdata}}<br><b>%{{y:.1f}}</b> {unidad}<extra></extra>",
        )
    )
    _bandas_estacion(fig, serie["datetime"].min(), serie["datetime"].max())
    fig.update_yaxes(title_text=unidad, rangemode="tozero")
    fig.update_xaxes(title_text=None)
    return _base(fig, 340)


def energia_por_uso(df: pl.DataFrame) -> go.Figure:
    """Porcentaje de la energía total por uso, de mayor a menor."""
    totales = df.select(pl.col(c).sum() for c in USOS).row(0, named=True)
    suma = sum(v for v in totales.values() if v) or 1
    filas = sorted(
        ((USOS[c][0], 100 * (v or 0) / suma, USOS[c][1]) for c, v in totales.items()),
        key=lambda f: f[1],
    )
    nombres, pct, colores = zip(*filas)
    fig = go.Figure(
        go.Bar(
            x=pct,
            y=nombres,
            orientation="h",
            marker={"color": colores, "cornerradius": 4},
            text=[f"{p:.1f} %" for p in pct],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: <b>%{x:.1f} %</b><extra></extra>",
        )
    )
    fig.update_xaxes(visible=False, range=[0, max(pct) * 1.4])
    fig.update_yaxes(title_text=None)
    return _base(fig, 300)


def perfil_diario(perfil: pl.DataFrame) -> go.Figure:
    """Potencia media por hora del día: laborable (Blue) y fin de semana (Peach)."""
    fig = go.Figure()
    for fin_semana, nombre, color in [
        (False, "Laborable", MOCHA["blue"]),
        (True, "Fin de semana", MOCHA["peach"]),
    ]:
        serie = perfil.filter(pl.col("es_fin_semana") == fin_semana)
        if serie.is_empty():
            continue
        fig.add_trace(
            go.Scatter(
                x=serie["hora"],
                y=serie["potencia_kw"],
                name=nombre,
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 6},
                hovertemplate=f"{nombre} · %{{x}}:00 h<br><b>%{{y:.2f}} kW</b><extra></extra>",
            )
        )
    fig.update_xaxes(
        title_text="Hora del día",
        tickmode="array",
        tickvals=list(range(0, 24, 3)),
        ticktext=[f"{h} h" for h in range(0, 24, 3)],
        range=[-0.5, 23.5],
    )
    fig.update_yaxes(title_text="kW promedio", rangemode="tozero")
    _base(fig, 320)
    return fig.update_layout(
        showlegend=True,
        legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.08},
    )


def histograma_diario(diario: pl.DataFrame) -> go.Figure:
    """Distribución del consumo diario (kWh) con media y mediana marcadas."""
    energia = diario["energia_kwh"]
    fig = go.Figure(
        go.Histogram(
            x=energia,
            xbins={"start": 0, "size": 4},
            marker={
                "color": MOCHA["mauve"],
                "line": {"color": MOCHA["base"], "width": 1},
            },
            hovertemplate="%{x} kWh<br><b>%{y} días</b><extra></extra>",
        )
    )
    for valor, color, guion in [
        (energia.mean(), MOCHA["text"], "solid"),
        (energia.median(), MOCHA["peach"], "dash"),
    ]:
        if valor is not None:
            fig.add_vline(x=valor, line={"color": color, "width": 2, "dash": guion})
    fig.update_xaxes(title_text="kWh por día")
    fig.update_yaxes(title_text="Días")
    return _base(fig, 320).update_layout(bargap=0.04)


# Variables de Pearson: etiqueta corta, columna del resumen por hora y unidad en esa columna
VARIABLES_PEARSON = {
    "Global_active_power": ("Pot. activa", "potencia_media_kw", "kW"),
    "Global_reactive_power": ("Pot. reactiva", "reactiva_media_kw", "kW"),
    "Voltage": ("Voltaje", "voltaje_medio_v", "V"),
    "Global_intensity": ("Intensidad", "intensidad_media_a", "A"),
    "Sub_metering_1": ("Cocina", "cocina_kwh", "kWh"),
    "Sub_metering_2": ("Lavandería", "lavanderia_kwh", "kWh"),
    "Sub_metering_3": ("Calent./AC", "calentador_ac_kwh", "kWh"),
    "no_submedido_wh": ("No submed.", "no_submedido_kwh", "kWh"),
}
ESCALA_DIVERGENTE = [[0, MOCHA["blue"]], [0.5, MOCHA["surface0"]], [1, MOCHA["red"]]]


def faltantes_mes(serie: pl.DataFrame) -> go.Figure:
    """Minutos sin medición por mes (Red), con el mes máximo anotado."""
    fig = go.Figure(
        go.Bar(
            x=serie["mes"],
            y=serie["minutos_faltantes"],
            marker={"color": MOCHA["red"], "cornerradius": 3},
            customdata=fechas_es(serie["mes"], "%b %Y"),
            hovertemplate="%{customdata}<br><b>%{y:,} min</b><extra></extra>",
        )
    )
    if serie["minutos_faltantes"].max():
        pico = serie.sort("minutos_faltantes").tail(1).row(0, named=True)
        fig.add_annotation(
            x=pico["mes"],
            y=pico["minutos_faltantes"],
            text=f"{fechas_es(pl.Series([pico['mes']]), '%b %Y')[0]} · "
            f"{pico['minutos_faltantes']:,} min",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font={"color": MOCHA["subtext0"], "size": 12},
        )
    fig.update_yaxes(title_text="Minutos sin medición", tickformat=",")
    fig.update_xaxes(title_text=None)
    return _base(fig, 300).update_layout(bargap=0.2)


def matriz_pearson(pearson: dict, x: str, y: str) -> go.Figure:
    """Matriz de Pearson con la diagonal vacía y el par elegido enmarcado (Lavender)."""
    variables = pearson["variables"]
    etiquetas = [VARIABLES_PEARSON[v][0] for v in variables]
    n = len(variables)
    valores = [
        [None if i == j else r for j, r in enumerate(fila)]
        for i, fila in enumerate(pearson["r"])
    ]
    texto = [
        ["" if r is None else f"{r:.3f}" if abs(r) > 0.8 else f"{r:.2f}" for r in fila]
        for fila in valores
    ]
    fig = go.Figure(
        go.Heatmap(
            z=valores,
            x=etiquetas,
            y=etiquetas,
            text=texto,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=ESCALA_DIVERGENTE,
            zmin=-1,
            zmax=1,
            xgap=2,
            ygap=2,
            colorbar={"thickness": 10, "tickvals": [-1, -0.5, 0, 0.5, 1]},
            hovertemplate="%{y} · %{x}<br><b>r = %{z:.3f}</b><extra></extra>",
            hoverongaps=False,
        )
    )
    i, j = variables.index(y), variables.index(x)
    for fila, columna in {(i, j), (j, i)}:
        fig.add_shape(
            type="rect",
            x0=columna - 0.5,
            x1=columna + 0.5,
            y0=fila - 0.5,
            y1=fila + 0.5,
            line={"color": MOCHA["lavender"], "width": 2},
        )
    fig.update_xaxes(tickangle=-40, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _base(fig, 420).update_layout(
        plot_bgcolor=MOCHA["mantle"], xaxis_range=[-0.5, n - 0.5]
    )


def dispersion(df: pl.DataFrame, x: str, y: str) -> go.Figure:
    """Promedios (o sumas, en submedidores) por hora de dos variables (Sapphire)."""
    nombre_x, columna_x, unidad_x = VARIABLES_PEARSON[x]
    nombre_y, columna_y, unidad_y = VARIABLES_PEARSON[y]
    puntos = df.select(columna_x, columna_y).drop_nulls()
    fig = go.Figure(
        go.Scattergl(
            x=puntos[columna_x],
            y=puntos[columna_y],
            mode="markers",
            marker={"color": MOCHA["sapphire"], "size": 3, "opacity": 0.25},
            hovertemplate=f"{nombre_x}: %{{x:.2f}} {unidad_x}<br>"
            f"{nombre_y}: %{{y:.2f}} {unidad_y}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text=f"{nombre_x} ({unidad_x})")
    fig.update_yaxes(title_text=f"{nombre_y} ({unidad_y})")
    return _base(fig, 340)
