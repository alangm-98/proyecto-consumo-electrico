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
            hovertemplate=f"%{{x|{formato_x.get(agregacion, '%b %Y')}}}<br>"
            f"<b>%{{y:.1f}}</b> {unidad}<extra></extra>",
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
