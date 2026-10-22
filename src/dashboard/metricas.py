"""Indicadores calculados sobre el resumen por hora (ya filtrado)."""

import polars as pl

# Un día cuenta como completo si tiene al menos 23 horas medidas
MINUTOS_DIA_COMPLETO = 1380


def consumo_diario(df: pl.DataFrame) -> pl.DataFrame:
    """Energía por día (kWh), solo de los días completos."""
    return (
        df.group_by(pl.col("datetime").dt.date().alias("fecha"))
        .agg(
            pl.col("energia_kwh").sum().alias("energia_kwh"),
            pl.col("minutos_medidos").cast(pl.Int32).sum().alias("minutos_medidos"),
        )
        .filter(pl.col("minutos_medidos") >= MINUTOS_DIA_COMPLETO)
        .sort("fecha")
    )


def potencia_media(df: pl.DataFrame) -> float | None:
    """Potencia activa promedio (kW) ponderada por los minutos medidos."""
    minutos = df["minutos_medidos"].cast(pl.Int64).sum()
    return df["energia_kwh"].sum() * 60 / minutos if minutos else None


def kpis(df: pl.DataFrame) -> dict:
    """Valores de los cuatro indicadores de la pestaña de consumo."""
    diario = consumo_diario(df)
    medidos = df["minutos_medidos"].cast(pl.Int64).sum()
    totales = df["minutos"].cast(pl.Int64).sum()

    laborable = potencia_media(df.filter(~pl.col("es_fin_semana")))
    fin_semana = potencia_media(df.filter(pl.col("es_fin_semana")))
    diferencia = (
        100 * (fin_semana / laborable - 1) if laborable and fin_semana else None
    )

    pico = df.drop_nulls("potencia_max_kw").sort("potencia_max_kw").tail(1)
    mensual = (
        diario.group_by(pl.col("fecha").dt.truncate("1mo"))
        .agg(pl.col("energia_kwh").mean())
        .sort("fecha")["energia_kwh"]
        .to_list()
    )
    return {
        "dias_completos": diario.height,
        "consumo_diario_media": diario["energia_kwh"].mean(),
        "consumo_diario_mediana": diario["energia_kwh"].median(),
        "consumo_mensual": mensual,
        "potencia_media": potencia_media(df),
        "diferencia_fin_semana": diferencia,
        "potencia_maxima": pico["potencia_max_kw"].item() if pico.height else None,
        "fecha_maxima": pico["datetime"].item() if pico.height else None,
        "cobertura": 100 * medidos / totales if totales else None,
        "minutos_faltantes": totales - medidos,
    }
