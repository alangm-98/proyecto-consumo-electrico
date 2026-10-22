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


def resumen_estadistico(diario: pl.DataFrame) -> pl.DataFrame:
    """Medidas de localización, dispersión y forma del consumo diario (kWh), como texto."""
    e = diario["energia_kwh"].cast(pl.Float64)
    if e.is_empty():
        return pl.DataFrame(schema={"Medida": pl.String, "Valor": pl.String})
    q1, q3 = e.quantile(0.25), e.quantile(0.75)
    medidas = {
        "Media": f"{e.mean():.2f}",
        "Mediana": f"{e.median():.2f}",
        "Desviación estándar": f"{e.std():.2f}" if e.len() > 1 else "—",
        "IQR": f"{q3 - q1:.2f}",
        "Coef. de variación": f"{e.std() / e.mean():.2f}" if e.len() > 1 else "—",
        "Asimetría": f"{e.skew():.2f}" if e.len() > 2 else "—",
        "Mínimo": f"{e.min():.2f}",
        "Máximo": f"{e.max():.2f}",
        "Días completos": f"{e.len():,}",
    }
    return pl.DataFrame({"Medida": list(medidas), "Valor": list(medidas.values())})


def perfil_diario(df: pl.DataFrame) -> pl.DataFrame:
    """Potencia media (kW) por hora del día, separada en laborable y fin de semana."""
    medidos = pl.col("minutos_medidos").cast(pl.Int64).sum()
    return (
        df.group_by("hora", "es_fin_semana")
        .agg(
            pl.when(medidos > 0)
            .then(pl.col("energia_kwh").sum() * 60 / medidos)
            .otherwise(None)
            .alias("potencia_kw")
        )
        .sort("es_fin_semana", "hora")
    )


def detalle_por_periodo(df: pl.DataFrame, cada: str) -> pl.DataFrame:
    """Tabla por periodo: energía, potencias, usos y cobertura.

    Los periodos sin ningún minuto medido quedan vacíos (no en cero).
    """
    medidos = pl.col("minutos_medidos").cast(pl.Int64).sum()
    valores = {
        "Energía (kWh)": pl.col("energia_kwh").sum(),
        "Potencia media (kW)": pl.col("energia_kwh").sum() * 60 / medidos,
        "Pico (kW)": pl.col("potencia_max_kw").max(),
        "Cocina (kWh)": pl.col("cocina_kwh").sum(),
        "Lavandería (kWh)": pl.col("lavanderia_kwh").sum(),
        "Calentador/AC (kWh)": pl.col("calentador_ac_kwh").sum(),
        "No submedido (kWh)": pl.col("no_submedido_kwh").sum(),
    }
    return (
        df.sort("datetime")
        .group_by_dynamic("datetime", every=cada)
        .agg(
            medidos.alias("_medidos"),
            *[
                pl.when(medidos > 0).then(expr).otherwise(None).alias(nombre)
                for nombre, expr in valores.items()
            ],
            (100 * medidos / pl.col("minutos").cast(pl.Int64).sum()).alias(
                "Cobertura (%)"
            ),
        )
        .drop("_medidos")
        .rename({"datetime": "Periodo"})
    )
