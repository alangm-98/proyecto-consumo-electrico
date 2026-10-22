"""Datos que consume el dashboard.

- Resumen por hora del dataset limpio, en unidades reales: energía, potencias,
  voltaje, intensidad, submedidores, minutos medidos y calendario.
- Matriz de correlación de Pearson del periodo de entrenamiento (Paso 4).

Guarda ambos archivos en data/data-processed/ e imprime un resumen.
"""

import json

import polars as pl

from config.rutas import ARCHIVO_HORARIO, ARCHIVO_LIMPIO, ARCHIVO_PEARSON
from src.features.correlacion import (
    cargar_entrenamiento,
    matriz_correlacion,
)

CALENDARIO = [
    "anio",
    "hora",
    "dia_semana",
    "mes",
    "estacion",
    "franja_horaria",
    "es_fin_semana",
]


def resumen_horario(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega el dataset limpio por hora.

    Los promedios usan solo los minutos con medición; la energía es la suma de la
    potencia activa de esos minutos (kW·min / 60 = kWh). Las horas sin ninguna
    medición quedan con nulos y `minutos_medidos = 0`.
    """
    medido = pl.col("Global_active_power").is_not_null()
    return (
        df.group_by(pl.col("datetime").dt.truncate("1h"), maintain_order=True)
        .agg(
            (pl.col("Global_active_power").sum() / 60).alias("energia_kwh"),
            pl.col("Global_active_power").mean().alias("potencia_media_kw"),
            pl.col("Global_active_power").max().alias("potencia_max_kw"),
            pl.col("Global_reactive_power").mean().alias("reactiva_media_kw"),
            pl.col("Voltage").mean().alias("voltaje_medio_v"),
            pl.col("Global_intensity").mean().alias("intensidad_media_a"),
            (pl.col("Sub_metering_1").sum() / 1000).alias("cocina_kwh"),
            (pl.col("Sub_metering_2").sum() / 1000).alias("lavanderia_kwh"),
            (pl.col("Sub_metering_3").sum() / 1000).alias("calentador_ac_kwh"),
            (pl.col("no_submedido_wh").sum() / 1000).alias("no_submedido_kwh"),
            medido.sum().cast(pl.Int16).alias("minutos_medidos"),
            pl.len().cast(pl.Int16).alias("minutos"),
            *[pl.col(c).first() for c in CALENDARIO],
        )
        .with_columns(
            # Horas sin ninguna medición: nulos en lugar de ceros
            pl.when(pl.col("minutos_medidos") > 0)
            .then(pl.col(c))
            .otherwise(None)
            .cast(pl.Float32)
            .alias(c)
            for c in [
                "energia_kwh",
                "cocina_kwh",
                "lavanderia_kwh",
                "calentador_ac_kwh",
                "no_submedido_kwh",
            ]
        )
        .with_columns(pl.selectors.float().cast(pl.Float32))
        .sort("datetime")
    )


def pearson_entrenamiento() -> dict:
    """Matriz de Pearson (entrenamiento, sin faltantes) como diccionario serializable."""
    matriz = matriz_correlacion(cargar_entrenamiento())
    variables = matriz["variable"].to_list()
    return {
        "variables": variables,
        "r": [
            [round(float(v), 4) for v in fila]
            for fila in matriz.drop("variable").rows()
        ],
    }


def main() -> None:
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    limpio = pl.read_parquet(ARCHIVO_LIMPIO)
    horario = resumen_horario(limpio)
    horario.write_parquet(ARCHIVO_HORARIO, compression="zstd")
    ARCHIVO_PEARSON.write_text(
        json.dumps(pearson_entrenamiento(), indent=2), encoding="utf-8"
    )

    print(f"Horas: {horario.height:,}   Columnas: {horario.width}")
    print("\nEsquema:")
    for c, t in horario.schema.items():
        print(f"  {c:<22} {t}")

    sin_medicion = horario.filter(pl.col("minutos_medidos") == 0).height
    print(f"\nHoras sin ninguna medición: {sin_medicion:,}")
    print(
        "Comprobación contra el dataset por minuto "
        "(deben coincidir: energía total, potencia máxima, minutos medidos):"
    )
    por_minuto = limpio.select(
        (pl.col("Global_active_power").sum() / 60).alias("energia_kwh"),
        pl.col("Global_active_power").max().alias("potencia_max_kw"),
        pl.col("Global_active_power").is_not_null().sum().alias("minutos_medidos"),
    )
    por_hora = horario.select(
        pl.col("energia_kwh").sum(),
        pl.col("potencia_max_kw").max(),
        pl.col("minutos_medidos").cast(pl.Int64).sum(),
    )
    print(
        pl.concat([por_minuto, por_hora], how="vertical_relaxed").with_columns(
            pl.Series("origen", ["por minuto", "por hora"])
        )
    )

    print(
        f"\nGuardado: {ARCHIVO_HORARIO} ({ARCHIVO_HORARIO.stat().st_size / 1e6:.2f} MB)"
    )
    print(f"Guardado: {ARCHIVO_PEARSON}")


if __name__ == "__main__":
    main()
