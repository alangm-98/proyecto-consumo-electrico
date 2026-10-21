"""Normalización z-score sin fuga de información.

- Partición cronológica: entrenamiento antes de `INICIO_PRUEBA`, prueba desde esa fecha.
- Media y desviación estándar (poblacional, ddof=0) calculadas solo con entrenamiento
  y aplicadas a todo el dataset.
- Las variables originales se conservan; las normalizadas se agregan con sufijo `_z`.

Guarda el dataset normalizado en Parquet y los parámetros en JSON, e imprime un resumen.
"""

import json
from datetime import datetime

import polars as pl

from config.rutas import ARCHIVO_LIMPIO, ARCHIVO_NORMALIZADO, PARAMETROS_ZSCORE
from src.cleaning.carga import COLUMNAS_NUMERICAS

INICIO_PRUEBA = datetime(2010, 1, 1)
VARIABLES = [*COLUMNAS_NUMERICAS, "no_submedido_wh"]
SUFIJO = "_z"


def particionar(
    df: pl.DataFrame, inicio_prueba: datetime = INICIO_PRUEBA
) -> pl.DataFrame:
    """Agrega la columna `particion` (entrenamiento / prueba) según la fecha."""
    return df.with_columns(
        pl.when(pl.col("datetime") < inicio_prueba)
        .then(pl.lit("entrenamiento"))
        .otherwise(pl.lit("prueba"))
        .cast(pl.Enum(["entrenamiento", "prueba"]))
        .alias("particion")
    )


def ajustar_zscore(df: pl.DataFrame, variables: list[str] = VARIABLES) -> dict:
    """Media y desviación estándar (ddof=0) de cada variable, ignorando nulos."""
    stats = df.select(
        *[pl.col(v).cast(pl.Float64).mean().alias(f"{v}__media") for v in variables],
        *[
            pl.col(v).cast(pl.Float64).std(ddof=0).alias(f"{v}__desv")
            for v in variables
        ],
    ).row(0, named=True)
    return {
        v: {"media": stats[f"{v}__media"], "desv": stats[f"{v}__desv"]}
        for v in variables
    }


def aplicar_zscore(df: pl.DataFrame, parametros: dict) -> pl.DataFrame:
    """Agrega `<variable>_z` = (x - media) / desv con los parámetros dados."""
    return df.with_columns(
        ((pl.col(v) - p["media"]) / p["desv"]).cast(pl.Float32).alias(f"{v}{SUFIJO}")
        for v, p in parametros.items()
    )


def revertir_zscore(df: pl.DataFrame, parametros: dict) -> pl.DataFrame:
    """Recupera las variables originales a partir de las columnas `_z`."""
    return df.with_columns(
        (pl.col(f"{v}{SUFIJO}") * p["desv"] + p["media"]).alias(v)
        for v, p in parametros.items()
    )


def resumen_por_particion(df: pl.DataFrame, variables: list[str]) -> pl.DataFrame:
    """Media y desviación de las columnas `_z` en cada partición."""
    filas = []
    for parte in ["entrenamiento", "prueba"]:
        sub = df.filter(pl.col("particion") == parte)
        for v in variables:
            s = sub[f"{v}{SUFIJO}"].drop_nulls()
            filas.append(
                {
                    "particion": parte,
                    "variable": v,
                    "media_z": s.mean(),
                    "desv_z": s.std(ddof=0),
                    "min_z": s.min(),
                    "max_z": s.max(),
                }
            )
    return pl.DataFrame(filas).with_columns(pl.selectors.float().round(3))


def comparar_con_fuga(df: pl.DataFrame, parametros: dict) -> pl.DataFrame:
    """Parámetros con solo entrenamiento frente a los calculados con todo el dataset."""
    con_fuga = ajustar_zscore(df, list(parametros))
    return pl.DataFrame(
        [
            {
                "variable": v,
                "media_entren": p["media"],
                "media_todo": con_fuga[v]["media"],
                "desv_entren": p["desv"],
                "desv_todo": con_fuga[v]["desv"],
            }
            for v, p in parametros.items()
        ]
    ).with_columns(pl.selectors.float().round(4))


def main() -> None:
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    df = particionar(pl.read_parquet(ARCHIVO_LIMPIO))
    entrenamiento = df.filter(pl.col("particion") == "entrenamiento")
    parametros = ajustar_zscore(entrenamiento)
    df = aplicar_zscore(df, parametros)

    df.write_parquet(ARCHIVO_NORMALIZADO, compression="zstd")
    registro = {
        "metodo": "z-score",
        "ddof": 0,
        "inicio_prueba": INICIO_PRUEBA.isoformat(),
        "filas_entrenamiento": entrenamiento.height,
        "parametros": parametros,
    }
    PARAMETROS_ZSCORE.write_text(json.dumps(registro, indent=2), encoding="utf-8")

    print(df["particion"].value_counts().sort("particion"))
    print("\nParámetros (solo entrenamiento):")
    for v, p in parametros.items():
        print(f"  {v:<24} media={p['media']:>10.4f}   desv={p['desv']:>9.4f}")
    print("\nColumnas _z por partición:")
    print(resumen_por_particion(df, VARIABLES))
    print("\nParámetros sin fuga frente a parámetros con fuga (todo el dataset):")
    print(comparar_con_fuga(df, parametros))

    max_error = (
        revertir_zscore(df.select(f"{v}{SUFIJO}" for v in VARIABLES), parametros)
        .select((pl.col(v) - df[v]).abs().max() for v in VARIABLES)
        .max_horizontal()
        .item()
    )
    print(f"\nError máximo al revertir la transformación: {max_error:.2e}")
    print(
        f"Guardado: {ARCHIVO_NORMALIZADO} ({ARCHIVO_NORMALIZADO.stat().st_size / 1e6:.1f} MB)"
    )
    print(f"Guardado: {PARAMETROS_ZSCORE}")


if __name__ == "__main__":
    main()
