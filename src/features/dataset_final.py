"""Consolidación del dataset preparado para modelado.

A partir del dataset normalizado:
- Conserva las variables seleccionadas por Pearson en su versión estandarizada (`_z`).
- Agrega los componentes principales con los parámetros de `parametros_pca.json`.
- Conserva el índice temporal, el origen de cada fila, la partición y el calendario.

Valida el resultado, lo guarda en Parquet e imprime un resumen.
"""

import json

import numpy as np
import polars as pl

from config.rutas import ARCHIVO_FINAL, ARCHIVO_NORMALIZADO, PARAMETROS_PCA
from src.features.pca import COLUMNAS_Z, nombres_pc, proyectar

COLUMNAS_CONTEXTO = [
    "datetime",
    "estado_dato",
    "particion",
    "anio",
    "hora",
    "dia_semana",
    "mes",
    "estacion",
    "franja_horaria",
    "es_fin_semana",
]


def cargar_pca(ruta=PARAMETROS_PCA) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Componentes, media y nombres de columna guardados en el Paso 5."""
    registro = json.loads(ruta.read_text(encoding="utf-8"))
    if registro["variables"] != COLUMNAS_Z:
        raise ValueError("Las variables del PCA no coinciden con las seleccionadas")
    componentes = np.array(registro["componentes"])
    return componentes, np.array(registro["media"]), nombres_pc(len(componentes))


def construir() -> tuple[pl.DataFrame, list[str]]:
    componentes, media, pcs = cargar_pca()
    df = proyectar(pl.read_parquet(ARCHIVO_NORMALIZADO), componentes, media)
    return df.select(*COLUMNAS_CONTEXTO, *COLUMNAS_Z, *pcs), pcs


def validar(df: pl.DataFrame, pcs: list[str]) -> pl.DataFrame:
    """Conteos de nulos por origen y media/desviación de los PCs por partición."""
    nulos_fuera_de_faltantes = (
        df.filter(pl.col("estado_dato") != "faltante")
        .select(pl.sum_horizontal(pl.col(*COLUMNAS_Z, *pcs).is_null()).sum())
        .item()
    )
    if nulos_fuera_de_faltantes:
        raise ValueError(f"{nulos_fuera_de_faltantes} nulos en filas no faltantes")
    if not df["datetime"].is_sorted() or df["datetime"].n_unique() != df.height:
        raise ValueError("El índice temporal no está ordenado o tiene duplicados")
    return (
        df.group_by("particion")
        .agg(
            *[pl.col(pc).cast(pl.Float64).mean().alias(f"media_{pc}") for pc in pcs],
            *[pl.col(pc).cast(pl.Float64).var().alias(f"var_{pc}") for pc in pcs],
        )
        .sort("particion")
        .with_columns(pl.selectors.float().round(3))
    )


def main() -> None:
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    df, pcs = construir()
    resumen = validar(df, pcs)
    df.write_parquet(ARCHIVO_FINAL, compression="zstd")

    print(f"Filas: {df.height:,}   Columnas: {df.width}")
    print("\nEsquema:")
    for c, t in df.schema.items():
        print(f"  {c:<26} {t}")
    print("\nFilas por origen:")
    print(df["estado_dato"].value_counts().sort("estado_dato"))
    print(
        "\nComponentes por partición (entrenamiento: media 0, varianza = valor propio):"
    )
    print(resumen)
    print(f"\nGuardado: {ARCHIVO_FINAL} ({ARCHIVO_FINAL.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
