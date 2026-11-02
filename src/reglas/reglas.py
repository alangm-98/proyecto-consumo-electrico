"""Reglas de asociación a partir de los itemsets frecuentes.

- Itemsets frecuentes con `MIN_SUPPORT` (FP-Growth; Apriori da los mismos).
- Reglas con confianza mayor o igual a `MIN_CONFIDENCE`; se conservan las que tienen
  lift > 1 y cuyo consecuente solo contiene habitaciones o `Consumo alto` (las reglas
  que predicen un día o una franja no aportan: el calendario se conoce de antemano).
- Tipo de regla: «rutina» si el antecedente solo tiene ítems temporales, «uso» si
  incluye habitaciones o consumo.

Guarda la tabla de reglas en CSV, ordenada por lift, e imprime un resumen.
"""

import math

import pandas as pd
import polars as pl
from mlxtend.frequent_patterns import association_rules

from config.rutas import ARCHIVO_REGLAS
from src.reglas.algoritmos import cargar_transacciones, codificar, itemsets_frecuentes
from src.reglas.transacciones import CONSUMO_ALTO, DIAS, FRANJAS, UMBRALES_WH

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.4

ITEMS_CONSECUENTE = {*UMBRALES_WH, CONSUMO_ALTO}
# Orden de lectura de los ítems dentro de una regla
ORDEN = [*UMBRALES_WH, CONSUMO_ALTO, *(f[0] for f in FRANJAS), *DIAS]


def texto(itemset: frozenset) -> str:
    return ", ".join(sorted(itemset, key=ORDEN.index))


def generar_reglas(tabla: pd.DataFrame) -> pd.DataFrame:
    """Todas las reglas que cumplen el soporte y la confianza mínimos."""
    itemsets = itemsets_frecuentes(tabla, MIN_SUPPORT)
    return association_rules(
        itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE
    )


def filtrar(reglas: pd.DataFrame) -> pd.DataFrame:
    """Lift > 1 y consecuente formado solo por habitaciones o consumo alto."""
    consecuente_util = reglas["consequents"].map(lambda c: c <= ITEMS_CONSECUENTE)
    return reglas[(reglas["lift"] > 1) & consecuente_util]


def tabla_reglas(reglas: pd.DataFrame, transacciones: int) -> pl.DataFrame:
    """Tabla legible, ordenada por lift (y por soporte en empates)."""
    return (
        pl.DataFrame(
            {
                "antecedente": [texto(a) for a in reglas["antecedents"]],
                "consecuente": [texto(c) for c in reglas["consequents"]],
                "tipo": [
                    "uso" if a & ITEMS_CONSECUENTE else "rutina"
                    for a in reglas["antecedents"]
                ],
                "soporte": reglas["support"].to_numpy(),
                "confianza": reglas["confidence"].to_numpy(),
                "lift": reglas["lift"].to_numpy(),
                "soporte_antecedente": reglas["antecedent support"].to_numpy(),
                "soporte_consecuente": reglas["consequent support"].to_numpy(),
            }
        )
        .with_columns(
            (pl.col("soporte") * transacciones).round(0).cast(pl.Int32).alias("horas"),
            pl.col("soporte", "confianza", "lift").round(4),
            pl.col("soporte_antecedente", "soporte_consecuente").round(4),
        )
        .sort(["lift", "soporte"], descending=True)
    )


def main() -> None:
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_fmt_str_lengths(60)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    tabla = codificar(cargar_transacciones())
    todas = generar_reglas(tabla)
    elegidas = filtrar(todas)
    reglas = tabla_reglas(elegidas, len(tabla))

    print(
        f"min_support = {MIN_SUPPORT} (≥ {math.ceil(MIN_SUPPORT * len(tabla)):,} horas)   "
        f"min_confidence = {MIN_CONFIDENCE}"
    )
    print(f"Reglas que cumplen soporte y confianza: {len(todas):,}")
    print(f"  con lift > 1: {(todas['lift'] > 1).sum():,}")
    print(f"  con lift > 1 y consecuente de habitación o consumo: {len(reglas):,}")
    print(reglas.group_by("tipo").len().sort("tipo"))

    columnas = ["antecedente", "consecuente", "soporte", "confianza", "lift", "horas"]
    for tipo in ["uso", "rutina"]:
        print(f"\nReglas de {tipo} con mayor lift:")
        print(reglas.filter(pl.col("tipo") == tipo).select(columnas).head(10))

    reglas.write_csv(ARCHIVO_REGLAS)
    print(f"\nGuardado: {ARCHIVO_REGLAS} ({reglas.height} reglas)")


if __name__ == "__main__":
    main()
