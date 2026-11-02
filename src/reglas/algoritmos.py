"""Itemsets frecuentes con Apriori y FP-Growth (mlxtend).

- Codifica las transacciones horarias en una tabla booleana (una columna por ítem).
- Ejecuta ambos algoritmos para cada valor de `SOPORTES`, verifica que encuentren
  los mismos itemsets con el mismo soporte y mide su tiempo (mediana de
  `REPETICIONES` ejecuciones).

Guarda la comparación en CSV y la figura de itemsets y tiempos en PNG.
"""

import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from config.rutas import ARCHIVO_COMPARACION, ARCHIVO_INPUT_REGLAS, FIG_SOPORTE
from src.dashboard.tema import MOCHA
from src.features.correlacion import FONDO, LINEA, TINTA, TINTA_SECUNDARIA
from src.reglas.transacciones import transacciones

SOPORTES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
REPETICIONES = 7
ALGORITMOS = {"Apriori": apriori, "FP-Growth": fpgrowth}


def cargar_transacciones(ruta=ARCHIVO_INPUT_REGLAS) -> list[list[str]]:
    return transacciones(pl.read_parquet(ruta))


def codificar(lista: list[list[str]]) -> pd.DataFrame:
    """Tabla booleana: una fila por transacción, una columna por ítem."""
    codificador = TransactionEncoder()
    return pd.DataFrame(
        codificador.fit(lista).transform(lista), columns=codificador.columns_
    )


def itemsets_frecuentes(tabla: pd.DataFrame, soporte: float, algoritmo="FP-Growth"):
    return ALGORITMOS[algoritmo](tabla, min_support=soporte, use_colnames=True)


def como_conjunto(itemsets: pd.DataFrame) -> set:
    """Itemsets y su soporte (redondeado) como conjunto comparable."""
    return set(zip(map(frozenset, itemsets["itemsets"]), itemsets["support"].round(12)))


def medir(
    tabla: pd.DataFrame, soporte: float, algoritmo: str
) -> tuple[pd.DataFrame, float]:
    """Itemsets frecuentes y mediana del tiempo de ejecución en milisegundos."""
    tiempos = []
    for _ in range(REPETICIONES):
        inicio = time.perf_counter()
        resultado = itemsets_frecuentes(tabla, soporte, algoritmo)
        tiempos.append(1000 * (time.perf_counter() - inicio))
    return resultado, float(np.median(tiempos))


def comparar(tabla: pd.DataFrame) -> pl.DataFrame:
    """Número de itemsets, coincidencia y tiempo de cada algoritmo por soporte mínimo."""
    filas = []
    for soporte in SOPORTES:
        (a, t_a), (f, t_f) = (medir(tabla, soporte, nombre) for nombre in ALGORITMOS)
        filas.append(
            {
                "min_support": soporte,
                "transacciones_minimas": int(np.ceil(soporte * len(tabla))),
                "itemsets_apriori": len(a),
                "itemsets_fpgrowth": len(f),
                "mismos_itemsets": como_conjunto(a) == como_conjunto(f),
                "tamano_maximo": int(a["itemsets"].map(len).max()) if len(a) else 0,
                "ms_apriori": round(t_a, 1),
                "ms_fpgrowth": round(t_f, 1),
            }
        )
    return pl.DataFrame(filas)


def _estilo(ax) -> None:
    ax.set_facecolor(FONDO)
    ax.set_xscale("log")
    ax.grid(color=MOCHA["surface0"], linewidth=0.8, zorder=0)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    for lado in ["left", "bottom"]:
        ax.spines[lado].set_color(LINEA)
    ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=9)
    ax.set_xticks(SOPORTES, [f"{s:g}" for s in SOPORTES])
    ax.minorticks_off()
    ax.set_xlabel("min_support (escala log)", fontsize=9, color=TINTA_SECUNDARIA)


def graficar(
    comparacion: pl.DataFrame, forma: tuple[int, int], ruta=FIG_SOPORTE
) -> None:
    """Itemsets frecuentes y tiempo de ejecución en función del soporte mínimo."""
    soporte = comparacion["min_support"].to_numpy()
    fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150, facecolor=FONDO)

    izq.plot(
        soporte,
        comparacion["itemsets_fpgrowth"],
        color=MOCHA["mauve"],
        linewidth=2.2,
        marker="o",
        zorder=3,
    )
    for s, n, k in comparacion.select(
        "min_support", "itemsets_fpgrowth", "tamano_maximo"
    ).iter_rows():
        izq.annotate(
            f"{n} (k≤{k})",
            (s, n),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=TINTA,
        )
    izq.set_yscale("log")
    _estilo(izq)
    izq.set_ylabel(
        "itemsets frecuentes (escala log)", fontsize=9, color=TINTA_SECUNDARIA
    )
    izq.set_title(
        "Itemsets frecuentes (iguales en ambos algoritmos)",
        fontsize=11,
        loc="left",
        color=TINTA,
    )

    for nombre, columna, color in [
        ("Apriori", "ms_apriori", MOCHA["blue"]),
        ("FP-Growth", "ms_fpgrowth", MOCHA["peach"]),
    ]:
        der.plot(
            soporte,
            comparacion[columna],
            color=color,
            linewidth=2.2,
            marker="o",
            label=nombre,
            zorder=3,
        )
    _estilo(der)
    der.set_ylim(bottom=0)
    der.set_ylabel(
        f"milisegundos (mediana de {REPETICIONES})", fontsize=9, color=TINTA_SECUNDARIA
    )
    der.set_title("Tiempo de ejecución", fontsize=11, loc="left", color=TINTA)
    leyenda = der.legend(frameon=False, fontsize=9)
    for texto in leyenda.get_texts():
        texto.set_color(TINTA)

    fig.suptitle(
        "Apriori y FP-Growth según el soporte mínimo",
        fontsize=12,
        x=0.01,
        ha="left",
        color=TINTA,
    )
    fig.text(
        0.01,
        0.01,
        f"{forma[0]:,} transacciones horarias, {forma[1]} ítems. "
        "k: tamaño máximo de los itemsets. "
        "Tiempos medidos en esta máquina.",
        fontsize=8,
        color=TINTA_SECUNDARIA,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, facecolor=FONDO)
    plt.close(fig)


def main() -> None:
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    tabla = codificar(cargar_transacciones())
    print(f"Transacciones: {tabla.shape[0]:,}   Ítems: {tabla.shape[1]}")
    print(f"Ítems: {', '.join(tabla.columns)}\n")

    comparacion = comparar(tabla)
    print(comparacion)
    print(
        "\nMismos itemsets y soportes en todos los valores: "
        f"{comparacion['mismos_itemsets'].all()}"
    )

    comparacion.write_csv(ARCHIVO_COMPARACION)
    graficar(comparacion, tabla.shape)
    for ruta in [ARCHIVO_COMPARACION, FIG_SOPORTE]:
        print(f"Guardado: {ruta}")


if __name__ == "__main__":
    main()
