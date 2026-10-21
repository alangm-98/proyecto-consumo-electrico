"""Correlación de Pearson y selección de variables.

- Matriz de correlación de las variables normalizadas, calculada solo con el periodo
  de entrenamiento y sin filas faltantes.
- Pares con |r| > `UMBRAL` marcados como redundantes.
- Verificación de la combinación lineal exacta que define `no_submedido_wh`.
- Mapa de calor guardado en PNG.

`VARIABLES_MODELO` es el subconjunto que se conserva para el PCA.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap

from config.rutas import ARCHIVO_NORMALIZADO, FIG_PEARSON
from src.features.normalizacion import SUFIJO, VARIABLES

UMBRAL = 0.8

DESCARTADAS = {
    "Global_intensity": "|r| = 0.999 con Global_active_power (I ≈ P/V)",
    "no_submedido_wh": "combinación lineal exacta de Global_active_power y Sub_metering_1..3",
}
VARIABLES_MODELO = [v for v in VARIABLES if v not in DESCARTADAS]

ETIQUETAS = {
    "Global_active_power": "Potencia activa",
    "Global_reactive_power": "Potencia reactiva",
    "Voltage": "Voltaje",
    "Global_intensity": "Intensidad",
    "Sub_metering_1": "Cocina (S1)",
    "Sub_metering_2": "Lavandería (S2)",
    "Sub_metering_3": "Calentador/AC (S3)",
    "no_submedido_wh": "No submedido",
}

# Divergente azul (negativo) - gris neutro (0) - rojo (positivo)
MAPA_COLOR = LinearSegmentedColormap.from_list(
    "azul_gris_rojo", ["#2a78d6", "#f0efec", "#e34948"]
)


def cargar_entrenamiento() -> pl.DataFrame:
    """Filas de entrenamiento sin faltantes, con las columnas `_z`."""
    columnas = [f"{v}{SUFIJO}" for v in VARIABLES]
    return (
        pl.read_parquet(ARCHIVO_NORMALIZADO)
        .filter(pl.col("particion") == "entrenamiento")
        .select(columnas)
        .drop_nulls()
        .rename({f"{v}{SUFIJO}": v for v in VARIABLES})
    )


def matriz_correlacion(df: pl.DataFrame) -> pl.DataFrame:
    """Matriz de Pearson con una columna `variable` como índice."""
    return df.corr().insert_column(0, pl.Series("variable", df.columns))


def pares_correlacionados(matriz: pl.DataFrame, umbral: float = UMBRAL) -> pl.DataFrame:
    """Pares distintos ordenados por |r|, con indicador de redundancia."""
    variables = matriz["variable"].to_list()
    filas = [
        {"variable_a": a, "variable_b": b, "r": matriz[b][i]}
        for i, a in enumerate(variables)
        for b in variables[i + 1 :]
    ]
    return (
        pl.DataFrame(filas)
        .with_columns(
            pl.col("r").round(3),
            (pl.col("r").abs() > umbral).alias("redundante"),
        )
        .sort(pl.col("r").abs(), descending=True)
    )


def correlacion_media(matriz: pl.DataFrame) -> pl.DataFrame:
    """Promedio de |r| de cada variable con las demás."""
    valores = np.abs(matriz.drop("variable").to_numpy())
    media = (valores.sum(axis=1) - 1) / (len(valores) - 1)
    return pl.DataFrame(
        {"variable": matriz["variable"], "media_abs_r": np.round(media, 3)}
    ).sort("media_abs_r", descending=True)


def verificar_combinacion_lineal() -> dict:
    """Compara la potencia activa en Wh con la suma de submedidores y no submedido."""
    df = (
        pl.read_parquet(ARCHIVO_NORMALIZADO)
        .filter(pl.col("particion") == "entrenamiento")
        .drop_nulls(VARIABLES)
    )
    suma = (
        pl.col("Sub_metering_1")
        + pl.col("Sub_metering_2")
        + pl.col("Sub_metering_3")
        + pl.col("no_submedido_wh")
    )
    energia = pl.col("Global_active_power") * 1000 / 60
    return df.select(
        pl.corr(energia, suma).alias("r"),
        (energia - suma).abs().mean().alias("error_medio_wh"),
    ).row(0, named=True)


def graficar_heatmap(matriz: pl.DataFrame, ruta=FIG_PEARSON) -> None:
    """Mapa de calor anotado de la matriz de correlación."""
    variables = matriz["variable"].to_list()
    valores = matriz.drop("variable").to_numpy()
    etiquetas = [ETIQUETAS[v] for v in variables]
    n = len(variables)

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)
    # Diagonal (r = 1 por definición) en gris para no competir con los pares
    sin_diagonal = valores.copy()
    np.fill_diagonal(sin_diagonal, np.nan)
    mapa = MAPA_COLOR.with_extremes(bad="#e6e5e1")
    imagen = ax.imshow(sin_diagonal, cmap=mapa, vmin=-1, vmax=1)

    for i in range(n):
        for j in range(n):
            r = valores[i, j]
            redundante = i != j and abs(r) > UMBRAL
            ax.text(
                j,
                i,
                "" if i == j else f"{r:.3f}" if redundante else f"{r:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(r) > 0.85 else "#2b2b29",
                fontweight="bold" if redundante else "normal",
            )

    ax.set_xticks(range(n), etiquetas, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n), etiquetas, fontsize=9)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for borde in ax.spines.values():
        borde.set_visible(False)

    barra = fig.colorbar(imagen, ax=ax, shrink=0.8, ticks=[-1, -0.5, 0, 0.5, 1])
    barra.outline.set_visible(False)
    barra.set_label("r de Pearson", fontsize=9)

    ax.set_title(
        "Solo la intensidad es redundante con la potencia activa (r = 0.999)",
        fontsize=12,
        loc="left",
        pad=14,
    )
    fig.text(
        0.01,
        0.01,
        f"Pearson sobre datos de entrenamiento (2006-2009). Umbral de redundancia: |r| > {UMBRAL}.",
        fontsize=8,
        color="#6b6b68",
    )
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, facecolor="white")
    plt.close(fig)


def main() -> None:
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    df = cargar_entrenamiento()
    matriz = matriz_correlacion(df)

    print(f"Filas usadas (entrenamiento, sin faltantes): {df.height:,}\n")
    print(matriz.with_columns(pl.selectors.float().round(3)))
    print(f"\nPares ordenados por |r| (umbral {UMBRAL}):")
    print(pares_correlacionados(matriz))
    print("\nCorrelación media de cada variable con las demás:")
    print(correlacion_media(matriz))

    lineal = verificar_combinacion_lineal()
    print(
        "\nPotencia activa (Wh) frente a S1 + S2 + S3 + no submedido: "
        f"r = {lineal['r']:.6f}, error medio = {lineal['error_medio_wh']:.4f} Wh"
    )

    print("\nDescartadas:")
    for v, motivo in DESCARTADAS.items():
        print(f"  {v:<22} {motivo}")
    print(
        f"Conservadas para el PCA ({len(VARIABLES_MODELO)}): {', '.join(VARIABLES_MODELO)}"
    )

    graficar_heatmap(matriz)
    print(f"\nGuardado: {FIG_PEARSON}")


if __name__ == "__main__":
    main()
