"""Análisis de componentes principales (PCA).

- Ajuste con las variables seleccionadas por Pearson, ya estandarizadas (`_z`),
  usando solo el periodo de entrenamiento sin faltantes.
- Número de componentes según varianza acumulada >= `UMBRAL_VARIANZA`.
- Signo de cada componente orientado para que su loading de mayor magnitud sea positivo.
- Varianza explicada, loadings, correlaciones variable-componente y medias de PC1/PC2
  por grupo de calendario.

Guarda los parámetros del modelo en JSON y las figuras (scree plot, loadings y
proyección 2D) en PNG.
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from sklearn.decomposition import PCA

from config.rutas import (
    ARCHIVO_NORMALIZADO,
    FIG_LOADINGS,
    FIG_PROYECCION,
    FIG_SCREE,
    PARAMETROS_PCA,
)
from src.features.correlacion import ETIQUETAS, MAPA_COLOR, VARIABLES_MODELO
from src.features.normalizacion import SUFIJO

UMBRAL_VARIANZA = 0.80
COLUMNAS_Z = [f"{v}{SUFIJO}" for v in VARIABLES_MODELO]

TINTA = "#2b2b29"
TINTA_SECUNDARIA = "#6b6b68"
AZUL = "#2a78d6"
GRIS = "#c9c8c3"
MAPA_DENSIDAD = LinearSegmentedColormap.from_list(
    "azul_secuencial", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def cargar_entrenamiento() -> np.ndarray:
    """Matriz de entrenamiento (filas sin faltantes) con las columnas `_z`."""
    return (
        pl.read_parquet(ARCHIVO_NORMALIZADO)
        .filter(pl.col("particion") == "entrenamiento")
        .select(COLUMNAS_Z)
        .drop_nulls()
        .to_numpy()
    )


def ajustar_pca(x: np.ndarray) -> PCA:
    """PCA con todos los componentes y signos orientados."""
    modelo = PCA().fit(x)
    mayor = np.abs(modelo.components_).argmax(axis=1)
    signos = np.sign(modelo.components_[np.arange(len(mayor)), mayor])
    modelo.components_ = modelo.components_ * signos[:, None]
    return modelo


def elegir_k(modelo: PCA, umbral: float = UMBRAL_VARIANZA) -> int:
    """Menor número de componentes cuya varianza acumulada alcanza `umbral`."""
    acumulada = np.cumsum(modelo.explained_variance_ratio_)
    return int(np.searchsorted(acumulada, umbral) + 1)


def nombres_pc(n: int) -> list[str]:
    return [f"PC{i}" for i in range(1, n + 1)]


def tabla_varianza(modelo: PCA) -> pl.DataFrame:
    """Valor propio, % de varianza y % acumulado por componente."""
    ratio = modelo.explained_variance_ratio_
    return pl.DataFrame(
        {
            "componente": nombres_pc(len(ratio)),
            "valor_propio": modelo.explained_variance_.astype(float),
            "pct_varianza": 100 * ratio.astype(float),
            "pct_acumulado": 100 * np.cumsum(ratio.astype(float)),
        }
    ).with_columns(pl.selectors.float().round(3))


def tabla_loadings(modelo: PCA, k: int) -> pl.DataFrame:
    """Pesos de cada variable en los primeros `k` componentes."""
    datos = {"variable": VARIABLES_MODELO}
    for i, pc in enumerate(nombres_pc(k)):
        datos[pc] = modelo.components_[i].astype(float)
    return pl.DataFrame(datos).with_columns(pl.selectors.float().round(3))


def tabla_correlaciones(modelo: PCA, k: int) -> pl.DataFrame:
    """Correlación variable-componente: loading * sqrt(valor propio)."""
    datos = {"variable": VARIABLES_MODELO}
    for i, pc in enumerate(nombres_pc(k)):
        datos[pc] = (
            modelo.components_[i] * np.sqrt(modelo.explained_variance_[i])
        ).astype(float)
    return pl.DataFrame(datos).with_columns(pl.selectors.float().round(3))


def proyectar(
    df: pl.DataFrame, componentes: np.ndarray, media: np.ndarray
) -> pl.DataFrame:
    """Agrega PC1..PCk; las filas con faltantes quedan en null."""
    completas = pl.all_horizontal(pl.col(c).is_not_null() for c in COLUMNAS_Z)
    x = df.select(COLUMNAS_Z).fill_null(0).to_numpy()
    scores = (x - media) @ componentes.T
    return df.with_columns(
        pl.when(completas).then(pl.Series(pc, scores[:, i], dtype=pl.Float32)).alias(pc)
        for i, pc in enumerate(nombres_pc(componentes.shape[0]))
    )


def medias_por_grupo(df: pl.DataFrame, grupo: str) -> pl.DataFrame:
    return (
        df.drop_nulls("PC1")
        .group_by(grupo)
        .agg(
            pl.col("PC1").cast(pl.Float64).mean().alias("media_PC1"),
            pl.col("PC2").cast(pl.Float64).mean().alias("media_PC2"),
            pl.col("PC1").cast(pl.Float64).std().alias("desv_PC1"),
            pl.len().alias("filas"),
        )
        .sort(grupo)
        .with_columns(pl.selectors.float().round(3))
    )


def guardar_parametros(modelo: PCA, k: int, ruta=PARAMETROS_PCA) -> None:
    registro = {
        "variables": COLUMNAS_Z,
        "umbral_varianza": UMBRAL_VARIANZA,
        "n_componentes": k,
        "media": modelo.mean_.tolist(),
        "componentes": modelo.components_[:k].tolist(),
        "valores_propios": modelo.explained_variance_.tolist(),
        "varianza_explicada": modelo.explained_variance_ratio_.tolist(),
    }
    ruta.write_text(json.dumps(registro, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ figuras
def _estilo(ax) -> None:
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    for lado in ["left", "bottom"]:
        ax.spines[lado].set_color(GRIS)
    ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=9)


def graficar_scree(modelo: PCA, k: int, ruta=FIG_SCREE) -> None:
    """Barras de % de varianza por componente y línea de % acumulado."""
    pct = 100 * modelo.explained_variance_ratio_
    acumulado = np.cumsum(pct)
    x = np.arange(1, len(pct) + 1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    colores = [AZUL if i <= k else GRIS for i in x]
    ax.bar(x, pct, color=colores, width=0.6, zorder=2)
    ax.plot(x, acumulado, color=TINTA, linewidth=2, marker="o", markersize=6, zorder=3)
    ax.axhline(
        100 * UMBRAL_VARIANZA, color=TINTA_SECUNDARIA, linewidth=1, linestyle="--"
    )

    for xi, p in zip(x, pct):
        dentro = p > 8
        ax.text(
            xi,
            p - 2 if dentro else p + 1.5,
            f"{p:.1f}%",
            ha="center",
            va="top" if dentro else "bottom",
            fontsize=9,
            color="white" if dentro and xi <= k else TINTA,
        )
    ax.annotate(
        f"{acumulado[k - 1]:.1f}% acumulado",
        xy=(k, acumulado[k - 1]),
        xytext=(k - 1.6, acumulado[k - 1] + 7),
        fontsize=9,
        color=TINTA,
        arrowprops={"arrowstyle": "-", "color": TINTA_SECUNDARIA},
    )
    ax.text(
        0.6,
        100 * UMBRAL_VARIANZA + 1.5,
        f"Umbral {100 * UMBRAL_VARIANZA:.0f}%",
        fontsize=9,
        color=TINTA_SECUNDARIA,
    )

    ax.set_xticks(x, nombres_pc(len(pct)))
    ax.set_ylim(0, 105)
    ax.set_ylabel("% de varianza explicada", fontsize=9, color=TINTA_SECUNDARIA)
    ax.grid(axis="y", color="#ecebe7", linewidth=0.8, zorder=0)
    _estilo(ax)
    ax.set_title(
        "Varianza explicada por componente principal",
        fontsize=12,
        loc="left",
        pad=12,
        color=TINTA,
    )
    fig.text(
        0.01,
        0.01,
        "Barras: varianza por componente (azul = conservados). Línea: varianza acumulada.",
        fontsize=8,
        color=TINTA_SECUNDARIA,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ruta, facecolor="white")
    plt.close(fig)


def graficar_loadings(modelo: PCA, k: int, ruta=FIG_LOADINGS) -> None:
    """Mapa de calor de los loadings de los componentes conservados."""
    valores = modelo.components_[:k].T
    filas = [ETIQUETAS[v] for v in VARIABLES_MODELO]
    pcs = nombres_pc(k)
    pct = 100 * modelo.explained_variance_ratio_[:k]

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    imagen = ax.imshow(valores, cmap=MAPA_COLOR, vmin=-1, vmax=1, aspect="auto")
    for i in range(valores.shape[0]):
        for j in range(k):
            w = valores[i, j]
            ax.text(
                j,
                i,
                f"{w:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(w) > 0.55 else TINTA,
                fontweight="bold" if abs(w) >= 0.4 else "normal",
            )
    ax.set_xticks(range(k), [f"{pc}\n{p:.1f}%" for pc, p in zip(pcs, pct)], fontsize=9)
    ax.set_yticks(range(len(filas)), filas, fontsize=9)
    ax.set_xticks(np.arange(-0.5, k), minor=True)
    ax.set_yticks(np.arange(-0.5, len(filas)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for borde in ax.spines.values():
        borde.set_visible(False)
    barra = fig.colorbar(imagen, ax=ax, shrink=0.8, ticks=[-1, -0.5, 0, 0.5, 1])
    barra.outline.set_visible(False)
    barra.set_label("Loading", fontsize=9)
    ax.set_title(
        "Loadings de los componentes conservados",
        fontsize=11,
        loc="left",
        pad=12,
        color=TINTA,
    )
    fig.text(
        0.01,
        0.01,
        "En negritas: |loading| >= 0.4. El signo de cada componente es arbitrario.",
        fontsize=8,
        color=TINTA_SECUNDARIA,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ruta, facecolor="white")
    plt.close(fig)


def graficar_proyeccion(df: pl.DataFrame, modelo: PCA, ruta=FIG_PROYECCION) -> None:
    """Densidad de PC1 frente a PC2 en un panel por franja horaria."""
    puntos = df.drop_nulls("PC1").select("PC1", "PC2", "franja_horaria")
    pct = 100 * modelo.explained_variance_ratio_[:2]
    x1, x2 = np.quantile(puntos["PC1"], [0.001, 0.995])
    y1, y2 = np.quantile(puntos["PC2"], [0.002, 0.998])
    norma = LogNorm(vmin=1, vmax=8000)
    franjas = list(puntos.schema["franja_horaria"].categories)
    titulos = {
        "madrugada": "Madrugada (0-6 h)",
        "manana": "Mañana (6-12 h)",
        "tarde": "Tarde (12-18 h)",
        "noche": "Noche (18-24 h)",
    }

    fig, ejes = plt.subplots(2, 2, figsize=(10, 7.5), dpi=150, sharex=True, sharey=True)
    for ax, franja in zip(ejes.flat, franjas):
        sub = puntos.filter(pl.col("franja_horaria") == franja)
        densidad = ax.hexbin(
            sub["PC1"],
            sub["PC2"],
            gridsize=60,
            extent=(x1, x2, y1, y2),
            cmap=MAPA_DENSIDAD,
            norm=norma,
            mincnt=1,
            linewidths=0,
        )
        c1, c2 = sub["PC1"].mean(), sub["PC2"].mean()
        ax.scatter(
            c1, c2, s=60, color="#eb6834", edgecolor="white", linewidth=2, zorder=3
        )
        ax.axvline(0, color=GRIS, linewidth=0.8, zorder=0)
        ax.axhline(0, color=GRIS, linewidth=0.8, zorder=0)
        ax.set_title(
            f"{titulos[franja]} · PC1 medio {c1:+.2f}",
            fontsize=10,
            loc="left",
            color=TINTA,
        )
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        _estilo(ax)
    for ax in ejes[1]:
        ax.set_xlabel(
            f"PC1 · carga total ({pct[0]:.1f}%)", fontsize=9, color=TINTA_SECUNDARIA
        )
    for ax in ejes[:, 0]:
        ax.set_ylabel(
            f"PC2 · tipo de carga ({pct[1]:.1f}%)", fontsize=9, color=TINTA_SECUNDARIA
        )

    barra = fig.colorbar(densidad, ax=ejes, shrink=0.7)
    barra.outline.set_visible(False)
    barra.set_label("Minutos por celda (escala log)", fontsize=9)
    fig.suptitle(
        "Proyección en PC1 y PC2 por franja horaria",
        fontsize=12,
        x=0.01,
        ha="left",
        color=TINTA,
    )
    fig.text(
        0.01,
        0.01,
        "Todos los minutos sin faltantes (2006-2010). Punto naranja: centroide de la franja.",
        fontsize=8,
        color=TINTA_SECUNDARIA,
    )
    fig.savefig(ruta, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    x = cargar_entrenamiento()
    modelo = ajustar_pca(x)
    k = elegir_k(modelo)

    print(f"Filas de ajuste (entrenamiento, sin faltantes): {x.shape[0]:,}")
    print(f"Variables ({len(VARIABLES_MODELO)}): {', '.join(VARIABLES_MODELO)}")
    print(f"Varianza de cada columna de ajuste: {np.round(x.var(axis=0), 3)}\n")
    print(tabla_varianza(modelo))
    print(
        f"\nComponentes con valor propio > 1 (Kaiser): "
        f"{int((modelo.explained_variance_ > 1).sum())}"
    )
    print(f"Componentes conservados (acumulado >= {UMBRAL_VARIANZA:.0%}): {k}")
    print("\nLoadings:")
    print(tabla_loadings(modelo, k))
    print("\nCorrelación variable-componente:")
    print(tabla_correlaciones(modelo, k))

    df = proyectar(
        pl.read_parquet(ARCHIVO_NORMALIZADO), modelo.components_[:k], modelo.mean_
    )
    for grupo in ["franja_horaria", "es_fin_semana", "estacion", "particion"]:
        print(f"\nPC1 y PC2 por {grupo}:")
        print(medias_por_grupo(df, grupo))

    guardar_parametros(modelo, k)
    FIG_SCREE.parent.mkdir(parents=True, exist_ok=True)
    graficar_scree(modelo, k)
    graficar_loadings(modelo, k)
    graficar_proyeccion(df, modelo)
    for ruta in [PARAMETROS_PCA, FIG_SCREE, FIG_LOADINGS, FIG_PROYECCION]:
        print(f"Guardado: {ruta}")


if __name__ == "__main__":
    main()
