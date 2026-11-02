"""Gráficas de las reglas de asociación (paleta Catppuccin Mocha).

- Dispersión de soporte contra confianza de todas las reglas, coloreada por lift.
- Reglas con mayor lift, separadas en reglas de uso y de rutina.
- Lift de {tiempo del día, día} ⇒ {Consumo alto} para las 35 combinaciones.

Guarda las figuras en PNG.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from config.rutas import (
    ARCHIVO_INPUT_REGLAS,
    ARCHIVO_REGLAS,
    FIG_DISPERSION,
    FIG_RUTINA,
    FIG_TOP,
)
from src.dashboard.tema import MOCHA
from src.features.correlacion import (
    FONDO,
    LINEA,
    MAPA_COLOR,
    TINTA,
    TINTA_SECUNDARIA,
    barra_color,
    color_texto,
)
from src.reglas.reglas import MIN_CONFIDENCE, MIN_SUPPORT
from src.reglas.transacciones import DIAS, FRANJAS

TOP = 8
MAPA_LIFT = LinearSegmentedColormap.from_list(
    "lift_mocha", [MOCHA["blue"], MOCHA["mauve"], MOCHA["red"]]
)


def _estilo(ax) -> None:
    ax.set_facecolor(FONDO)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    for lado in ["left", "bottom"]:
        ax.spines[lado].set_color(LINEA)
    ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=9)


def _guardar(fig, ruta, nota: str) -> None:
    fig.text(0.01, 0.01, nota, fontsize=8, color=TINTA_SECUNDARIA)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, facecolor=FONDO)
    plt.close(fig)


def graficar_dispersion(reglas: pl.DataFrame, ruta=FIG_DISPERSION) -> None:
    """Soporte (log) contra confianza; color = lift, forma = tipo de regla."""
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=150, facecolor=FONDO)
    norma = plt.Normalize(1, reglas["lift"].max())
    for tipo, marcador, tamano in [("uso", "o", 46), ("rutina", "D", 52)]:
        sub = reglas.filter(pl.col("tipo") == tipo)
        puntos = ax.scatter(
            sub["soporte"],
            sub["confianza"],
            c=sub["lift"],
            cmap=MAPA_LIFT,
            norm=norma,
            marker=marcador,
            s=tamano,
            edgecolors=FONDO,
            linewidths=0.6,
            label=f"{tipo} ({sub.height})",
            zorder=3,
        )
    ax.set_xscale("log")
    ax.axhline(MIN_CONFIDENCE, color=MOCHA["peach"], linewidth=1.2, linestyle="--")
    ax.axvline(MIN_SUPPORT, color=MOCHA["peach"], linewidth=1.2, linestyle="--")
    ax.grid(color=MOCHA["surface0"], linewidth=0.8, zorder=0)
    _estilo(ax)
    ax.set_xlabel("soporte (escala log)", fontsize=9, color=TINTA_SECUNDARIA)
    ax.set_ylabel("confianza", fontsize=9, color=TINTA_SECUNDARIA)
    leyenda = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for texto, color in zip(leyenda.get_texts(), [TINTA, TINTA]):
        texto.set_color(color)
    for marca in leyenda.legend_handles:
        marca.set_facecolor(MOCHA["overlay1"])
    barra_color(fig, puntos, ax, "lift")
    ax.set_title(
        "Soporte y confianza de las reglas", fontsize=12, loc="left", color=TINTA
    )
    _guardar(
        fig,
        ruta,
        f"{reglas.height} reglas con lift > 1. Líneas punteadas: min_support = "
        f"{MIN_SUPPORT} y min_confidence = {MIN_CONFIDENCE}. "
        "Círculo: regla de uso; rombo: regla de rutina.",
    )


def graficar_top(reglas: pl.DataFrame, ruta=FIG_TOP) -> None:
    """Reglas con mayor lift de cada tipo, con su confianza y horas."""
    fig, ejes = plt.subplots(2, 1, figsize=(11, 9), dpi=150, facecolor=FONDO)
    for ax, tipo, color in [
        (ejes[0], "uso", MOCHA["mauve"]),
        (ejes[1], "rutina", MOCHA["sapphire"]),
    ]:
        sub = reglas.filter(pl.col("tipo") == tipo).head(TOP).reverse()
        etiquetas = [
            f"{{{a}}} ⇒ {{{c}}}"
            for a, c in sub.select("antecedente", "consecuente").rows()
        ]
        y = np.arange(sub.height)
        ax.barh(y, sub["lift"], color=color, height=0.65, zorder=2)
        for yi, (lift, conf, horas) in zip(
            y, sub.select("lift", "confianza", "horas").rows()
        ):
            ax.text(
                lift + 0.04,
                yi,
                f"{lift:.2f}  ·  conf {conf:.2f}  ·  {horas:,} h",
                va="center",
                fontsize=8.5,
                color=TINTA,
            )
        ax.axvline(1, color=MOCHA["overlay1"], linewidth=1.5, linestyle="--", zorder=3)
        ax.set_yticks(y, etiquetas, fontsize=8.5, color=TINTA)
        ax.set_xlim(0, sub["lift"].max() * 1.5)
        ax.grid(axis="x", color=MOCHA["surface0"], linewidth=0.8, zorder=0)
        _estilo(ax)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("lift", fontsize=9, color=TINTA_SECUNDARIA)
        ax.set_title(
            f"Reglas de {tipo} con mayor lift", fontsize=11, loc="left", color=TINTA
        )
    fig.suptitle(
        "Reglas de asociación ordenadas por lift",
        fontsize=12,
        x=0.01,
        ha="left",
        color=TINTA,
    )
    _guardar(
        fig,
        ruta,
        "Uso: el antecedente incluye habitaciones o consumo. Rutina: el antecedente solo "
        "tiene franja y día. Línea punteada: lift = 1 (independencia).",
    )


def lift_rutina(entrada: pl.DataFrame) -> pl.DataFrame:
    """Confianza y lift de {tiempo, día} ⇒ {Consumo alto} en cada combinación."""
    base = entrada["Consumo_Alto"].mean()
    return (
        entrada.group_by("Tiempo", "Dia")
        .agg(pl.col("Consumo_Alto").mean().alias("confianza"), pl.len().alias("horas"))
        .with_columns((pl.col("confianza") / base).alias("lift"))
        .sort("Tiempo", "Dia")
    )


def graficar_rutina(entrada: pl.DataFrame, ruta=FIG_RUTINA) -> None:
    """Mapa de calor del lift de consumo alto por tiempo del día y día de la semana."""
    tabla = lift_rutina(entrada)
    franjas = [f[0] for f in FRANJAS]
    valores = np.array(
        [
            [
                tabla.filter((pl.col("Tiempo") == f) & (pl.col("Dia") == d))[
                    "lift"
                ].item()
                for d in DIAS
            ]
            for f in franjas
        ]
    )
    norma = TwoSlopeNorm(vmin=min(valores.min(), 0.5), vcenter=1, vmax=valores.max())
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150, facecolor=FONDO)
    imagen = ax.imshow(valores, cmap=MAPA_COLOR, norm=norma, aspect="auto")
    for i in range(len(franjas)):
        for j in range(len(DIAS)):
            ax.text(
                j,
                i,
                f"{valores[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=color_texto(MAPA_COLOR(norma(valores[i, j]))),
            )
    rangos = [f"{n} ({a}–{b} h)" for n, a, b in FRANJAS]
    ax.set_xticks(range(len(DIAS)), DIAS, fontsize=9, color=TINTA)
    ax.set_yticks(range(len(franjas)), rangos, fontsize=9, color=TINTA)
    ax.set_xticks(np.arange(-0.5, len(DIAS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(franjas)), minor=True)
    ax.grid(which="minor", color=FONDO, linewidth=2)
    ax.tick_params(which="both", length=0)
    for borde in ax.spines.values():
        borde.set_visible(False)
    barra_color(fig, imagen, ax, "lift")
    ax.set_title(
        "Lift de {tiempo del día, día} ⇒ {Consumo alto}",
        fontsize=12,
        loc="left",
        color=TINTA,
    )
    _guardar(
        fig,
        ruta,
        f"Consumo alto en {entrada['Consumo_Alto'].mean():.0%} de todas las horas. "
        "Lift > 1 (rojo): la combinación tiene consumo alto más seguido que el promedio.",
    )


def main() -> None:
    reglas = pl.read_csv(ARCHIVO_REGLAS)
    entrada = pl.read_parquet(ARCHIVO_INPUT_REGLAS)

    graficar_dispersion(reglas)
    graficar_top(reglas)
    graficar_rutina(entrada)

    rutina = lift_rutina(entrada).sort("lift", descending=True)
    print("Combinaciones con mayor y menor lift de consumo alto:")
    print(pl.concat([rutina.head(5), rutina.tail(5)]))
    for ruta in [FIG_DISPERSION, FIG_TOP, FIG_RUTINA]:
        print(f"Guardado: {ruta}")


if __name__ == "__main__":
    main()
