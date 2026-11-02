"""Transacciones horarias e ítems para reglas de asociación.

- Carga el dataset limpio de la Práctica 5 (marca de tiempo ya construida desde el
  archivo original).
- Una transacción es una hora: suma de los submedidores (Wh) y promedio de las
  métricas globales; solo se conservan las horas con los 60 minutos medidos.
- Ítems de habitación: un submedidor está activo si su energía en la hora supera
  el umbral de `UMBRALES_WH` (valle entre el consumo base y el uso real).
- Ítems temporales: tiempo del día (`FRANJAS`) y día de la semana.
- Ítem `Consumo alto`: potencia media de la hora mayor o igual al percentil
  `PERCENTIL_CONSUMO_ALTO`.

Imprime el resumen de los ítems y guarda la figura de los umbrales en PNG.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from config.rutas import ARCHIVO_LIMPIO, FIG_UMBRALES
from src.dashboard.tema import MOCHA
from src.features.correlacion import FONDO, LINEA, TINTA, TINTA_SECUNDARIA

MINUTOS_HORA = 60

# Ítem: (columna, umbral en Wh por hora)
UMBRALES_WH = {
    "Cocina": ("Sub_metering_1", 50),
    "Lavandería": ("Sub_metering_2", 100),
    "Climatización": ("Sub_metering_3", 200),
}

# Tiempo del día: (ítem, hora inicial, hora final), ambas incluidas
FRANJAS = [
    ("Madrugada", 0, 5),
    ("Mañana", 6, 11),
    ("Mediodía", 12, 13),
    ("Tarde", 14, 18),
    ("Noche", 19, 23),
]
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

PERCENTIL_CONSUMO_ALTO = 0.75
CONSUMO_ALTO = "Consumo alto"

GLOBALES = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
]
SUBMEDIDORES = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]


def cargar_limpio() -> pl.DataFrame:
    return pl.read_parquet(
        ARCHIVO_LIMPIO, columns=["datetime", *GLOBALES, *SUBMEDIDORES]
    )


def resumen_horario(df: pl.DataFrame) -> pl.DataFrame:
    """Una fila por hora: promedio de las métricas globales y suma de los submedidores."""
    return (
        df.group_by(pl.col("datetime").dt.truncate("1h"), maintain_order=True)
        .agg(
            *[pl.col(c).mean() for c in GLOBALES],
            *[pl.col(c).sum() for c in SUBMEDIDORES],
            pl.col("Global_active_power").is_not_null().sum().alias("minutos_medidos"),
        )
        .sort("datetime")
    )


def horas_completas(horario: pl.DataFrame) -> pl.DataFrame:
    return horario.filter(pl.col("minutos_medidos") == MINUTOS_HORA)


def tiempo_del_dia(hora: pl.Expr) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.String)
    for nombre, inicio, fin in reversed(FRANJAS):
        expr = (
            pl.when(hora.is_between(inicio, fin)).then(pl.lit(nombre)).otherwise(expr)
        )
    return expr.cast(pl.Enum([f[0] for f in FRANJAS]))


def agregar_items(horas: pl.DataFrame) -> pl.DataFrame:
    """Agrega `Habitaciones_Activas`, `Tiempo`, `Dia` y `Consumo_Alto`."""
    limite = horas["Global_active_power"].quantile(PERCENTIL_CONSUMO_ALTO)
    activas = [
        pl.when(pl.col(columna) > umbral).then(pl.lit(item))
        for item, (columna, umbral) in UMBRALES_WH.items()
    ]
    return horas.with_columns(
        pl.concat_list(activas).list.drop_nulls().alias("Habitaciones_Activas"),
        tiempo_del_dia(pl.col("datetime").dt.hour()).alias("Tiempo"),
        pl.col("datetime")
        .dt.weekday()
        .replace_strict(dict(enumerate(DIAS, 1)), return_dtype=pl.Enum(DIAS))
        .alias("Dia"),
        (pl.col("Global_active_power") >= limite).alias("Consumo_Alto"),
    )


def transacciones(items: pl.DataFrame) -> list[list[str]]:
    """Lista de ítems por hora: habitaciones activas, tiempo, día y consumo alto."""
    return [
        [*habitaciones, tiempo, dia, *([CONSUMO_ALTO] if alto else [])]
        for habitaciones, tiempo, dia, alto in items.select(
            "Habitaciones_Activas", "Tiempo", "Dia", "Consumo_Alto"
        ).iter_rows()
    ]


def soporte_por_item(lista: list[list[str]]) -> pl.DataFrame:
    """Proporción de transacciones que contienen cada ítem."""
    return (
        pl.DataFrame({"item": [i for t in lista for i in t]})
        .group_by("item")
        .len()
        .with_columns((pl.col("len") / len(lista)).round(4).alias("soporte"))
        .rename({"len": "transacciones"})
        .sort("soporte", descending=True)
    )


def graficar_umbrales(horas: pl.DataFrame, ruta=FIG_UMBRALES) -> None:
    """Histograma de Wh por hora de cada submedidor (escala log) con su umbral."""
    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=150, facecolor=FONDO)
    bordes = np.logspace(0, np.log10(5000), 45)
    for ax, (item, (columna, umbral)) in zip(ejes, UMBRALES_WH.items()):
        wh = horas[columna].to_numpy()
        ceros = (wh == 0).mean()
        activas = (wh > umbral).mean()
        ax.hist(
            wh[wh > 0],
            bins=bordes,
            color=MOCHA["blue"],
            weights=np.full((wh > 0).sum(), 100 / len(wh)),
            zorder=2,
        )
        ax.axvline(umbral, color=MOCHA["peach"], linewidth=2, linestyle="--", zorder=3)
        ax.set_xscale("log")
        ax.set_facecolor(FONDO)
        ax.grid(axis="y", color=MOCHA["surface0"], linewidth=0.8, zorder=0)
        for lado in ["top", "right"]:
            ax.spines[lado].set_visible(False)
        for lado in ["left", "bottom"]:
            ax.spines[lado].set_color(LINEA)
        ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=9)
        ax.set_title(f"{item} ({columna})", fontsize=11, loc="left", color=TINTA)
        ax.set_xlabel("Wh en la hora (escala log)", fontsize=9, color=TINTA_SECUNDARIA)
        # Margen superior para el texto, en la esquina izquierda (consumo bajo, sin barras altas)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        ax.text(
            0.03,
            0.96,
            f"umbral > {umbral} Wh\nactiva: {activas:.1%} de las horas\n"
            f"en 0 Wh: {ceros:.1%}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=TINTA,
        )
    ejes[0].set_ylabel("% de horas", fontsize=9, color=TINTA_SECUNDARIA)
    fig.suptitle(
        "Energía por hora de cada submedidor y umbral de actividad",
        fontsize=12,
        x=0.01,
        ha="left",
        color=TINTA,
    )
    fig.text(
        0.01,
        0.01,
        "Horas con los 60 minutos medidos. Las horas en 0 Wh no aparecen en el eje log; "
        "su porcentaje se indica en cada panel.",
        fontsize=8,
        color=TINTA_SECUNDARIA,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, facecolor=FONDO)
    plt.close(fig)


def main() -> None:
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    horario = resumen_horario(cargar_limpio())
    horas = horas_completas(horario)
    items = agregar_items(horas)
    lista = transacciones(items)

    sin_medicion = horario.filter(pl.col("minutos_medidos") == 0).height
    print(f"Horas en el periodo: {horario.height:,}")
    print(f"  sin medición: {sin_medicion:,}")
    print(f"  parciales: {horario.height - horas.height - sin_medicion:,}")
    print(f"Transacciones (horas con {MINUTOS_HORA} minutos medidos): {len(lista):,}")
    limite = horas["Global_active_power"].quantile(PERCENTIL_CONSUMO_ALTO)
    print(
        f"Consumo alto: potencia media ≥ {limite:.3f} kW "
        f"(percentil {PERCENTIL_CONSUMO_ALTO:.0%})"
    )

    print("\nSoporte de cada ítem:")
    print(soporte_por_item(lista))

    tamanos = items["Habitaciones_Activas"].list.len().alias("habitaciones")
    print("\nHabitaciones activas por hora:")
    print(tamanos.value_counts().sort(tamanos.name))
    print("\nEjemplos de transacciones:")
    for i in [0, 8, 20, 4000]:
        print(f"  {items['datetime'][i]:%Y-%m-%d %H:%M}  {lista[i]}")

    graficar_umbrales(items)
    print(f"\nGuardado: {FIG_UMBRALES}")


if __name__ == "__main__":
    main()
