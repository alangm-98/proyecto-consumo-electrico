"""Inspección inicial del dataset crudo.

Imprime dimensiones, tipos inferidos, valores no numéricos, formato de fechas,
continuidad temporal, duplicados, patrón de faltantes, estadísticos descriptivos
y chequeos de coherencia física. No escribe archivos.
"""

import polars as pl

from src.cleaning.carga import (
    COLUMNAS_NUMERICAS,
    MARCA_FALTANTE,
    cargar_crudo_inferido,
    cargar_crudo_texto,
)


def titulo(texto: str) -> None:
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


# ------------------------------------------------------------------ tipos
def tipos_detectados(crudo: pl.DataFrame) -> pl.DataFrame:
    inferido = cargar_crudo_inferido()
    return pl.DataFrame(
        {
            "columna": crudo.columns,
            "tipo_inferido": [str(t) for t in inferido.dtypes],
            "ejemplo": [str(crudo[c][0]) for c in crudo.columns],
        }
    )


def valores_no_numericos(crudo: pl.DataFrame) -> pl.DataFrame:
    """Conteo y valores distintos que no se pueden convertir a número, por columna."""
    filas = []
    for c in COLUMNAS_NUMERICAS:
        s = crudo[c]
        no_num = s.filter(s.cast(pl.Float64, strict=False).is_null())
        filas.append(
            {
                "columna": c,
                "no_numericos": no_num.len(),
                "pct": round(100 * no_num.len() / s.len(), 3),
                "valores": ", ".join(
                    "(campo vacío)" if v is None else repr(v)
                    for v in no_num.unique().to_list()
                ),
                "con_espacios": s.filter(s != s.str.strip_chars()).len(),
            }
        )
    return pl.DataFrame(filas)


def formato_fechas(crudo: pl.DataFrame) -> pl.DataFrame:
    """Conteo de filas según los dígitos de día y mes en `Date`."""
    partes = pl.col("Date").str.split("/")
    return (
        crudo.select(
            partes.list.get(0).str.len_chars().alias("digitos_dia"),
            partes.list.get(1).str.len_chars().alias("digitos_mes"),
        )
        .group_by("digitos_dia", "digitos_mes")
        .len()
        .sort("digitos_dia", "digitos_mes")
    )


def a_numerico(crudo: pl.DataFrame) -> pl.DataFrame:
    """Conversión provisional a `datetime` y `Float64` para la inspección."""
    return crudo.with_columns(
        pl.concat_str(["Date", "Time"], separator=" ")
        .str.to_datetime("%d/%m/%Y %H:%M:%S", strict=True)
        .alias("datetime"),
        *[
            pl.col(c).replace(MARCA_FALTANTE, None).cast(pl.Float64).alias(c)
            for c in COLUMNAS_NUMERICAS
        ],
    )


# ------------------------------------------------------------------ faltantes
def nulos_por_fila(df: pl.DataFrame) -> pl.DataFrame:
    n = pl.sum_horizontal([pl.col(c).is_null() for c in COLUMNAS_NUMERICAS])
    return (
        df.select(n.alias("mediciones_nulas_en_la_fila"))
        .group_by("mediciones_nulas_en_la_fila")
        .len()
        .sort("mediciones_nulas_en_la_fila")
    )


def bloques_faltantes(df: pl.DataFrame) -> pl.DataFrame:
    """Rachas de minutos consecutivos sin medición: inicio, fin y duración."""
    falta = pl.col("Global_active_power").is_null()
    return (
        df.select("datetime", falta.alias("falta"))
        .with_columns(pl.col("falta").rle_id().alias("bloque"))
        .filter(pl.col("falta"))
        .group_by("bloque")
        .agg(
            pl.col("datetime").min().alias("inicio"),
            pl.col("datetime").max().alias("fin"),
            pl.len().alias("minutos"),
        )
        .drop("bloque")
        .with_columns((pl.col("minutos") / 60).round(1).alias("horas"))
        .sort("minutos", descending=True)
    )


def resumen_bloques(bloques: pl.DataFrame) -> pl.DataFrame:
    m = pl.col("minutos")
    duracion = (
        pl.when(m == 1)
        .then(pl.lit("a) 1 min"))
        .when(m <= 5)
        .then(pl.lit("b) 2-5 min"))
        .when(m <= 60)
        .then(pl.lit("c) 6-60 min"))
        .when(m <= 24 * 60)
        .then(pl.lit("d) 1-24 h"))
        .otherwise(pl.lit("e) > 24 h"))
    )
    return (
        bloques.group_by(duracion.alias("duracion"))
        .agg(pl.len().alias("bloques"), m.sum().alias("minutos"))
        .with_columns(
            (100 * pl.col("minutos") / pl.col("minutos").sum())
            .round(1)
            .alias("pct_minutos")
        )
        .sort("duracion")
    )


def faltantes_por_mes(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(pl.col("datetime").dt.strftime("%Y-%m").alias("mes"))
        .agg(
            pl.col("Global_active_power").is_null().sum().alias("min_faltantes"),
            pl.len().alias("min_totales"),
        )
        .with_columns(
            (100 * pl.col("min_faltantes") / pl.col("min_totales"))
            .round(2)
            .alias("pct")
        )
        .sort("mes")
    )


# ------------------------------------------------------------------ tiempo y duplicados
def revision_temporal(df: pl.DataFrame) -> dict:
    t = df["datetime"]
    inicio, fin = t.min(), t.max()
    esperados = int((fin - inicio).total_seconds() // 60) + 1
    validas = df.drop_nulls(COLUMNAS_NUMERICAS)
    return {
        "Inicio": inicio,
        "Fin": fin,
        "Minutos esperados en el rango": f"{esperados:,}",
        "Registros": f"{df.height:,}",
        "Minutos sin fila (huecos en la marca de tiempo)": f"{esperados - t.n_unique():,}",
        "Ordenado cronológicamente": t.is_sorted(),
        "Marcas de tiempo duplicadas": f"{df.height - t.n_unique():,}",
        "Filas completas duplicadas": f"{df.height - df.unique().height:,}",
        "Minutos cuyas 7 mediciones repiten las de otro minuto": f"{validas.height - validas.select(COLUMNAS_NUMERICAS).unique().height:,}",
    }


def saltos_entre_registros(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select(pl.col("datetime").diff().dt.total_minutes().alias("salto_minutos"))
        .drop_nulls()
        .group_by("salto_minutos")
        .len()
        .sort("len", descending=True)
    )


# ------------------------------------------------------------------ estadísticos
def descriptivos(df: pl.DataFrame) -> pl.DataFrame:
    filas = []
    for c in COLUMNAS_NUMERICAS:
        s = df[c].drop_nulls()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        filas.append(
            {
                "variable": c,
                "n": s.len(),
                "media": s.mean(),
                "desv_est": s.std(),
                "min": s.min(),
                "q1": q1,
                "mediana": s.median(),
                "q3": q3,
                "max": s.max(),
                "asimetria": s.skew(),
                "pct_ceros": 100 * (s == 0).sum() / s.len(),
                "pct_atipicos_iqr": 100
                * ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
                / s.len(),
            }
        )
    return pl.DataFrame(filas).with_columns(pl.selectors.float().round(3))


def coherencia_fisica(df: pl.DataFrame) -> pl.DataFrame:
    """Relaciones físicas entre variables.

    - No submedido (Wh) = Global_active_power * 1000 / 60 - Sub_metering_1..3.
    - Intensidad calculada (A) = sqrt(P² + Q²) * 1000 / V, comparada con Global_intensity.
    """
    d = df.drop_nulls(COLUMNAS_NUMERICAS).with_columns(
        (
            pl.col("Global_active_power") * 1000 / 60
            - pl.col("Sub_metering_1")
            - pl.col("Sub_metering_2")
            - pl.col("Sub_metering_3")
        ).alias("no_submedido_wh"),
        (
            (
                pl.col("Global_active_power") ** 2
                + pl.col("Global_reactive_power") ** 2
            ).sqrt()
            * 1000
            / pl.col("Voltage")
        ).alias("intensidad_calculada"),
    )
    return d.select(
        (pl.col("no_submedido_wh") < 0).sum().alias("no_submedido_negativo"),
        pl.col("no_submedido_wh").min().round(3).alias("no_submedido_min_wh"),
        pl.col("no_submedido_wh").mean().round(3).alias("no_submedido_media_wh"),
        (pl.col("intensidad_calculada") - pl.col("Global_intensity"))
        .abs()
        .mean()
        .round(3)
        .alias("error_medio_intensidad_A"),
        pl.corr("intensidad_calculada", "Global_intensity")
        .round(4)
        .alias("r_intensidad"),
    )


# ------------------------------------------------------------------ main
def main() -> None:
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_fmt_str_lengths(60)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    crudo = cargar_crudo_texto()

    titulo("1. Dimensiones")
    print(f"Filas: {crudo.height:,}   Columnas: {crudo.width}")
    print(f"Memoria (todo como texto): {crudo.estimated_size('mb'):.1f} MB")
    print(crudo.head(3))

    titulo("2. Tipos que infiere polars sin indicarle nada")
    print(tipos_detectados(crudo))

    titulo("3. Valores no numéricos en las columnas de medición")
    print(valores_no_numericos(crudo))

    titulo("4. Formato de la columna Date")
    print(formato_fechas(crudo))

    df = a_numerico(crudo)

    titulo("5. Revisión temporal y duplicados")
    for k, v in revision_temporal(df).items():
        print(f"{k:<55} {v}")
    print("\nSaltos entre registros consecutivos:")
    print(saltos_entre_registros(df))

    titulo("6. Patrón de faltantes")
    print(nulos_por_fila(df))
    bloques = bloques_faltantes(df)
    print(f"\nBloques de minutos consecutivos sin medición: {bloques.height}")
    print(resumen_bloques(bloques))
    print("\n10 bloques más largos:")
    print(bloques.head(10))
    print("\nMeses con más de 1 % de faltantes:")
    print(faltantes_por_mes(df).filter(pl.col("pct") > 1))

    titulo("7. Estadísticos descriptivos (sin faltantes)")
    print(descriptivos(df))

    titulo("8. Coherencia física")
    print(coherencia_fisica(df))


if __name__ == "__main__":
    main()
