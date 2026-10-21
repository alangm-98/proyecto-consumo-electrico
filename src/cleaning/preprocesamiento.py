"""Preprocesamiento del dataset crudo.

1. Unificación de faltantes: espacios recortados; '?' y campo vacío a null.
2. Casteo: Date + Time a `datetime`; mediciones a Float32.
3. Deduplicación por marca de tiempo.
4. Interpolación lineal en bloques de faltantes de hasta 60 minutos; los bloques más
   largos permanecen null. `estado_dato` registra el origen de cada fila.
5. Variables derivadas: consumo no submedido y categóricas de calendario con
   etiquetas normalizadas (minúsculas, sin acentos ni espacios extra).

Guarda el resultado en Parquet e imprime un resumen.
"""

import polars as pl

from config.rutas import ARCHIVO_LIMPIO
from src.cleaning.carga import COLUMNAS_NUMERICAS, MARCA_FALTANTE, cargar_crudo_texto

MAX_MINUTOS_INTERPOLAR = 60

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
]
# Estaciones meteorológicas del hemisferio norte
ESTACIONES = ["Invierno", "Primavera", "Verano", "Otoño"]
ESTACION_POR_MES = {m: ESTACIONES[(m % 12) // 3] for m in range(1, 13)}
# Franjas de 6 horas: 0-6, 6-12, 12-18, 18-24
FRANJAS = ["Madrugada", "Mañana", "Tarde", "Noche"]

ESTADOS = ["original", "interpolado", "faltante"]


def normalizar_texto(expr: pl.Expr) -> pl.Expr:
    """Minúsculas, sin acentos y con espacios simples."""
    return (
        expr.str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .str.to_lowercase()
        .str.normalize("NFD")
        .str.replace_all(r"\p{Mn}", "")
    )


def normalizar_lista(valores: list[str]) -> list[str]:
    """Aplica `normalizar_texto` a una lista y conserva el orden."""
    serie = pl.select(normalizar_texto(pl.lit(pl.Series(valores)))).to_series()
    return list(dict.fromkeys(serie.to_list()))


def unificar_faltantes(df: pl.DataFrame) -> pl.DataFrame:
    """Recorta espacios en todos los campos y convierte '?' y '' en null."""
    return df.with_columns(
        pl.when(pl.col(c).str.strip_chars().is_in([MARCA_FALTANTE, ""]))
        .then(None)
        .otherwise(pl.col(c).str.strip_chars())
        .alias(c)
        for c in df.columns
    )


def castear(df: pl.DataFrame) -> pl.DataFrame:
    """Date + Time a `datetime`; mediciones a Float32 (casteo estricto)."""
    return df.select(
        pl.concat_str(["Date", "Time"], separator=" ")
        .str.to_datetime("%d/%m/%Y %H:%M:%S", strict=True)
        .alias("datetime"),
        *[pl.col(c).cast(pl.Float32, strict=True) for c in COLUMNAS_NUMERICAS],
    )


def deduplicar(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Conserva el primer registro de cada minuto y ordena cronológicamente."""
    unico = df.unique(subset="datetime", keep="first", maintain_order=True)
    return unico.sort("datetime"), df.height - unico.height


def imputar(
    df: pl.DataFrame, max_minutos: int = MAX_MINUTOS_INTERPOLAR
) -> pl.DataFrame:
    """Interpola linealmente los bloques de faltantes de hasta `max_minutos`.

    Los faltantes del crudo son filas completas, por lo que el bloque se define con
    `Global_active_power` y se aplica a las siete mediciones.
    """
    falta = pl.col("Global_active_power").is_null()
    df = df.with_columns(falta.alias("_falta"), falta.rle_id().alias("_bloque"))
    df = df.with_columns(pl.len().over("_bloque").alias("_largo"))
    corto = pl.col("_falta") & (pl.col("_largo") <= max_minutos)
    return df.with_columns(
        *[
            pl.when(corto).then(pl.col(c).interpolate()).otherwise(pl.col(c)).alias(c)
            for c in COLUMNAS_NUMERICAS
        ],
        pl.when(~pl.col("_falta"))
        .then(pl.lit("original"))
        .when(corto)
        .then(pl.lit("interpolado"))
        .otherwise(pl.lit("faltante"))
        .cast(pl.Enum(ESTADOS))
        .alias("estado_dato"),
    ).drop("_falta", "_bloque", "_largo")


def no_submedido() -> pl.Expr:
    """Energía activa del minuto (Wh) menos los tres submedidores."""
    return (
        pl.col("Global_active_power") * 1000 / 60
        - pl.col("Sub_metering_1")
        - pl.col("Sub_metering_2")
        - pl.col("Sub_metering_3")
    )


def categorica(expr: pl.Expr, mapa: dict, etiquetas: list[str]) -> pl.Expr:
    """Mapea enteros a etiquetas, las normaliza y las convierte en Enum ordenado."""
    texto = expr.replace_strict(mapa, return_dtype=pl.String)
    return normalizar_texto(texto).cast(pl.Enum(normalizar_lista(etiquetas)))


def derivar(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega `no_submedido_wh` (recortado a 0) y las variables de calendario."""
    t = pl.col("datetime")
    return df.with_columns(
        no_submedido().clip(lower_bound=0).cast(pl.Float32).alias("no_submedido_wh"),
        t.dt.year().cast(pl.Int16).alias("anio"),
        t.dt.hour().cast(pl.Int8).alias("hora"),
        categorica(t.dt.weekday(), dict(enumerate(DIAS, 1)), DIAS).alias("dia_semana"),
        categorica(t.dt.month(), dict(enumerate(MESES, 1)), MESES).alias("mes"),
        categorica(t.dt.month(), ESTACION_POR_MES, ESTACIONES).alias("estacion"),
        categorica(t.dt.hour() // 6, dict(enumerate(FRANJAS)), FRANJAS).alias(
            "franja_horaria"
        ),
        (t.dt.weekday() >= 6).alias("es_fin_semana"),
    )


def preprocesar() -> tuple[pl.DataFrame, dict]:
    """Ejecuta las etapas en orden y devuelve el dataset limpio y sus conteos."""
    crudo = cargar_crudo_texto()
    info = {"Filas en crudo": crudo.height}

    df = unificar_faltantes(crudo)
    info["Nulos por medición tras unificar"] = sorted(
        {df[c].null_count() for c in COLUMNAS_NUMERICAS}
    )

    df = castear(df)
    df, info["Duplicados eliminados"] = deduplicar(df)

    df = imputar(df)
    info["No submedido negativo (recortado a 0)"] = df.select(
        (no_submedido() < 0).sum()
    ).item()

    df = derivar(df)
    info["Filas finales"] = df.height
    info["Memoria crudo -> limpio (MB)"] = (
        f"{crudo.estimated_size('mb'):.0f} -> {df.estimated_size('mb'):.0f}"
    )
    return df, info


def main() -> None:
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    df, info = preprocesar()
    ARCHIVO_LIMPIO.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(ARCHIVO_LIMPIO, compression="zstd")

    for k, v in info.items():
        print(f"{k:<40} {v}")
    print("\nOrigen de las filas:")
    print(df["estado_dato"].value_counts().sort("estado_dato"))
    print("\nEsquema:")
    for c, t in df.schema.items():
        print(f"  {c:<24} {t}")
    print(
        f"\nGuardado: {ARCHIVO_LIMPIO} ({ARCHIVO_LIMPIO.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
