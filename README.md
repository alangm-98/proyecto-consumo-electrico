# Consumo eléctrico doméstico · Preparación de datos, dashboard y reglas de asociación

Prácticas 5 y 6 · Almacenes y Minería de Datos · Facultad de Ciencias, UNAM

**Equipo:**

- Flores Juárez Luis Enrique
- García Morales Carlos Alan
- Rangel Salcedo Melanie Valeria

## Objetivo

Preparar una serie de tiempo de consumo eléctrico de un hogar para modelos de regresión y
agrupamiento: preprocesamiento, normalización numérica, reducción de dimensionalidad
(correlación de Pearson y PCA) y un dashboard interactivo en Streamlit (Práctica 5).
Buscar con reglas de asociación qué habitaciones, horarios y niveles de consumo ocurren juntos
(Práctica 6).

## Dataset

**Individual Household Electric Power Consumption** (Hebrail y Berard, 2006),
UCI Machine Learning Repository:
<https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption>

- 2,075,259 mediciones por minuto de un hogar en Sceaux, Francia
  (diciembre de 2006 a noviembre de 2010).
- 9 columnas: `Date`, `Time`, `Global_active_power` (kW), `Global_reactive_power` (kW),
  `Voltage` (V), `Global_intensity` (A), `Sub_metering_1` (cocina, Wh),
  `Sub_metering_2` (lavandería, Wh) y `Sub_metering_3` (calentador de agua y aire
  acondicionado, Wh).

El archivo original (≈ 127 MB) supera el límite de GitHub y no se versiona. Se descarga
desde la página de UCI y `household_power_consumption.txt` se coloca, sin modificar, en
`data/data-raw/`.

## Estructura

```
├── .streamlit/
│   └── config.toml        # tema del dashboard (Catppuccin Mocha)
├── config/
│   └── rutas.py           # rutas estáticas
├── data/
│   ├── data-raw/          # dato original (no versionado)
│   └── data-processed/    # dato preparado, parámetros y datos del dashboard
├── reports/
│   └── figures/           # Pearson, PCA y reglas de asociación
├── src/
│   ├── cleaning/          # carga, inspección y preprocesamiento
│   ├── features/          # normalización, Pearson, PCA y dataset final
│   ├── dashboard/         # datos, métricas, gráficas, filtros y tema del dashboard
│   └── reglas/            # transacciones, Apriori/FP-Growth y reglas (Práctica 6)
├── visualization/
│   └── app.py             # dashboard en Streamlit
├── README.md
└── requirements.txt
```

## Instalación

Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

Desde la raíz del proyecto, con el entorno activado:

```bash
python -m src.cleaning.inspeccion        # inspección del dato crudo
python -m src.cleaning.preprocesamiento  # limpieza -> data/data-processed/consumo_limpio.parquet
python -m src.features.normalizacion     # z-score (ajustado con 2006-2009) -> consumo_normalizado.parquet
python -m src.features.correlacion       # Pearson y selección de variables -> reports/figures/pearson_heatmap.png
python -m src.features.pca               # PCA (4 componentes) -> parametros_pca.json y figuras en reports/figures/
python -m src.features.dataset_final     # dataset preparado -> data/data-processed/dataset_final.parquet
python -m src.dashboard.datos            # datos del dashboard -> consumo_horario.parquet, pearson_entrenamiento.json y calidad_datos.json
```

Dashboard:

```bash
streamlit run visualization/app.py
```

El dashboard solo lee archivos versionados (resumen por hora, JSON y figuras), así que funciona
sin descargar el dato original ni ejecutar el pipeline.

## Preparación de los datos

**Preprocesamiento** (`src/cleaning/preprocesamiento.py`)

- Faltantes: `?` y campos vacíos se unifican como nulos (25,979 minutos, 1.25 %; siempre las
  7 mediciones a la vez).
- Casteo estricto: `Date` + `Time` a una columna `datetime`; mediciones a `Float32`.
- Deduplicación por marca de tiempo: 0 duplicados; la serie tiene un registro por minuto, sin
  huecos de marca de tiempo. No se deduplica por valores: 142,582 minutos repiten las mediciones
  de otro y son consumo real.
- Interpolación lineal solo en huecos de hasta 60 minutos (288 minutos). Los 9 huecos de más de 60 minutos
  (25,691 minutos; el mayor, de 120 h en agosto de 2010) quedan nulos. La columna `estado_dato`
  indica si cada fila es `original`, `interpolado` o `faltante`.
- Variable derivada `no_submedido_wh` = potencia activa en Wh − (S1 + S2 + S3), recortada a 0
  (era negativa en 1,050 minutos por el redondeo de los submedidores).
- Normalización cualitativa: año, hora, día de la semana, mes, estación (hemisferio norte),
  franja horaria y fin de semana; etiquetas en minúsculas y sin acentos, como categorías ordenadas.

**Normalización numérica** (`src/features/normalizacion.py`)

- Partición cronológica: entrenamiento del 16/12/2006 al 31/12/2009 (1,600,236 filas, 77 %) y
  prueba 2010 (475,023 filas, 23 %).
- Z-score de las 7 mediciones y `no_submedido_wh`, ajustado solo con entrenamiento para no filtrar
  información del periodo de prueba. Parámetros en `parametros_zscore.json`.
- Se elige z-score sobre min-max porque las variables tienen picos reales y fuerte asimetría:
  con min-max, un solo pico fija la escala. Los atípicos se conservan.

**Correlación de Pearson** (`src/features/correlacion.py`)

- Calculada con entrenamiento sin faltantes (1,592,164 filas); umbral de redundancia |r| > 0.8.
- Se descarta `Global_intensity` (r = 0.999 con la potencia activa): se conserva la potencia
  activa por ser la variable objetivo natural y la más interpretable.
- Se descarta `no_submedido_wh`: es una combinación lineal exacta de la potencia activa y los
  submedidores (r = 1.000000), aunque ningún par la marque como redundante.
- Quedan 6 variables: potencia activa, potencia reactiva, voltaje y los 3 submedidores.

**PCA** (`src/features/pca.py`)

- Ajustado con las 6 variables estandarizadas del periodo de entrenamiento.
- Se conservan 4 componentes por varianza acumulada ≥ 80 % (84.6 %). El criterio de Kaiser
  conservaría solo 1, porque las variables restantes están poco correlacionadas.
- PC1 (38.7 %) resume la carga total; PC2 (16.5 %) separa cargas con motor (reactiva,
  lavandería) de cargas resistivas (calentador/AC); PC3 y PC4 contrastan usos entre sí.
- La reducción es moderada (6 → 4) porque Pearson ya eliminó la redundancia fuerte.
- Parámetros en `parametros_pca.json`; figuras en `reports/figures/`.

**Dataset final** (`data/data-processed/dataset_final.parquet`)

2,075,259 filas y 20 columnas:

- contexto: `datetime`, `estado_dato`, `particion` y las 7 variables de calendario;
- 6 variables estandarizadas (`_z`) seleccionadas por Pearson;
- `PC1` a `PC4`, calculados con los parámetros guardados del PCA.

Las filas `faltante` se conservan con nulos para no romper la continuidad de la serie.

## Dashboard

Dos pestañas:

- **Consumo del hogar**: filtros de fechas, agregación, tipo de día y estación; indicadores
  (consumo diario, potencia media, potencia máxima, cobertura); tendencia con estaciones
  sombreadas; energía por uso; perfil diario; distribución del consumo diario con su resumen
  estadístico; tabla por periodo descargable en CSV.
- **Preparación de datos**: calidad del dato (faltantes por mes y huecos sin rellenar),
  decisiones de preparación, correlación de Pearson con dispersión de cualquier par de
  variables y figuras del PCA con la tabla de loadings.

## Práctica 6 · Reglas de asociación

Busca qué habitaciones, momentos del día y niveles de consumo ocurren juntos en una misma
hora (Apriori y FP-Growth con `mlxtend`). Parte del dataset limpio de la Práctica 5
(`consumo_limpio.parquet`, generado por `src.cleaning.preprocesamiento`).

```bash
python -m src.reglas.transacciones   # transacciones e ítems -> input_reglas_asociacion (.parquet y .csv) y reglas_umbrales.png
python -m src.reglas.algoritmos      # Apriori vs FP-Growth -> comparacion_apriori_fpgrowth.csv y reglas_itemsets_soporte.png
python -m src.reglas.reglas          # reglas con lift > 1 -> data/data-processed/reglas_asociacion.csv
python -m src.reglas.graficas        # gráficas de las reglas -> reports/figures/reglas_*.png
```

**Transacciones e ítems** (`src/reglas/transacciones.py`)

- Una transacción es una hora con los 60 minutos medidos: 34,149 horas (se excluyen 421 sin
  medición y 19 parciales).
- Habitaciones activas: energía del submedidor en la hora por encima de un umbral. Con `> 0`,
  Lavandería (incluye el refrigerador) y Climatización (el calentador mantiene temperatura)
  estarían activas en 70 % y 77 % de las horas; los umbrales se colocan en el valle entre ese
  consumo base y el uso real (`reglas_umbrales.png`):

  | Ítem          | Submedidor     | Umbral   | Horas activas |
  | ------------- | -------------- | -------- | ------------- |
  | Cocina        | Sub_metering_1 | > 50 Wh  | 12.4 %        |
  | Lavandería    | Sub_metering_2 | > 100 Wh | 7.4 %         |
  | Climatización | Sub_metering_3 | > 200 Wh | 42.8 %        |

- Tiempo del día: Madrugada 0–5 h, Mañana 6–11 h, Mediodía 12–13 h, Tarde 14–18 h, Noche
  19–23 h (el pico de consumo de 19 a 21 h queda completo en Noche). Día de la semana: Lunes a
  Domingo.
- `Consumo alto`: potencia activa media de la hora ≥ 1.579 kW (percentil 75).
- `input_reglas_asociacion`: métricas globales y submedidores por hora, `Habitaciones_Activas`
  (la canasta), `Dia`, `Tiempo` y `Consumo_Alto`; en Parquet y en CSV (lista separada por `|`).

**Apriori y FP-Growth** (`src/reglas/algoritmos.py`)

- Con `min_support` de 0.001 a 0.3, ambos encuentran los mismos itemsets con el mismo soporte
  (de 530 a 1).
- Apriori es más rápido en todos los valores: hay solo 16 ítems y cada transacción tiene de 2 a
  6, así que construir el FP-tree cuesta más de lo que ahorra.

**Reglas** (`src/reglas/reglas.py`)

- `min_support` = 0.01 (al menos 342 horas) y `min_confidence` = 0.4; se conservan las reglas con
  lift > 1 cuyo consecuente es una habitación o `Consumo alto` (predecir un día o una franja no
  aporta). Resultado: 145 reglas, ordenadas por lift.
- Tipo `uso` (115): el antecedente incluye habitaciones o consumo. Tipo `rutina` (30): el
  antecedente solo tiene franja y día.

**Hallazgos**

- Cocina y calentador encendidos en la misma hora: consumo alto el 90 % de las veces (lift 3.6);
  con las tres habitaciones, el 97.6 %.
- La noche tiene consumo alto todos los días (lift 1.55–2.04); el sábado por la noche, la mitad
  de las horas.
- En fin de semana el mediodía y la tarde también son de consumo alto (lift 1.59–1.74); entre
  semana quedan por debajo del promedio.
- Entre semana el calentador se enciende en la mañana: {Mañana, Miércoles} ⇒ {Climatización},
  confianza 0.76.
- Las reglas indican co-ocurrencia, no causalidad.
