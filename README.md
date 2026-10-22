# Consumo eléctrico doméstico · Preparación de datos y dashboard

Práctica 5 · Almacenes y Minería de Datos · Facultad de Ciencias, UNAM

**Equipo:**

- Flores Juárez Luis Enrique
- García Morales Carlos Alan
- Rangel Salcedo Melanie Valeria

## Objetivo

Preparar una serie de tiempo de consumo eléctrico de un hogar para modelos de regresión y
agrupamiento: preprocesamiento, normalización numérica, reducción de dimensionalidad
(correlación de Pearson y PCA) y un dashboard interactivo en Streamlit.

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
│   └── figures/           # mapa de calor de Pearson y figuras del PCA
├── src/
│   ├── cleaning/          # carga, inspección y preprocesamiento
│   ├── features/          # normalización, Pearson, PCA y dataset final
│   └── dashboard/         # datos, métricas, gráficas, filtros y tema del dashboard
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
