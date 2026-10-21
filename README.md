# Consumo eléctrico doméstico · Preparación de datos y dashboard

Práctica 5 · Almacenes y Minería de Datos · Facultad de Ciencias, UNAM

**Equipo:** _(nombres de los 3 integrantes)_

## Objetivo

Preparar una serie de tiempo de consumo eléctrico de un hogar para entrenar modelos de
regresión y agrupamiento: preprocesamiento, normalización numérica, reducción de
dimensionalidad (correlación de Pearson y PCA) y un dashboard interactivo en Streamlit
para explorar los datos y comunicar hallazgos.

## Dataset

**Individual Household Electric Power Consumption** (Hebrail y Berard, 2006),
UCI Machine Learning Repository:
<https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption>

- 2,075,259 mediciones **por minuto** de un hogar en Sceaux, Francia
  (diciembre de 2006 a noviembre de 2010).
- 9 columnas: `Date`, `Time`, `Global_active_power` (kW), `Global_reactive_power` (kW),
  `Voltage` (V), `Global_intensity` (A), `Sub_metering_1` (cocina, Wh),
  `Sub_metering_2` (lavandería, Wh) y `Sub_metering_3` (calentador de agua y aire
  acondicionado, Wh).

### Cómo obtener los datos

El archivo original (≈ 127 MB) supera el límite de GitHub, así que **no se versiona**.

1. Descargar el `.zip` desde la página de UCI.
2. Descomprimir y colocar `household_power_consumption.txt`, **sin modificarlo**, en
   `data/data-raw/`.

## Estructura del proyecto

```
proyecto/
├── config/
│   └── rutas.py              # rutas estáticas del proyecto
├── data/
│   ├── data-raw/             # dato original, sin modificar (no versionado)
│   └── data-processed/       # dato preparado
├── src/
│   ├── cleaning/             # carga, inspección y preprocesamiento
│   ├── features/             # normalización, Pearson y PCA
│   └── dashboard/            # carga, métricas y gráficas que usa el dashboard
├── visualization/
│   └── app.py                # código principal del dashboard
├── README.md
└── requirements.txt
```

## Instalación

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

Todos los comandos se corren desde la raíz del proyecto.

_(Se agregarán los comandos de cada paso y del dashboard.)_

## Decisiones principales

_(Se completa al avanzar cada paso.)_
