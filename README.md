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
├── config/
│   └── rutas.py           # rutas estáticas
├── data/
│   ├── data-raw/          # dato original (no versionado)
│   └── data-processed/    # dato preparado
├── src/
│   ├── cleaning/          # carga, inspección y preprocesamiento
│   ├── features/          # normalización, Pearson y PCA
│   └── dashboard/         # módulos auxiliares del dashboard
├── visualization/         # código principal del dashboard
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
python -m src.cleaning.inspeccion    # inspección del dato crudo
```
