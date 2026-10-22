"""Paleta Catppuccin Mocha del dashboard.

Los colores de la interfaz (fondos, controles, badges, paletas por defecto de las
gráficas) se definen en .streamlit/config.toml. Este módulo da los colores con
significado que las gráficas usan de forma explícita (por ejemplo, un color por
uso de energía).
"""

from pathlib import Path

# Paleta oficial: https://catppuccin.com/palette
MOCHA = {
    "rosewater": "#f5e0dc",
    "flamingo": "#f2cdcd",
    "pink": "#f5c2e7",
    "mauve": "#cba6f7",
    "red": "#f38ba8",
    "maroon": "#eba0ac",
    "peach": "#fab387",
    "yellow": "#f9e2af",
    "green": "#a6e3a1",
    "teal": "#94e2d5",
    "sky": "#89dceb",
    "sapphire": "#74c7ec",
    "blue": "#89b4fa",
    "lavender": "#b4befe",
    "text": "#cdd6f4",
    "subtext1": "#bac2de",
    "subtext0": "#a6adc8",
    "overlay2": "#9399b2",
    "overlay1": "#7f849c",
    "overlay0": "#6c7086",
    "surface2": "#585b70",
    "surface1": "#45475a",
    "surface0": "#313244",
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
}

ICONO = Path(__file__).parent / "assets" / "icono.svg"


def mezclar(color: str, fondo: str, proporcion: float) -> str:
    """Mezcla dos colores hex; `proporcion` es el peso de `color` (0 a 1)."""
    a = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(fondo[i : i + 2], 16) for i in (1, 3, 5)]
    mezcla = [round(x * proporcion + y * (1 - proporcion)) for x, y in zip(a, b)]
    return "#" + "".join(f"{v:02x}" for v in mezcla)
