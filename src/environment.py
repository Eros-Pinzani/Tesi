# Classe che contiene gli ostacoli (gestiti con Shapely), i confini e funzioni di utilità geometriche minime

import matplotlib.pyplot as plt  # per disegnare bounds e ostacoli
from shapely.geometry import box  # Primitive geometriche di Shapely necessarie
from shapely.ops import unary_union  # Operazione per unire più geometrie
from shapely.geometry.base import BaseGeometry  # Tipo base per geometrie Shapely
from typing import List, Optional  # Tipi per annotazioni statiche


class Environment:
    def __init__(self):
        # Inizializza la lista di ostacoli presenti nell'ambiente
        self.obstacles: List[BaseGeometry] = []  # Lista di ostacoli (oggetti Shapely)
        # Inizializza i confini dell'ambiente (rettangolo), assente all'inizio
        self.bounds: Optional[BaseGeometry] = None  # Confini dell’ambiente (oggetto Shapely)

    def set_bounds(self, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        """Imposta i confini dell’ambiente come un rettangolo axis-aligned."""
        self.bounds = box(float(xmin), float(ymin), float(xmax), float(ymax))

    def add_rectangle(self, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        """Aggiunge un ostacolo rettangolare all’ambiente."""
        self.obstacles.append(box(float(xmin), float(ymin), float(xmax), float(ymax)))

    def obstacles_union(self) -> Optional[BaseGeometry]:
        """Unisce tutti gli ostacoli in un'unica geometria per intersezioni più veloci; None se non ci sono ostacoli."""
        if not self.obstacles:
            return None
        return unary_union(self.obstacles)

    def plot(self, ax=None, facecolor: str = 'lightgrey', edgecolor: str = 'k') -> None:
        """Disegna bounds e ostacoli (se ax è None crea una figura nuova)."""
        own_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
            own_fig = True
        # Disegna bounds (se presenti)
        if self.bounds is not None:
            x, y = self.bounds.exterior.xy  # type: ignore[attr-defined]
            ax.plot(x, y, color=edgecolor, linewidth=1.0, zorder=0)
            ax.fill(x, y, alpha=0.04, facecolor=facecolor, edgecolor='none', zorder=0)
        # Disegna ogni ostacolo come poligono riempito
        for poly in self.obstacles:
            x, y = poly.exterior.xy  # type: ignore[attr-defined]
            ax.fill(x, y, alpha=0.6, facecolor='tab:gray', edgecolor=edgecolor, linewidth=1.0, zorder=1)
        if own_fig:
            ax.set_aspect('equal', 'box')
            plt.show()
