"""
Gestione dell'ambiente 2D con ostacoli geometrici.
Utilizza Shapely per operazioni geometriche efficienti e ray-casting.
"""

import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, Polygon, LineString
from typing import List, Optional
from shapely.errors import TopologicalError


class Environment:
    def __init__(self):
        """
        Inizializza un ambiente vuoto con lista di ostacoli e confini non definiti.
        """
        self.obstacles: List[BaseGeometry] = []  # Lista di geometrie Shapely rappresentanti ostacoli
        self.bounds: Optional[BaseGeometry] = None  # Confini dell'ambiente
        self._union_cache: Optional[BaseGeometry] = None  # Cache dell'unione degli ostacoli
        self._union_dirty: bool = True  # Flag per invalidare la cache

    def set_bounds(self, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        """
        Imposta i confini dell'ambiente come rettangolo axis-aligned.

        Args:
            xmin: Coordinata x minima
            ymin: Coordinata y minima
            xmax: Coordinata x massima
            ymax: Coordinata y massima
        """
        self.bounds = box(float(xmin), float(ymin), float(xmax), float(ymax))

    def add_rectangle(self, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        """
        Aggiunge un ostacolo rettangolare axis-aligned.

        Args:
            xmin: Coordinata x minima del rettangolo
            ymin: Coordinata y minima del rettangolo
            xmax: Coordinata x massima del rettangolo
            ymax: Coordinata y massima del rettangolo
        """
        self.obstacles.append(box(float(xmin), float(ymin), float(xmax), float(ymax)))
        self._union_dirty = True

    def add_circle(self, cx: float, cy: float, radius: float, *, resolution: int = 32) -> None:
        """
        Aggiunge un ostacolo circolare approssimato come poligono.

        Args:
            cx: Coordinata x del centro
            cy: Coordinata y del centro
            radius: Raggio del cerchio (metri)
            resolution: Numero di segmenti per approssimare il cerchio
        """
        r = max(1e-6, float(radius))
        self.obstacles.append(Point(float(cx), float(cy)).buffer(r, resolution=resolution))
        self._union_dirty = True

    def add_polygon(self, vertices: List[tuple]) -> None:
        """
        Aggiunge un ostacolo poligonale generico.

        Args:
            vertices: Lista di tuple (x, y) che definiscono i vertici del poligono
        """
        if not vertices:
            return
        self.obstacles.append(Polygon([(float(x), float(y)) for x, y in vertices]))
        self._union_dirty = True

    def add_wall(self, x0: float, y0: float, x1: float, y1: float, thickness: float = 0.10) -> None:
        """
        Aggiunge un muro sottile tra due punti.

        Il muro è creato come buffer di un segmento lineare.

        Args:
            x0: Coordinata x del primo estremo
            y0: Coordinata y del primo estremo
            x1: Coordinata x del secondo estremo
            y1: Coordinata y del secondo estremo
            thickness: Spessore del muro (metri)
        """
        t = max(1e-6, float(thickness))
        seg = LineString([(float(x0), float(y0)), (float(x1), float(y1))])
        self.obstacles.append(seg.buffer(0.5 * t, cap_style='square', join_style='mitre'))
        self._union_dirty = True

    def obstacles_union(self) -> Optional[BaseGeometry]:
        """
        Restituisce l'unione di tutti gli ostacoli come singola geometria.

        Usa una cache per evitare ricalcoli ripetuti. La cache viene invalidata
        quando vengono aggiunti nuovi ostacoli.

        Returns:
            Geometria unita di tutti gli ostacoli, o None se non ci sono ostacoli
        """
        if not self.obstacles:
            self._union_cache = None
            self._union_dirty = False
            return None

        if self._union_cache is None or self._union_dirty:
            self._union_cache = unary_union(self.obstacles)
            self._union_dirty = False

        return self._union_cache

    def first_intersection_with_line(self, line: LineString):
        """
        Trova il punto di intersezione più vicino tra un raggio e gli ostacoli.

        Args:
            line: Geometria LineString rappresentante il raggio

        Returns:
            Tupla (x, y) del punto di intersezione più vicino all'origine del raggio,
            o None se non ci sono intersezioni
        """
        union = self.obstacles_union()
        if union is None:
            return None

        inter = line.intersection(union)
        if inter.is_empty:
            return None

        origin = Point(line.coords[0])

        try:
            # Caso 1: Intersezione singola
            if isinstance(inter, Point):
                return float(inter.x), float(inter.y)

            # Caso 2: Intersezioni multiple
            if getattr(inter, 'geom_type', '') == 'MultiPoint':
                best = None
                best_d = float('inf')
                for pt in inter.geoms:  # type: ignore[attr-defined]
                    d = origin.distance(pt)
                    if d < best_d:
                        best_d = d
                        best = pt
                if best is not None:
                    return float(best.x), float(best.y)

            # Caso 3: Geometrie complesse
            from shapely.ops import nearest_points
            _, p = nearest_points(origin, inter)
            return float(p.x), float(p.y)

        except (TopologicalError, AttributeError, TypeError, ValueError):
            # Fallback robusto
            def _iter_points(g):
                """Itera ricorsivamente su tutti i punti di una geometria."""
                gt = getattr(g, 'geom_type', '')
                if gt == 'Point':
                    yield g
                elif gt == 'MultiPoint':
                    for h in g.geoms:
                        yield h
                elif gt in ('LineString', 'LinearRing'):
                    for (x, y) in g.coords:
                        yield Point(x, y)
                elif gt == 'MultiLineString':
                    for h in g.geoms:
                        yield from _iter_points(h)
                elif gt == 'GeometryCollection':
                    for h in g.geoms:
                        yield from _iter_points(h)

            best = None
            best_d = float('inf')
            for pt in _iter_points(inter):
                d = origin.distance(pt)
                if d < best_d:
                    best_d = d
                    best = pt

            if best is None:
                return None
            return float(best.x), float(best.y)

    def plot(self, ax=None, facecolor: str = 'lightgrey', edgecolor: str = 'k') -> None:
        """
        Visualizza l'ambiente con matplotlib.

        Args:
            ax: Axes matplotlib su cui disegnare (crea una nuova figura se None)
            facecolor: Colore di riempimento dei bounds
            edgecolor: Colore dei bordi
        """
        own_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
            own_fig = True

        # Disegna i confini dell'ambiente
        if self.bounds is not None:
            x, y = self.bounds.exterior.xy  # type: ignore[attr-defined]
            ax.plot(x, y, color=edgecolor, linewidth=1.0, zorder=0)
            ax.fill(x, y, alpha=0.04, facecolor=facecolor, edgecolor='none', zorder=0)

        # Disegna ogni ostacolo
        for poly in self.obstacles:
            x, y = poly.exterior.xy  # type: ignore[attr-defined]
            ax.fill(x, y, alpha=0.6, facecolor='tab:gray', edgecolor=edgecolor, linewidth=1.0, zorder=1)

        if own_fig:
            ax.set_aspect('equal', 'box')
            plt.show()
