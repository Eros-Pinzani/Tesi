# Classe che contiene gli ostacoli (gestiti con Shapely), i confini, le intersezioni e funzioni di utilità geometriche

import matplotlib.pyplot as plt  # Libreria per la visualizzazione grafica
from shapely.geometry import box, LineString, Point  # Primitive geometriche di Shapely
from shapely.ops import unary_union  # Operazione per unire più geometrie
from shapely.geometry.base import BaseGeometry  # Tipo base per geometrie Shapely
from typing import List, Optional, Tuple  # Tipi per annotazioni statiche


class Environment:
    def __init__(self):
        # Inizializza la lista di ostacoli presenti nell'ambiente
        self.obstacles: List[BaseGeometry] = []  # Lista di ostacoli (oggetti Shapely)
        # Inizializza i confini dell'ambiente (rettangolo), assente all'inizio
        self.bounds: Optional[BaseGeometry] = None  # Confini dell’ambiente (oggetto Shapely)

    def set_bounds(self, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        """Imposta i confini dell’ambiente come un rettangolo."""
        # Crea un rettangolo (bbox) con coordinate min/max e lo assegna ai confini
        self.bounds = box(xmin, ymin, xmax, ymax)

    def add_obstacle(self, poly: BaseGeometry) -> None:
        """Aggiunge un ostacolo all’ambiente."""
        # Inserisce l'oggetto geometrico passato nella lista di ostacoli
        self.obstacles.append(poly)

    def add_rectangle(self, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        """Aggiunge un ostacolo rettangolare all’ambiente."""
        # Crea e aggiunge un rettangolo come ostacolo con le coordinate fornite
        self.obstacles.append(box(xmin, ymin, xmax, ymax))

    def clear(self) -> None:
        # Svuota la lista degli ostacoli e rimuove i confini
        self.obstacles = []
        self.bounds = None

    def obstacles_union(self) -> Optional[BaseGeometry]:
        """Unisce tutti gli ostacoli in un'unica geometria per operazioni di intersezione più veloci"""
        # Se non ci sono ostacoli, non c'è nulla da unire
        if not self.obstacles:
            return None
        # Unisce tutte le geometrie degli ostacoli in un'unione geometrica (multi-poligono o simile)
        return unary_union(self.obstacles)

    def first_intersection_with_line(self, line: LineString) -> Optional[Tuple[float, float]]:
        """Restituisce il primo punto di intersezione tra una linea e gli ostacoli dell’ambiente come (x,y) tuple, oppure None se non c’è intersezione."""
        # Calcola l'unione degli ostacoli per ottimizzare le intersezioni
        union = self.obstacles_union()
        if union is None:
            return None  # Nessun ostacolo => nessuna intersezione
        # Interseca la linea con l'unione degli ostacoli
        intersection = line.intersection(union)
        if intersection.is_empty:
            return None  # La linea non interseca nessun ostacolo
        # Possibili tipi di intersezione: Point, MultiPoint, LineString, GeometryCollection
        # Estrazione dei punti e scelta del primo punto lungo la linea (il più vicino all'origine del raggio)
        origin = Point(line.coords[0])  # Punto di origine della linea (primo vertice)

        # Funzione ausiliaria per estrarre punti da diverse geometrie di intersezione
        def _extract_points(geom: BaseGeometry) -> List[Point]:
            pts: List[Point] = []  # Lista dei punti estratti
            geom_type = geom.geom_type  # Tipo della geometria risultante
            if geom_type == 'Point':
                # Intersezione puntuale
                pts.append(geom)  # type: ignore[arg-type]
            elif geom_type in ('MultiPoint', 'GeometryCollection'):
                # Collezione di più geometrie: itera e raccogli punti e estremi di linee
                for g in geom.geoms:  # type: ignore[attr-defined]
                    if g.geom_type == 'Point':
                        pts.append(g)  # type: ignore[arg-type]
                    elif g.geom_type == 'LineString':
                        # Prendo i punti estremi della linea di intersezione
                        pts.append(Point(g.coords[0]))
                        pts.append(Point(g.coords[-1]))
            elif geom_type in ('LineString', 'LinearRing'):
                # Per una linea, considero gli estremi come possibili punti di contatto
                pts.append(Point(geom.coords[0]))
                pts.append(Point(geom.coords[-1]))
            elif geom_type == 'Polygon':
                # Per poligoni, prendo un punto dell'anello esterno (qui l'inizio)
                pts.append(Point(geom.exterior.coords[0]))  # type: ignore[attr-defined]
            else:
                # Fallback: prova a iterare eventuali sotto-geometrie
                try:
                    for g in geom.geoms:  # type: ignore[attr-defined]
                        pts.extend(_extract_points(g))  # Estrai ricorsivamente
                except (AttributeError, TypeError):
                    # Il tipo non è iterabile o non possiede 'geoms'
                    pass
            return pts  # Restituisce la lista dei punti trovati

        # Estrae una lista di punti significativi dall'intersezione
        pts = _extract_points(intersection)
        if not pts:
            return None  # Nessun punto identificabile
        # Scegli il punto con distanza minima dall'origine della linea
        pts_sorted = sorted(pts, key=lambda p: p.distance(origin))  # Ordina per distanza crescente
        nearest = pts_sorted[0]  # Primo è il più vicino
        return float(nearest.x), float(nearest.y)  # Restituisce coordinate (x, y)

    def plot(self, ax=None, facecolor: str = 'lightgrey', edgecolor: str = 'k') -> None:
        """Disegna bounds e ostacoli (se ax è None crea una figura nuova)"""
        own_fig = False  # Indica se la figura è gestita internamente (show) o dall'esterno
        if ax is None:
            _fig, ax = plt.subplots(figsize=(7, 7))  # Crea figura e assi se non forniti
            own_fig = True
        if self.bounds is not None:
            x, y = self.bounds.exterior.xy  # type: ignore[attr-defined]  # Estrae le coordinate del contorno dei confini
            ax.plot(x, y, color=edgecolor, linewidth=1)  # Disegna il bordo dei confini
            ax.fill(x, y, alpha=0.03, facecolor=facecolor)  # Riempie leggermente l'area dei confini
        for poly in self.obstacles:
            x, y = poly.exterior.xy  # type: ignore[attr-defined]  # Estrae le coordinate del contorno dell'ostacolo
            # Disegna e riempie ogni ostacolo con colore e bordo specificati
            ax.fill(x, y, alpha=0.6, facecolor='tab:gray', edgecolor=edgecolor)
        if own_fig:
            ax.set_aspect('equal', 'box')  # Mantiene il rapporto d'aspetto 1:1
            plt.show()  # Mostra la figura se creata internamente

