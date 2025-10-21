# Classe che contiene gli ostacoli (gestiti con Shapely), i confini, le intersezioni e funzioni di utilità geometriche

from shapely.geometry import Polygon, box, LineString, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt

class Environment:
    def __init__(self):
        self.obstacles = [] # Lista di ostacoli (oggetti Shapely)
        self.bounds = None # Confini dell’ambiente (oggetto Shapely)

    def set_bounds(self, xmin, ymin, xmax, ymax):
        """Imposta i confini dell’ambiente come un rettangolo."""
        self.bounds = box(xmin, ymin, xmax, ymax)

    def add_obstacle(self, poly):
        """Aggiunge un ostacolo all’ambiente."""
        self.obstacles.append(poly)

    def add_rectangle(self, xmin, ymin, xmax, ymax):
        """Aggiunge un ostacolo rettangolare all’ambiente."""
        self.obstacles.append(box(xmin, ymin, xmax, ymax))

    def clear(self):
        self.obstacles = []
        self.bounds = None

    def obstacles_union(self):
        """Unisce tutti gli ostacoli in un'unica geometria per operazioni di intersezione più veloci"""
        if not self.obstacles:
            return None
        return unary_union(self.obstacles)

    def first_intersection_with_line(self, line: LineString):
        """Restituisce il primo punto di intersezione tra una linea e gli ostacoli dell’ambiente come (x,y) tuple, oppure None se non c’è intersezione."""
        union = self.obstacles_union()
        if union is None:
            return None
        intersection = line.intersection(union)
        if intersection.is_empty:
            return None
        # Possibili tipi di intersezione: Point, MultiPoint, LineString, GeometryCollection
        # Estrazione dei punti e scelta del primo punto lungo la linea (il più vicino all'origine del raggio)
        origin = Point(line.coords[0])

        # Funzione ausiliaria per estrarre punti da diverse geometrie
        def _extract_points(geom):
            pass
            # TODO

    def plot(self, ax):
        """Placeholder per il disegno dell’ambiente (attualmente vuoto)."""
        pass