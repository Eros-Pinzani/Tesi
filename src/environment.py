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
        pass
        #TODO

    def plot(self, ax):
        """Placeholder per il disegno dell’ambiente (attualmente vuoto)."""
        pass