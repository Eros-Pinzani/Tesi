# Preset e utility per creare e configurare Environment sulla base delle traiettorie

from typing import List
import numpy as np
from environment import Environment


def setup_environment(histories: List[np.ndarray]) -> Environment:
    """Crea e configura un Environment a partire dall'estensione complessiva delle traiettorie.

    - Calcola bounds con un padding proporzionale all'estensione complessiva.
    - Aggiunge alcuni ostacoli di prova ben visibili vicino alle traiettorie.
    """
    env = Environment()
    try:
        all_xy = np.vstack([h[:, :2] for h in histories])
        x_min, y_min = np.min(all_xy[:, 0]), np.min(all_xy[:, 1])
        x_max, y_max = np.max(all_xy[:, 0]), np.max(all_xy[:, 1])
        span_x = float(x_max) - float(x_min)
        span_y = float(y_max) - float(y_min)
        pad = 0.15 * max(span_x, span_y, 1.0)
        env.set_bounds(float(x_min - pad), float(y_min - pad), float(x_max + pad), float(y_max + pad))
    except Exception:
        # Fallback in caso di problemi: bounds standard centrati in (0,0)
        env.set_bounds(-5.0, -5.0, 5.0, 5.0)

    # Ostacoli di prova (vicini alle traiettorie per essere ben visibili)
    env.add_rectangle(-0.25, -0.25, 0.25, 0.25)   # pilastro centrale
    env.add_rectangle(2.0, -0.5, 3.0, 0.5)        # rettangolo lungo la retta
    env.add_rectangle(6.0, 0.8, 7.0, 1.8)         # rettangolo sopra la retta
    return env


def setup_environments_per_trajectory(histories: List[np.ndarray], titles: List[str]) -> List[Environment]:
    """Crea un Environment distinto per ogni traiettoria, con ostacoli specifici.

    - I bounds sono calcolati per-traiettoria a partire dall'estensione della singola storia con un padding moderato.
    - Gli ostacoli vengono posizionati come frazioni dei bounds per evitare degenerazioni su traiettorie piatte.
    """
    envs: List[Environment] = []

    def _compute_bounds_for_hist(hist: np.ndarray) -> tuple[float, float, float, float]:
        xs = hist[:, 0]
        ys = hist[:, 1]
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        y_min, y_max = float(np.min(ys)), float(np.max(ys))
        span_x = max(1e-9, x_max - x_min)
        span_y = max(1e-9, y_max - y_min)
        pad = 0.15 * max(span_x, span_y, 1.0)
        return x_min - pad, y_min - pad, x_max + pad, y_max + pad

    def _add_rect_frac(env: Environment, bx0: float, by0: float, bx1: float, by1: float, fx0: float, fy0: float, fx1: float, fy1: float) -> None:
        """Aggiunge un rettangolo come frazioni dei bounds [bx0,bx1]×[by0,by1]."""
        xa = bx0 + fx0 * (bx1 - bx0)
        xb = bx0 + fx1 * (bx1 - bx0)
        ya = by0 + fy0 * (by1 - by0)
        yb = by0 + fy1 * (by1 - by0)
        # Garantisce un minimo spessore nel caso (per sicurezza)
        if abs(yb - ya) < 1e-6:
            yb = ya + 0.05 * (by1 - by0)
        if abs(xb - xa) < 1e-6:
            xb = xa + 0.05 * (bx1 - bx0)
        env.add_rectangle(min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))

    for idx, (hist, _title) in enumerate(zip(histories, titles)):
        env = Environment()
        bx0, by0, bx1, by1 = _compute_bounds_for_hist(hist)
        env.set_bounds(bx0, by0, bx1, by1)

        # Posiziona ostacoli come frazioni dei bounds per garantire dimensioni ben visibili
        if idx == 0:  # Rettilinea (v costante)
            _add_rect_frac(env, bx0, by0, bx1, by1, 0.28, 0.12, 0.36, 0.32)  # piccolo ostacolo basso a sinistra
            _add_rect_frac(env, bx0, by0, bx1, by1, 0.62, 0.58, 0.78, 0.82)  # ostacolo alto a destra
        elif idx == 1:  # Rettilinea (v variabile)
            _add_rect_frac(env, bx0, by0, bx1, by1, 0.18, 0.62, 0.30, 0.86)  # in alto a sinistra
            _add_rect_frac(env, bx0, by0, bx1, by1, 0.54, 0.12, 0.70, 0.30)  # in basso a destra
        elif idx == 2:  # Circolare (v costante)
            _add_rect_frac(env, bx0, by0, bx1, by1, 0.10, 0.66, 0.22, 0.94)  # vicino al top-left della corona
            _add_rect_frac(env, bx0, by0, bx1, by1, 0.78, 0.36, 0.92, 0.54)  # lato destro
        elif idx == 3:  # Circolare (v variabile)
            _add_rect_frac(env, bx0, by0, bx1, by1, 0.25, 0.16, 0.40, 0.36)  # basso-sinistra
            _add_rect_frac(env, bx0, by0, bx1, by1, 0.60, 0.66, 0.80, 0.90)  # alto-destra
        elif idx == 4:  # Traiettoria a 8
            _add_rect_frac(env, bx0, by0, bx1, by1, 0.15, 0.36, 0.30, 0.56)  # vicino al lobo sinistro
            _add_rect_frac(env, bx0, by0, bx1, by1, 0.70, 0.46, 0.88, 0.72)  # vicino al lobo destro
        elif idx == 5:  # Random walk
            # Tre ostacoli sparsi in zone diverse dei bounds
            _add_rect_frac(env, bx0, by0, bx1, by1, 0.20, 0.20, 0.32, 0.38)
            _add_rect_frac(env, bx0, by0, bx1, by1, 0.48, 0.55, 0.62, 0.75)
            _add_rect_frac(env, bx0, by0, bx1, by1, 0.75, 0.25, 0.90, 0.40)
        else:
            # Caso generico
            _add_rect_frac(env, bx0, by0, bx1, by1, 0.35, 0.35, 0.50, 0.55)

        envs.append(env)

    return envs

