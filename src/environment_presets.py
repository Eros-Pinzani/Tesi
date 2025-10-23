# Preset e utility per creare e configurare Environment sulla base delle traiettorie

from typing import List, Tuple
import numpy as np
from environment import Environment
# Geometrie per calcoli strategici (buffer corridoio e piazzamento)
from shapely.geometry import LineString, Point, box as shapely_box


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

    Principi di posizionamento "strategico" per un LIDAR:
    - Ostacoli distribuiti a differenti portate e direzioni attorno al percorso per produrre scansioni ricche.
    - Evita l'ambiguità (niente simmetrie perfette): forme/scale diverse e posizioni non speculari.
    - Nessun ostacolo sul percorso: si usa un corridoio di sicurezza attorno alla traiettoria.
    """
    envs: List[Environment] = []

    def _compute_bounds_for_hist(hist: np.ndarray) -> Tuple[float, float, float, float]:
        xs = hist[:, 0]
        ys = hist[:, 1]
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        y_min, y_max = float(np.min(ys)), float(np.max(ys))
        span_x = max(1e-9, x_max - x_min)
        span_y = max(1e-9, y_max - y_min)
        pad = 0.15 * max(span_x, span_y, 1.0)
        return x_min - pad, y_min - pad, x_max + pad, y_max + pad

    def _safety_clearance(bx0: float, by0: float, bx1: float, by1: float) -> float:
        """Spessore del corridoio di sicurezza attorno al percorso, in metri.
        Scala con l'estensione, ma non scende sotto un minimo ragionevole legato al corpo del robot (~0.4 m)."""
        span = max(bx1 - bx0, by1 - by0, 1.0)
        return float(min(max(0.20, 0.08 * span), 0.60))

    def _dims_from_frac(bx0: float, by0: float, bx1: float, by1: float, wf: float, hf: float, *, min_size: float = 0.20) -> Tuple[float, float]:
        """Dimensioni assolute (w, h) a partire da frazioni dei bounds, con un minimo fisso."""
        W = max(min_size, float(wf) * (bx1 - bx0))
        H = max(min_size, float(hf) * (by1 - by0))
        return W, H

    def _clamp(val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    def _rect_from_center(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
        return float(cx - w/2), float(cy - h/2), float(cx + w/2), float(cy + h/2)

    def _inside_bounds(env: Environment, rect_poly) -> bool:
        try:
            return env.bounds.contains(rect_poly)  # type: ignore[union-attr]
        except Exception:
            return True

    def _intersects_any(env: Environment, geom) -> bool:
        # Interseca la traiettoria (buffer) o altri ostacoli già piazzati
        if hasattr(geom, 'is_empty') and geom.is_empty:
            return True
        # Contro ostacoli esistenti
        for ob in env.obstacles:
            if geom.intersects(ob):
                return True
        return False

    def _place_rect_frac_strategic(
        env: Environment,
        bx0: float,
        by0: float,
        bx1: float,
        by1: float,
        path_line: LineString,
        path_buffer,
        fx: float,
        fy: float,
        wf: float,
        hf: float,
        *,
        grow_out: float = 0.12,
        max_iter: int = 18,
    ) -> None:
        """Piazza un rettangolo centrato alla frazione (fx, fy) dei bounds, con dimensioni
        proporzionali (wf, hf), spostandolo leggermente se interseca il corridoio di sicurezza
        o altri ostacoli.
        """
        # Dimensioni iniziali
        W, H = _dims_from_frac(bx0, by0, bx1, by1, wf, hf, min_size=0.20)
        # Centro iniziale da frazioni
        cx = bx0 + float(fx) * (bx1 - bx0)
        cy = by0 + float(fy) * (by1 - by0)

        # Helper per costruire poligono rettangolo shapely
        def _rect_poly_at(cx_: float, cy_: float, W_: float, H_: float):
            x0, y0, x1, y1 = _rect_from_center(cx_, cy_, W_, H_)
            return shapely_box(x0, y0, x1, y1)

        # Assicurati che parta dentro ai bounds
        cx = _clamp(cx, bx0 + W/2, bx1 - W/2)
        cy = _clamp(cy, by0 + H/2, by1 - H/2)

        rect = _rect_poly_at(cx, cy, W, H)

        # Step di spostamento proporzionale alla clearance
        step = max(0.02 * max(bx1 - bx0, by1 - by0), 0.10)

        # Strategia: se interseca la traiettoria o ostacoli, sposta il centro nella direzione dal punto più vicino del percorso verso l'esterno
        it = 0
        while (rect.intersects(path_buffer) or _intersects_any(env, rect) or not _inside_bounds(env, rect)) and it < max_iter:
            # Punto del percorso più vicino al centro
            try:
                s = float(path_line.project(Point(cx, cy)))
                p_closest = path_line.interpolate(s)
                vx = float(cx - p_closest.x)
                vy = float(cy - p_closest.y)
            except Exception:
                # fallback: allontanati dal centro dei bounds
                vx = float(cx - 0.5 * (bx0 + bx1))
                vy = float(cy - 0.5 * (by0 + by1))
            # Se vettore è troppo piccolo, scegli direzione verso l'angolo più vicino al bordo opposto
            norm = float(np.hypot(vx, vy))
            if norm < 1e-6:
                vx = (0.5 - fx)
                vy = (0.5 - fy)
                norm = float(np.hypot(vx, vy)) or 1.0
            ux, uy = vx / norm, vy / norm
            # Sposta
            cx += ux * step
            cy += uy * step
            # Rimani entro i bounds (tenendo margine metà lato)
            cx = _clamp(cx, bx0 + W/2, bx1 - W/2)
            cy = _clamp(cy, by0 + H/2, by1 - H/2)
            rect = _rect_poly_at(cx, cy, W, H)
            it += 1

        # Se ancora problematico, prova a ridurre dimensioni e ripeti piccoli tentativi locali
        shrink_attempts = 0
        while (rect.intersects(path_buffer) or _intersects_any(env, rect) or not _inside_bounds(env, rect)) and shrink_attempts < 6:
            W *= (1.0 - 0.12)
            H *= (1.0 - 0.12)
            W = max(W, 0.18)
            H = max(H, 0.18)
            rect = _rect_poly_at(cx, cy, W, H)
            shrink_attempts += 1

        # Se alla fine è valido, aggiungi. In caso contrario, non aggiungere (evita collisioni certo)
        if not (rect.intersects(path_buffer) or _intersects_any(env, rect) or not _inside_bounds(env, rect)):
            x0, y0, x1, y1 = rect.bounds
            env.add_rectangle(x0, y0, x1, y1)

    for idx, (hist, _title) in enumerate(zip(histories, titles)):
        env = Environment()
        bx0, by0, bx1, by1 = _compute_bounds_for_hist(hist)
        env.set_bounds(bx0, by0, bx1, by1)

        # Geometria della traiettoria e corridoio di sicurezza
        path_line = LineString(hist[:, :2].tolist())
        clearance = _safety_clearance(bx0, by0, bx1, by1)
        path_buffer = path_line.buffer(clearance, cap_style='flat', join_style='bevel')

        # Posizioni e dimensioni pensate per ogni traiettoria (frazioni dei bounds)
        # Ogni tripla: (fx, fy, w_frac, h_frac)
        candidates: List[Tuple[float, float, float, float]]
        if idx == 0:  # Rettilinea (v costante) — landmark laterali a distanze diverse
            candidates = [
                (0.22, 0.28, 0.10, 0.16),  # basso-sinistra, rett. verticale
                (0.56, 0.72, 0.14, 0.10),  # alto-centro, rett. orizzontale
                (0.82, 0.34, 0.10, 0.14),  # medio-destra, quasi quadrato
            ]
        elif idx == 1:  # Rettilinea (v variabile) — landmark con diversa combinazione e più lontani
            candidates = [
                (0.18, 0.70, 0.12, 0.10),  # alto-sinistra, orizzontale
                (0.48, 0.24, 0.10, 0.18),  # basso-centro, verticale
                (0.74, 0.58, 0.12, 0.12),  # alto-destra, quadrato
            ]
        elif idx == 2:  # Circolare (v costante) — landmark esterni alla corona in tre settori
            candidates = [
                (0.14, 0.54, 0.10, 0.16),
                (0.50, 0.14, 0.14, 0.10),
                (0.86, 0.62, 0.10, 0.14),
            ]
        elif idx == 3:  # Circolare (v variabile) — simili ma con angoli diversi per spezzare simmetrie
            candidates = [
                (0.22, 0.20, 0.12, 0.10),
                (0.60, 0.84, 0.10, 0.16),
                (0.86, 0.36, 0.12, 0.12),
            ]
        elif idx == 4:  # Traiettoria a 8 — landmark presso i lobi e un separatore centrale
            candidates = [
                (0.18, 0.44, 0.10, 0.16),  # vicino lobo sinistro, verticale
                (0.52, 0.22, 0.16, 0.10),  # sotto incrocio, orizzontale largo
                (0.82, 0.56, 0.12, 0.12),  # vicino lobo destro, quadrato
            ]
        else:  # 5) Random walk — landmark sparsi in tre quadranti
            candidates = [
                (0.20, 0.24, 0.12, 0.10),
                (0.50, 0.72, 0.10, 0.16),
                (0.82, 0.32, 0.12, 0.12),
            ]

        # Inserisci i rettangoli strategici, con auto-aggiustamento anti-collisione percorso/ostacoli
        for fx, fy, wf, hf in candidates:
            _place_rect_frac_strategic(env, bx0, by0, bx1, by1, path_line, path_buffer, fx, fy, wf, hf)

        envs.append(env)

    return envs

