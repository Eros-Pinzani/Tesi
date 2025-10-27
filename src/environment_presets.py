# Preset e utility per creare e configurare Environment sulla base delle traiettorie

from typing import List, Tuple
import numpy as np
from environment import Environment
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

    # ---------- Nuovi helper per forme non rettangolari ----------
    def _bounds_spans(bx0: float, by0: float, bx1: float, by1: float) -> Tuple[float, float]:
        return float(bx1 - bx0), float(by1 - by0)

    def _nearest_outward_dir(path_line: LineString, bx0: float, by0: float, bx1: float, by1: float, cx: float, cy: float, fx: float, fy: float) -> Tuple[float, float]:
        try:
            s = float(path_line.project(Point(cx, cy)))
            p_close = path_line.interpolate(s)
            vx = float(cx - p_close.x)
            vy = float(cy - p_close.y)
        except Exception:
            vx = float(cx - 0.5 * (bx0 + bx1))
            vy = float(cy - 0.5 * (by0 + by1))
        n = float(np.hypot(vx, vy))
        if n < 1e-6:
            vx = (0.5 - fx)
            vy = (0.5 - fy)
            n = float(np.hypot(vx, vy)) or 1.0
        return vx / n, vy / n

    def _place_circle_frac(env: Environment, bx0: float, by0: float, bx1: float, by1: float, path_line: LineString, path_buffer, fx: float, fy: float, r_frac: float, *, max_iter: int = 20) -> None:
        spanx, spany = _bounds_spans(bx0, by0, bx1, by1)
        R = max(0.10, float(r_frac) * 0.5 * min(spanx, spany))
        cx = bx0 + float(fx) * (bx1 - bx0)
        cy = by0 + float(fy) * (by1 - by0)
        cx = _clamp(cx, bx0 + R, bx1 - R)
        cy = _clamp(cy, by0 + R, by1 - R)
        from shapely.geometry import Point as ShapelyPoint
        geom = ShapelyPoint(cx, cy).buffer(R, resolution=32)
        step = max(0.02 * max(spanx, spany), 0.10)
        it = 0
        while (geom.intersects(path_buffer) or _intersects_any(env, geom) or not _inside_bounds(env, geom)) and it < max_iter:
            ux, uy = _nearest_outward_dir(path_line, bx0, by0, bx1, by1, cx, cy, fx, fy)
            cx += ux * step
            cy += uy * step
            cx = _clamp(cx, bx0 + R, bx1 - R)
            cy = _clamp(cy, by0 + R, by1 - R)
            geom = ShapelyPoint(cx, cy).buffer(R, resolution=32)
            it += 1
        shrink = 0
        while (geom.intersects(path_buffer) or _intersects_any(env, geom) or not _inside_bounds(env, geom)) and shrink < 6:
            R *= 0.88
            R = max(R, 0.08)
            cx = _clamp(cx, bx0 + R, bx1 - R)
            cy = _clamp(cy, by0 + R, by1 - R)
            geom = ShapelyPoint(cx, cy).buffer(R, resolution=32)
            shrink += 1
        if not (geom.intersects(path_buffer) or _intersects_any(env, geom) or not _inside_bounds(env, geom)):
            env.add_circle(cx, cy, R)

    def _poly_vertices(template: str, W: float, H: float) -> List[Tuple[float, float]]:
        t = 0.35 * min(W, H)
        if template == 'L':
            return [
                (-W/2, -H/2), (W/2, -H/2), (W/2, -H/2 + t), (-W/2 + t, -H/2 + t),
                (-W/2 + t, H/2), (-W/2, H/2)
            ]
        else:
            return [(-W/2, -H/2), (W/2, -H/2), (0.0, H/2)]

    def _rotate_points(pts: List[Tuple[float, float]], angle_deg: float) -> List[Tuple[float, float]]:
        th = np.deg2rad(float(angle_deg))
        c, s = float(np.cos(th)), float(np.sin(th))
        return [(c*x - s*y, s*x + c*y) for (x, y) in pts]

    def _translate_points(pts: List[Tuple[float, float]], dx: float, dy: float) -> List[Tuple[float, float]]:
        return [(x + dx, y + dy) for (x, y) in pts]

    def _place_polygon_frac(env: Environment, bx0: float, by0: float, bx1: float, by1: float, path_line: LineString, path_buffer, fx: float, fy: float, wf: float, hf: float, angle_deg: float, template: str = 'L', *, max_iter: int = 22) -> None:
        W, H = _dims_from_frac(bx0, by0, bx1, by1, wf, hf, min_size=0.22)
        cx = bx0 + float(fx) * (bx1 - bx0)
        cy = by0 + float(fy) * (by1 - by0)
        local = _poly_vertices(template, W, H)
        world = _translate_points(_rotate_points(local, angle_deg), cx, cy)
        from shapely.geometry import Polygon as ShapelyPolygon
        geom = ShapelyPolygon(world)
        def _clamp_center_inside(cx_: float, cy_: float, poly) -> Tuple[float, float, object]:
            x0, y0, x1, y1 = poly.bounds
            half_w = 0.5 * (x1 - x0)
            half_h = 0.5 * (y1 - y0)
            cx2 = _clamp(cx_, bx0 + half_w, bx1 - half_w)
            cy2 = _clamp(cy_, by0 + half_h, by1 - half_h)
            world2 = _translate_points(_rotate_points(local, angle_deg), cx2, cy2)
            return cx2, cy2, ShapelyPolygon(world2)
        cx, cy, geom = _clamp_center_inside(cx, cy, geom)
        step = max(0.02 * max(bx1 - bx0, by1 - by0), 0.10)
        it = 0
        while (geom.intersects(path_buffer) or _intersects_any(env, geom) or not _inside_bounds(env, geom)) and it < max_iter:
            ux, uy = _nearest_outward_dir(path_line, bx0, by0, bx1, by1, cx, cy, fx, fy)
            cx += ux * step
            cy += uy * step
            cx, cy, geom = _clamp_center_inside(cx, cy, geom)
            it += 1
        shrink = 0
        while (geom.intersects(path_buffer) or _intersects_any(env, geom) or not _inside_bounds(env, geom)) and shrink < 6:
            W *= 0.88
            H *= 0.88
            local = _poly_vertices(template, W, H)
            world = _translate_points(_rotate_points(local, angle_deg), cx, cy)
            geom = ShapelyPolygon(world)
            cx, cy, geom = _clamp_center_inside(cx, cy, geom)
            shrink += 1
        if not (geom.intersects(path_buffer) or _intersects_any(env, geom) or not _inside_bounds(env, geom)):
            env.add_polygon(list(geom.exterior.coords)[:-1])

    def _place_wall_frac(env: Environment, bx0: float, by0: float, bx1: float, by1: float, path_line: LineString, path_buffer, fx0: float, fy0: float, fx1: float, fy1: float, thick_frac: float, *, max_iter: int = 22) -> None:
        spanx, spany = _bounds_spans(bx0, by0, bx1, by1)
        t = max(0.06, float(thick_frac) * 0.10 * max(spanx, spany))
        x0 = bx0 + float(fx0) * (bx1 - bx0)
        y0 = by0 + float(fy0) * (by1 - by0)
        x1 = bx0 + float(fx1) * (bx1 - bx0)
        y1 = by0 + float(fy1) * (by1 - by0)
        from shapely.geometry import LineString as ShapelyLine
        seg = ShapelyLine([(x0, y0), (x1, y1)])
        geom = seg.buffer(0.5 * t, cap_style='flat', join_style='bevel')
        def _translate_wall(dx, dy):
            s2 = ShapelyLine([(x0 + dx, y0 + dy), (x1 + dx, y1 + dy)])
            return s2.buffer(0.5 * t, cap_style='flat', join_style='bevel')
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        step = max(0.02 * max(spanx, spany), 0.10)
        it = 0
        while (geom.intersects(path_buffer) or _intersects_any(env, geom) or not _inside_bounds(env, geom)) and it < max_iter:
            ux, uy = _nearest_outward_dir(path_line, bx0, by0, bx1, by1, cx, cy, 0.5*(fx0+fx1), 0.5*(fy0+fy1))
            cx += ux * step
            cy += uy * step
            dx = cx - 0.5 * (x0 + x1)
            dy = cy - 0.5 * (y0 + y1)
            geom = _translate_wall(dx, dy)
            it += 1
        shrink = 0
        while (geom.intersects(path_buffer) or _intersects_any(env, geom) or not _inside_bounds(env, geom)) and shrink < 6:
            t *= 0.88
            geom = seg.buffer(0.5 * t, cap_style='flat', join_style='bevel')
            shrink += 1
        if not (geom.intersects(path_buffer) or _intersects_any(env, geom) or not _inside_bounds(env, geom)):
            env.add_wall(x0, y0, x1, y1, thickness=t)

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

        # Sostituisco la logica che inseriva rettangoli con forme miste per traiettoria
        if idx == 0:  # Rettilinea (v costante)
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.22, 0.28, 0.06)
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.58, 0.70, 0.16, 0.12, 15.0, 'triangle')
            _place_wall_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.78, 0.30, 0.92, 0.46, 0.04)
        elif idx == 1:  # Rettilinea (v variabile)
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.18, 0.70, 0.18, 0.12, -20.0, 'L')
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.46, 0.26, 0.05)
            _place_wall_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.70, 0.56, 0.82, 0.62, 0.03)
        elif idx == 2:  # Circolare (v costante)
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.14, 0.54, 0.06)
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.50, 0.14, 0.18, 0.12, 30.0, 'triangle')
            _place_wall_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.84, 0.60, 0.92, 0.74, 0.04)
            # Nuovo ostacolo esterno al cerchio
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.08, 0.90, 0.04)
        elif idx == 3:  # Circolare (v variabile)
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.22, 0.20, 0.16, 0.12, -35.0, 'L')
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.60, 0.84, 0.05)
            _place_wall_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.84, 0.34, 0.94, 0.38, 0.03)
            # Nuovo ostacolo esterno al cerchio
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.92, 0.10, 0.04)
        elif idx == 4:  # Traiettoria a 8
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.18, 0.44, 0.16, 0.18, 10.0, 'L')
            _place_wall_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.46, 0.22, 0.64, 0.22, 0.05)
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.82, 0.56, 0.05)
            # Nuovo ostacolo in alto: triangolo compatto nella parte superiore dei bounds
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.70, 0.88, 0.14, 0.12, 5.0, 'triangle')
        else:  # Random walk
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.20, 0.24, 0.05)
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.50, 0.72, 0.14, 0.16, -12.0, 'triangle')
            # Sposto il muro rettangolare lungo più in alto a destra per evitare sovrapposizione con la traiettoria
            _place_wall_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.88, 0.80, 0.96, 0.88, 0.04)

        envs.append(env)

    return envs
