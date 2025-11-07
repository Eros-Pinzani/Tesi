# Preset e utility per creare e configurare Environment sulla base delle traiettorie

from typing import List, Tuple
import numpy as np
from environment import Environment
from shapely.geometry import LineString, Point
from shapely.geometry import Polygon as ShapelyPolygon


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
        geom = ShapelyPolygon(world)
        def _clamp_center_inside(cx_: float, cy_: float, poly: ShapelyPolygon) -> Tuple[float, float, ShapelyPolygon]:
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

        # Sostituisco la logica di posizionamento SOLO per il caso rettilineo v costante (idx==0)
        if idx == 0:
            # Direzione globale della traiettoria (tangente) e normale
            try:
                length = float(path_line.length)
            except Exception:
                length = 0.0
            # Fallback: usa delta tra primo e ultimo punto
            if length <= 1e-6:
                xs = hist[:, 0]; ys = hist[:, 1]
                p0 = np.array([float(xs[0]), float(ys[0])], dtype=float)
                p1 = np.array([float(xs[-1]), float(ys[-1])], dtype=float)
                v = p1 - p0
                vn = float(np.hypot(v[0], v[1])) or 1.0
                t_hat = v / vn
                n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
                # Centro percorso
                cx = 0.5 * (float(np.min(xs)) + float(np.max(xs)))
                cy = 0.5 * (float(np.min(ys)) + float(np.max(ys)))
                def _interp_point(alpha: float) -> np.ndarray:
                    return np.array([cx, cy], dtype=float) + (alpha - 0.5) * vn * t_hat
            else:
                def _as_np(pt):
                    return np.array([float(pt.x), float(pt.y)], dtype=float)
                p0 = _as_np(path_line.interpolate(0.0))
                p1 = _as_np(path_line.interpolate(length))
                v = p1 - p0
                vn = float(np.hypot(v[0], v[1])) or 1.0
                t_hat = v / vn
                n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
                def _interp_point(alpha: float) -> np.ndarray:
                    s = float(np.clip(alpha, 0.0, 1.0)) * length
                    pt = path_line.interpolate(s)
                    return np.array([float(pt.x), float(pt.y)], dtype=float)

            # Offset laterale: appena oltre il corridoio di sicurezza
            span = max(bx1 - bx0, by1 - by0)
            d_off = float(clearance + 0.06 * span)  # prima 0.10*span: avvicina gli ostacoli per entrare in r_max
            safe_margin = 0.02 * float(span)

            # Clamp punto ai bounds
            def _clamp_pt(pt: np.ndarray) -> np.ndarray:
                return np.array([
                    float(np.clip(pt[0], bx0 + safe_margin, bx1 - safe_margin)),
                    float(np.clip(pt[1], by0 + safe_margin, by1 - safe_margin))
                ], dtype=float)

            # Helper: aggiungi un cerchio riducendo il raggio finché sta nei bounds e non tocca il buffer
            def _add_circle_safe(cx: float, cy: float, r_des: float) -> None:
                from shapely.geometry import Point as ShapelyPoint
                cx = float(np.clip(cx, bx0 + safe_margin, bx1 - safe_margin))
                cy = float(np.clip(cy, by0 + safe_margin, by1 - safe_margin))
                # r massimo consentito dai bounds (meno margine)
                r_max_bounds = float(min(cx - bx0, bx1 - cx, cy - by0, by1 - cy) - safe_margin)
                r = max(0.01 * span, min(r_des, r_max_bounds))
                # Riduci se interseca il corridoio o supera bounds
                it = 0
                while it < 12:
                    geom = ShapelyPoint(cx, cy).buffer(r, resolution=32)
                    inside = True
                    try:
                        inside = env.bounds.contains(geom)  # type: ignore[union-attr]
                    except Exception:
                        inside = True
                    if inside and (not geom.intersects(path_buffer)):
                        env.add_circle(cx, cy, r)
                        return
                    r *= 0.86
                    if r < 0.02 * span:
                        break
                    it += 1
                # Se fallisce, non aggiunge il cerchio
                return

            # Helper: aggiungi un muro corto perpendicolare riducendo spessore e lunghezza se serve
            def _add_wall_safe(a: np.ndarray, b: np.ndarray, t_des: float) -> None:
                from shapely.geometry import LineString as ShapelyLine
                a = _clamp_pt(a); b = _clamp_pt(b)
                L = float(np.linalg.norm(b - a))
                if L < 1e-6:
                    return
                t = float(t_des)
                scale = 1.0
                it = 0
                while it < 14:
                    aa = a + 0.5 * (1.0 - scale) * (b - a)
                    bb = b - 0.5 * (1.0 - scale) * (b - a)
                    seg = ShapelyLine([(float(aa[0]), float(aa[1])), (float(bb[0]), float(bb[1]))])
                    geom = seg.buffer(0.5 * t, cap_style='flat', join_style='bevel')
                    inside = True
                    try:
                        inside = env.bounds.contains(geom)  # type: ignore[union-attr]
                    except Exception:
                        inside = True
                    if inside and (not geom.intersects(path_buffer)):
                        env.add_wall(float(aa[0]), float(aa[1]), float(bb[0]), float(bb[1]), thickness=float(t))
                        return
                    # Shrink progressivo
                    if (it % 2) == 0:
                        t *= 0.85
                    else:
                        scale *= 0.88
                    if t < 0.015 * span or scale < 0.40:
                        break
                    it += 1
                return

            # Helper: aggiungi un rettangolo ruotato in modo safe
            def _add_rot_rect_safe(cx: float, cy: float, w_des: float, h_des: float, angle_deg: float) -> None:
                from shapely.geometry import Polygon as _Poly
                cx = float(np.clip(cx, bx0 + safe_margin, bx1 - safe_margin))
                cy = float(np.clip(cy, by0 + safe_margin, by1 - safe_margin))
                w = float(w_des); h = float(h_des)
                it = 0
                while it < 14 and w > 0.02*span and h > 0.02*span:
                    local = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
                    world = _translate_points(_rotate_points(local, angle_deg), cx, cy)
                    geom = _Poly(world)
                    inside = True
                    try:
                        inside = env.bounds.contains(geom)  # type: ignore[union-attr]
                    except Exception:
                        inside = True
                    if inside and (not geom.intersects(path_buffer)) and (not _intersects_any(env, geom)):
                        env.add_polygon(world)
                        return
                    w *= 0.88
                    h *= 0.88
                    it += 1
                return

            # Helper: aggiungi un triangolo in modo safe
            def _add_triangle_safe(cx: float, cy: float, w_des: float, h_des: float, angle_deg: float) -> None:
                from shapely.geometry import Polygon as _Poly
                cx = float(np.clip(cx, bx0 + safe_margin, bx1 - safe_margin))
                cy = float(np.clip(cy, by0 + safe_margin, by1 - safe_margin))
                W = float(w_des); H = float(h_des)
                it = 0
                while it < 14 and W > 0.02*span and H > 0.02*span:
                    local = _poly_vertices('triangle', W, H)
                    world = _translate_points(_rotate_points(local, angle_deg), cx, cy)
                    geom = _Poly(world)
                    inside = True
                    try:
                        inside = env.bounds.contains(geom)  # type: ignore[union-attr]
                    except Exception:
                        inside = True
                    if inside and (not geom.intersects(path_buffer)) and (not _intersects_any(env, geom)):
                        env.add_polygon(world)
                        return
                    W *= 0.88
                    H *= 0.88
                    it += 1
                return

            # Due cerchi laterali a posizioni diverse lungo il percorso
            c0 = _interp_point(0.12)
            c1 = _interp_point(0.35)
            c2 = _interp_point(0.65)
            # Ostacolo extra vicino all'inizio, lato destro (sotto la retta se y cresce verso l'alto)
            d0 = float(clearance + 0.04 * span)
            c0R = _clamp_pt(c0 - d0 * n_hat)
            r0 = float(min(0.65 * clearance, 0.05 * span))
            _add_circle_safe(c0R[0], c0R[1], r0)

            c1L = _clamp_pt(c1 + d_off * n_hat)   # lato sinistro
            c2R = _clamp_pt(c2 - d_off * n_hat)   # lato destro
            r1 = float(min(0.75 * clearance, 0.06 * span))
            r2 = float(min(0.75 * clearance, 0.06 * span))
            _add_circle_safe(c1L[0], c1L[1], r1)
            _add_circle_safe(c2R[0], c2R[1], r2)

            # Un muro corto perpendicolare su un lato, per vincolare ulteriormente la rotazione
            c3 = _interp_point(0.50) + 1.10 * d_off * n_hat
            L = float(max(0.40, 0.18 * span))
            t = float(max(0.04, 0.02 * span))
            a = _clamp_pt(c3 - 0.5 * L * n_hat)
            b = _clamp_pt(c3 + 0.5 * L * n_hat)
            _add_wall_safe(a, b, t)

            # Aggiunta ostacoli NON circolari extra: triangolo (lato sinistro) e rettangolo ruotato (lato destro)
            cT = _interp_point(0.22) + 0.90 * d_off * n_hat
            _add_triangle_safe(cT[0], cT[1], 0.10 * span, 0.12 * span, angle_deg=15.0)

            cR = _interp_point(0.80) - 0.90 * d_off * n_hat
            _add_rot_rect_safe(cR[0], cR[1], 0.18 * span, 0.08 * span, angle_deg=-20.0)

            # Controparti simmetriche per bilanciamento
            cT_sym = _interp_point(0.22) - 0.90 * d_off * n_hat  # triangolo lato destro
            _add_triangle_safe(cT_sym[0], cT_sym[1], 0.10 * span, 0.12 * span, angle_deg=-15.0)

            cL_sym = _interp_point(0.80) + 0.90 * d_off * n_hat  # rettangolo ruotato lato sinistro
            _add_rot_rect_safe(cL_sym[0], cL_sym[1], 0.18 * span, 0.08 * span, angle_deg=20.0)

            # Muro perpendicolare speculare sull'altro lato
            c3_sym = _interp_point(0.50) - 1.10 * d_off * n_hat
            a_sym = _clamp_pt(c3_sym - 0.5 * L * n_hat)
            b_sym = _clamp_pt(c3_sym + 0.5 * L * n_hat)
            _add_wall_safe(a_sym, b_sym, t)
        elif idx == 1:  # Rettilinea (v variabile)
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.18, 0.70, 0.18, 0.12, -20.0, 'L')
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.46, 0.26, 0.05)
            # Sposta il muro rettangolare a destra più in alto per evitare collisioni con la traiettoria
            _place_wall_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.70, 0.74, 0.82, 0.80, 0.03)
            # --- Nuovi ostacoli per spezzare ambiguita' traslazionale ---
            # Triangolo compatto lato opposto al primo L-shape (basso-destra)
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.82, 0.24, 0.14, 0.10, 12.0, 'triangle')
            # Muro diagonale corto per fornire riferimento angolare
            _place_wall_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.30, 0.18, 0.38, 0.32, 0.025)
            # Piccolo cerchio lontano (alto-centro) per parallax a distanza diversa
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.52, 0.90, 0.035)
        elif idx == 2:  # Circolare (v costante)
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.14, 0.54, 0.06)
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.50, 0.14, 0.18, 0.12, 30.0, 'triangle')
            _place_wall_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.84, 0.60, 0.92, 0.74, 0.04)
            # Nuovo ostacolo esterno al cerchio (alto a sinistra)
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.08, 0.90, 0.04)
            # Nuovo ostacolo in alto a destra (triangolo compatto)
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.90, 0.90, 0.12, 0.10, -10.0, 'triangle')
        elif idx == 3:  # Circolare (v variabile)
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.22, 0.20, 0.16, 0.12, -35.0, 'L')
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.60, 0.84, 0.05)
            _place_wall_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.84, 0.34, 0.94, 0.38, 0.03)
            # Nuovo ostacolo esterno al cerchio (in basso a destra)
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.92, 0.10, 0.04)
            # Nuovo ostacolo in alto a destra (triangolo compatto)
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.88, 0.90, 0.12, 0.10, 8.0, 'triangle')
            # EXTRA: L-shape in alto-sinistra e muro diagonale corto per aumentare vincolo angolare
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.12, 0.82, 0.14, 0.12, 25.0, 'L')
            _place_wall_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.18, 0.30, 0.26, 0.18, 0.025)
        elif idx == 4:  # Traiettoria a 8
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.18, 0.44, 0.16, 0.18, 10.0, 'L')
            _place_wall_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.46, 0.22, 0.64, 0.22, 0.05)
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.82, 0.56, 0.05)
            # Nuovo ostacolo in alto: triangolo compatto nella parte superiore dei bounds
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.70, 0.88, 0.14, 0.12, 5.0, 'triangle')
            # EXTRA: muro diagonale in basso-sinistra, triangolo in basso-destra e piccolo cerchio lontano
            _place_wall_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.18, 0.18, 0.28, 0.30, 0.025)
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.88, 0.18, 0.12, 0.10, -18.0, 'triangle')
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.52, 0.92, 0.035)
        else:  # Random walk
            _place_circle_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.20, 0.24, 0.05)
            _place_polygon_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.50, 0.72, 0.14, 0.16, -12.0, 'triangle')
            # Sposto il muro rettangolare lungo più in alto a destra per evitare sovrapposizione con la traiettoria
            _place_wall_frac(env, bx0, by0, bx1, by1, path_line, path_buffer, 0.88, 0.80, 0.96, 0.88, 0.04)

        envs.append(env)

    return envs
