"""Algoritmo ICP 2D (point-to-point) + utility per eseguirlo su scansioni LiDAR consecutive.

- Lavora in frame ROBOT (locale) come richiesto.
- Esegue due varianti: init_pose=None (nessuna inizializzazione) e init_pose da odometria (stima relativa fra pose consecutive).
- Non richiede SciPy; se presente, usa cKDTree per i nearest neighbors, altrimenti fallback O(N*M).
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple

# Prova a importare KDTree da SciPy in modo sicuro e definisci un builder
try:
    from scipy.spatial import cKDTree as _SciPyKDTree  # type: ignore
    _HAS_KDTREE = True
    def _build_kdtree(data: np.ndarray):
        return _SciPyKDTree(np.asarray(data, dtype=float))
except ImportError:
    _HAS_KDTREE = False
    def _build_kdtree(_data: np.ndarray):  # parametro prefissato con underscore per evitare warning unused
        raise RuntimeError("SciPy KDTree non disponibile")

# --------------------------- Algebra di base ---------------------------

def rot2d(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=float)


def pose_to_r_t(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converte una posa [x, y, theta] in (r 2x2, t 2,) con nome funzione minuscolo."""
    x, y, th = map(float, pose)
    return rot2d(th), np.array([x, y], dtype=float)


def relative_local_transform(prev_pose: np.ndarray, curr_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Trasformazione che porta punti espressi nel frame del robot a tempo k (curr)
    nel frame del robot a tempo k-1 (prev): p_{k-1} = R_rel @ p_k + t_rel.

    Derivazione:
      p_w = Rk p_k + tk  ;  p_{k-1} = R_{k-1}^T (p_w - t_{k-1})
      => p_{k-1} = (R_{k-1}^T Rk) p_k + R_{k-1}^T (tk - t_{k-1})
    """
    r_prev, t_prev = pose_to_r_t(prev_pose)
    r_curr, t_curr = pose_to_r_t(curr_pose)
    r_rel = r_prev.T @ r_curr
    t_rel = r_prev.T @ (t_curr - t_prev)
    return r_rel, t_rel


# --------------------------- Utility campionamento ---------------------------

def _angle_uniform_subsample(points: np.ndarray, bin_deg: float = 10.0, max_per_bin: int = 12, prefer_far: bool = True) -> np.ndarray:
    """Sotto-campiona i punti in modo quasi uniforme in angolo (alpha=atan2(y,x) in frame locale).
    - bin_deg: ampiezza del bin angolare in gradi
    - max_per_bin: massimo numero di punti per bin
    - prefer_far: se True tiene i punti piu' lontani (maggior parallasse)
    """
    if points is None or len(points) <= 0:
        return points
    pts = np.asarray(points, dtype=float)
    ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    ang = (ang + 360.0) % 360.0
    r = np.hypot(pts[:, 0], pts[:, 1])
    bin_w = max(1e-6, float(bin_deg))
    idx_bins = (ang // bin_w).astype(int)
    n_bins = int(np.ceil(360.0 / bin_w))
    keep_idx: List[int] = []
    for b in range(n_bins):
        sel = np.where(idx_bins == b)[0]
        if sel.size == 0:
            continue
        if prefer_far:
            order = np.argsort(r[sel])[::-1]  # dal piu' lontano
        else:
            order = np.argsort(r[sel])        # dal piu' vicino
        take = sel[order[:max(1, int(max_per_bin))]]
        keep_idx.extend(take.tolist())
    if not keep_idx:
        return pts
    keep_idx = sorted(set(keep_idx))
    return pts[keep_idx]

# --------------------------- Mattoncini ICP ---------------------------

def nearest_neighbors(src: np.ndarray, dst: np.ndarray, max_distance: Optional[float] = None, use_scipy: bool = True):
    """Per ogni punto src, trova il nearest neighbor in dst. Ritorna (idxs, dists, mask_inliers)."""
    if use_scipy and _HAS_KDTREE:
        try:
            tree = _build_kdtree(dst)
            dists, idxs = tree.query(np.asarray(src, dtype=float), k=1)
            mask = (dists <= float(max_distance)) if (max_distance is not None) else np.ones_like(dists, dtype=bool)
            return idxs, dists, mask
        except (ValueError, TypeError, RuntimeError):
            # Fallback sicuro al brute-force in caso di input malformati o problemi runtime della KDTree
            pass
    # Fallback O(N*M)
    n_pts = src.shape[0]
    idxs = np.empty(n_pts, dtype=int)
    dists = np.empty(n_pts, dtype=float)
    mask = np.ones(n_pts, dtype=bool)
    for i in range(n_pts):
        dif = dst - src[i]
        dist2 = np.sum(dif * dif, axis=1)
        j = int(np.argmin(dist2))
        d = float(np.sqrt(dist2[j]))
        idxs[i] = j
        dists[i] = d
        if (max_distance is not None) and (d > float(max_distance)):
            mask[i] = False
    return idxs, dists, mask


def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    *,
    init_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
    max_correspondence_distance: Optional[float] = None,
    trim_fraction: Optional[float] = None,
    use_scipy: bool = True,
    verbose: bool = False,
    damping_enabled: bool = True,
    angle_thresh_deg: float = 7.5,
    struct_ratio_thresh: float = 0.03,
    damp_factor: float = 0.5,
    sliding_filter_enabled: bool = True,
    sliding_cos_threshold: float = 0.985,
    min_after_sliding: int = 6,
    angle_balance_enabled: bool = True,
    angle_bin_deg: float = 10.0,
    angle_max_per_bin: int = 12,
    angle_prefer_far: bool = True,
    robust_enabled: bool = True,
    huber_c_factor: float = 1.5,
    dynamic_maxdist: bool = True,
    dynamic_factor: float = 2.0,
    dynamic_min: float = 0.20,
    dynamic_max: float = 0.50,
    stop_if_init_good: bool = True,
    init_rmse_threshold: float = 0.10,
    adaptive_filters: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict], np.ndarray]:
    """ICP point-to-point 2D con filtri (sliding, robust, damping) + bilanciamento angolare opzionale.

    Bilanciamento angolare: se angle_balance_enabled True, prima della stima i punti source/target
    vengono sotto-campionati con _angle_uniform_subsample usando i parametri angle_bin_deg, angle_max_per_bin,
    angle_prefer_far.

    Ritorna: (R_finale, t_finale, source_trasformata, history, errors)
    - history: lista di dict {'iter','R','t','mean_error','n_corr'}
    - errors: array RMSE per iterazione.
    """
    assert source.ndim == 2 and source.shape[1] == 2 and target.ndim == 2 and target.shape[1] == 2
    src = np.asarray(source, dtype=float).copy()
    dst = np.asarray(target, dtype=float).copy()
    if angle_balance_enabled:
        src = _angle_uniform_subsample(src, bin_deg=angle_bin_deg, max_per_bin=angle_max_per_bin, prefer_far=angle_prefer_far)
        dst = _angle_uniform_subsample(dst, bin_deg=angle_bin_deg, max_per_bin=angle_max_per_bin, prefer_far=angle_prefer_far)
        # Se il bilanciamento elimina troppi punti, uscita anticipata (nessuna iterazione)
        if len(src) < 3 or len(dst) < 3:
            return np.eye(2), np.zeros(2), src, [], np.array([], dtype=float)

    total_r = np.eye(2)
    total_t = np.zeros(2)
    if init_pose is not None:
        r0, t0 = init_pose
        src = (r0 @ src.T).T + t0.reshape(1, 2)
        total_r = r0 @ total_r
        total_t = r0 @ total_t + t0
        # Valuta se l'init è già buono e fermati subito
        if stop_if_init_good:
            try:
                _idx0, d0, m0 = nearest_neighbors(src, dst, max_correspondence_distance, use_scipy=use_scipy)
                if m0.any():
                    rmse0 = float(np.sqrt(np.mean((d0[m0])**2)))
                    if rmse0 <= float(init_rmse_threshold) and int(m0.sum()) >= int(min_after_sliding):
                        source_transformed = (total_r @ np.asarray(source).T).T + total_t.reshape(1, 2)
                        return total_r, total_t, source_transformed, [], np.array([rmse0], dtype=float)
            except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError):
                pass

    prev_error: Optional[float] = None
    history: List[Dict] = []

    adaptive_sliding = sliding_filter_enabled
    adaptive_trim = trim_fraction

    for it in range(int(max_iterations)):
        idxs, dists, mask = nearest_neighbors(src, dst, max_correspondence_distance, use_scipy=use_scipy)
        # Adattività: se poche corrispondenze disabilita sliding/trim per non perdere informazione
        if adaptive_filters and mask.sum() < (2 * min_after_sliding):
            adaptive_sliding = False
            adaptive_trim = None

        # Soglia dinamica per corrispondenze (stringe il max_distance effettivo in base alla mediana)
        if dynamic_maxdist and mask.any():
            vd = dists[mask]
            if vd.size >= 6:
                med = float(np.median(vd))
                thr_dyn = max(dynamic_min, min(dynamic_max, dynamic_factor * med))
                mask = mask & (dists <= thr_dyn)

        # Trimming corrispondenze più lontane
        if mask.any() and adaptive_trim is not None and 0.0 < float(adaptive_trim) < 1.0:
            valid_d = dists[mask]
            if len(valid_d) >= 6:
                thr_trim = float(np.quantile(valid_d, float(adaptive_trim)))
                mask = mask & (dists <= thr_trim)

        # Filtro sliding (solo se abilitato e abbastanza corrispondenze)
        if adaptive_sliding and mask.sum() >= min_after_sliding:
            # PCA sulla nuvola target accoppiata (direzione principale)
            paired_dst = dst[idxs[mask]]
            centered = paired_dst - paired_dst.mean(axis=0)
            try:
                cov = centered.T @ centered
                eigvals, eigvecs = np.linalg.eigh(cov)
                principal = eigvecs[:, np.argsort(eigvals)[::-1][0]]
                principal /= np.linalg.norm(principal) + 1e-12
                # Per ogni corrispondenza calcola vettore differenza (prima della stima) e valuta allineamento
                diffs = paired_dst - src[mask]
                norms = np.linalg.norm(diffs, axis=1) + 1e-12
                dirs = diffs / norms[:, None]
                cos_vals = np.abs(dirs @ principal)
                sliding_mask = cos_vals >= float(sliding_cos_threshold)
                # Evita di rimuovere tutto: mantieni almeno min_after_sliding
                if sliding_mask.sum() > 0 and (mask.sum() - sliding_mask.sum()) >= min_after_sliding:
                    # Aggiorna mask globale: crea copia booleana su lunghezza totale
                    full_mask = mask.copy()
                    # Indici globali dei validi
                    global_valid_idx = np.where(mask)[0]
                    # Quelli da tenere sono dove sliding_mask è False
                    to_drop_global = global_valid_idx[sliding_mask]
                    full_mask[to_drop_global] = False
                    mask = full_mask
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                # In caso di problemi numerici nella PCA, salta il filtro senza sopprimere altri errori
                pass

        valid_src = src[mask]
        valid_dst = dst[idxs[mask]]
        if len(valid_src) < 3:
            if verbose:
                print(f"[ICP] Iter {it}: corrispondenze insufficienti ({len(valid_src)}). Stop.")
            break

        # Pesi robusti Huber
        weights = None
        if robust_enabled:
            vd = dists[mask]
            if vd.size >= 3:
                med = float(np.median(vd))
                if med > 1e-12:
                    c = huber_c_factor * med
                    w = np.ones_like(vd)
                    big = vd > c
                    w[big] = c / (vd[big] + 1e-12)
                    weights = w

        # Calcola trasformazione ottima via SVD (come best_fit_transform_2d), con possibilità di damping
        if weights is None:
            centroid_a = valid_src.mean(axis=0)
            centroid_b = valid_dst.mean(axis=0)
            aa = valid_src - centroid_a
            bb = valid_dst - centroid_b
            h = aa.T @ bb
        else:
            w = weights.reshape(-1, 1)
            w_sum = float(np.sum(w)) or 1.0  # rinominato da Wsum
            centroid_a = (valid_src * w).sum(axis=0) / w_sum
            centroid_b = (valid_dst * w).sum(axis=0) / w_sum
            aa = valid_src - centroid_a
            bb = valid_dst - centroid_b
            h = aa.T @ (bb * w)
        u, s, vt = np.linalg.svd(h)
        r_delta_raw = vt.T @ u.T
        if np.linalg.det(r_delta_raw) < 0:
            vt[1, :] *= -1
            r_delta_raw = vt.T @ u.T
        t_delta_raw = centroid_b - r_delta_raw @ centroid_a

        # Damping opzionale
        r_delta = r_delta_raw  # rinominato da R_delta
        t_delta = t_delta_raw
        if damping_enabled:
            s_max = float(np.max(s)) if s.size > 0 else 1.0
            s_min = float(np.min(s)) if s.size > 0 else 1.0
            struct_ratio = (s_min / s_max) if s_max > 1e-12 else 1.0
            angle_raw = float(np.arctan2(r_delta_raw[1, 0], r_delta_raw[0, 0]))
            angle_thresh = float(np.deg2rad(angle_thresh_deg))
            if struct_ratio < float(struct_ratio_thresh) and abs(angle_raw) > angle_thresh:
                r = struct_ratio / float(struct_ratio_thresh)
                r = max(0.0, min(1.0, r))
                dyn_factor = float(damp_factor) + (1.0 - float(damp_factor)) * r
                angle_new = angle_raw * dyn_factor
                r_delta = rot2d(angle_new)
                t_delta = centroid_b - r_delta @ centroid_a

        # Aggiorna i punti src e accumula la trasformazione
        src = (r_delta @ src.T).T + t_delta.reshape(1, 2)
        total_t = r_delta @ total_t + t_delta
        total_r = r_delta @ total_r

        d_valid = dists[mask]
        rmse = float(np.sqrt(np.mean(d_valid * d_valid))) if d_valid.size > 0 else float('inf')
        history.append({'iter': it, 'R': r_delta, 't': t_delta, 'mean_error': rmse, 'n_corr': int(mask.sum())})
        if verbose:
            # Calcolo sicuro del struct_ratio per logging
            if s.size > 0:
                max_s = float(np.max(s))
                min_s = float(np.min(s))
                if np.isfinite(max_s) and max_s > 1e-12 and np.isfinite(min_s):
                    sr = min_s / max_s
                else:
                    sr = 1.0
            else:
                sr = 1.0
            print(f"[ICP] Iter {it}: rmse={rmse:.6f}, n_corr={int(mask.sum())}, struct_ratio={sr:.3f}")
        if prev_error is not None and abs(prev_error - rmse) < float(tolerance):
            if verbose:
                print(f"[ICP] Converged at iter {it} (Δrmse < tol).")
            break
        prev_error = rmse

    source_transformed = (total_r @ np.asarray(source).T).T + total_t.reshape(1, 2)
    errors = np.array([h['mean_error'] for h in history], dtype=float)
    return total_r, total_t, source_transformed, history, errors


# --------------------------- Runner su history ---------------------------

def run_icp_pair_local(
    lidar,
    env,
    prev_pose: np.ndarray,
    curr_pose: np.ndarray,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
    max_correspondence_distance: Optional[float] = None,
    use_scipy: bool = True,
    trim_fraction: Optional[float] = None,
    damping_enabled: bool = True,
    angle_thresh_deg: float = 7.5,
    struct_ratio_thresh: float = 0.03,
    damp_factor: float = 0.5,
    sliding_filter_enabled: bool = True,
    sliding_cos_threshold: float = 0.985,
    angle_balance_enabled: bool = True,
    angle_bin_deg: float = 10.0,
    angle_max_per_bin: int = 12,
    angle_prefer_far: bool = True,
    robust_enabled: bool = True,
    huber_c_factor: float = 1.5,
    dynamic_maxdist: bool = True,
    dynamic_factor: float = 2.0,
    dynamic_min: float = 0.20,
    dynamic_max: float = 0.50,
) -> Dict:
    """Esegue ICP tra le scansioni consecutive con filtro sliding opzionale.

    Esegue due run: (A) senza init e (B) con init da odometria (relativa curr->prev nel frame locale).
    Ritorna un dizionario con risultati e metriche per entrambe le varianti.
    """
    # Estrai punti di impatto nel frame locale del sensore
    tgt_local = lidar.scan_hits(prev_pose, env, frame='local')  # target = k-1
    src_local = lidar.scan_hits(curr_pose, env, frame='local')  # source = k
    if tgt_local is None or src_local is None or len(tgt_local) < 3 or len(src_local) < 3:
        return {
            'ok': False,
            'reason': 'not_enough_points',
            'n_src': 0 if src_local is None else int(len(src_local)),
            'n_tgt': 0 if tgt_local is None else int(len(tgt_local)),
        }

    # (A) ICP senza inizializzazione
    r_none, t_none, src_tf_none, hist_none, errs_none = icp_point_to_point(
        src_local, tgt_local,
        init_pose=None,
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_distance=max_correspondence_distance,
        trim_fraction=trim_fraction,
        use_scipy=use_scipy,
        verbose=False,
        damping_enabled=damping_enabled,
        angle_thresh_deg=angle_thresh_deg,
        struct_ratio_thresh=struct_ratio_thresh,
        damp_factor=damp_factor,
        sliding_filter_enabled=sliding_filter_enabled,
        sliding_cos_threshold=sliding_cos_threshold,
        angle_balance_enabled=angle_balance_enabled,
        angle_bin_deg=angle_bin_deg,
        angle_max_per_bin=angle_max_per_bin,
        angle_prefer_far=angle_prefer_far,
        robust_enabled=robust_enabled,
        huber_c_factor=huber_c_factor,
        dynamic_maxdist=dynamic_maxdist,
        dynamic_factor=dynamic_factor,
        dynamic_min=dynamic_min,
        dynamic_max=dynamic_max,
    )
    # (B) ICP con inizializzazione odometrica
    r0, t0 = relative_local_transform(prev_pose, curr_pose)
    r_odo, t_odo, src_tf_odo, hist_odo, errs_odo = icp_point_to_point(
        src_local, tgt_local,
        init_pose=(r0, t0),
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_distance=max_correspondence_distance,
        trim_fraction=trim_fraction,
        use_scipy=use_scipy,
        verbose=False,
        damping_enabled=damping_enabled,
        angle_thresh_deg=angle_thresh_deg,
        struct_ratio_thresh=struct_ratio_thresh,
        damp_factor=damp_factor,
        sliding_filter_enabled=sliding_filter_enabled,
        sliding_cos_threshold=sliding_cos_threshold,
        angle_balance_enabled=angle_balance_enabled,
        angle_bin_deg=angle_bin_deg,
        angle_max_per_bin=angle_max_per_bin,
        angle_prefer_far=angle_prefer_far,
        robust_enabled=robust_enabled,
        huber_c_factor=huber_c_factor,
        dynamic_maxdist=dynamic_maxdist,
        dynamic_factor=dynamic_factor,
        dynamic_min=dynamic_min,
        dynamic_max=dynamic_max,
    )
    # (C) ICP RAW
    r_raw_none, t_raw_none, src_tf_raw_none, hist_raw_none, errs_raw_none = icp_point_to_point(
        src_local, tgt_local,
        init_pose=None,
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_distance=None,
        trim_fraction=None,
        use_scipy=use_scipy,
        verbose=False,
        damping_enabled=False,
        sliding_filter_enabled=False,
        sliding_cos_threshold=sliding_cos_threshold,
        angle_balance_enabled=False,
        robust_enabled=False,
        dynamic_maxdist=False,
    )
    r_raw_odo, t_raw_odo, src_tf_raw_odo, hist_raw_odo, errs_raw_odo = icp_point_to_point(
        src_local, tgt_local,
        init_pose=(r0, t0),
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_distance=None,
        trim_fraction=None,
        use_scipy=use_scipy,
        verbose=False,
        damping_enabled=False,
        sliding_filter_enabled=False,
        sliding_cos_threshold=sliding_cos_threshold,
        angle_balance_enabled=False,
        robust_enabled=False,
        dynamic_maxdist=False,
    )
    def _theta_from_r(r_mat: np.ndarray) -> float:
        return float(np.arctan2(r_mat[1, 0], r_mat[0, 0]))
    def _deg(rad: float) -> float:
        return float(rad * 180.0 / np.pi)
    def _pose_errors(r_rel: np.ndarray, t_rel: np.ndarray) -> Tuple[float, float]:
        x_prev, y_prev, th_prev = map(float, prev_pose)
        x_curr, y_curr, th_curr = map(float, curr_pose)
        ang_rel = float(np.arctan2(r_rel[1, 0], r_rel[0, 0]))
        dx_w = np.cos(th_prev) * t_rel[0] - np.sin(th_prev) * t_rel[1]
        dy_w = np.sin(th_prev) * t_rel[0] + np.cos(th_prev) * t_rel[1]
        x_pred = x_prev + dx_w
        y_pred = y_prev + dy_w
        th_pred = th_prev + ang_rel
        th_pred = (th_pred + np.pi) % (2 * np.pi) - np.pi
        th_curr_wrapped = (th_curr + np.pi) % (2 * np.pi) - np.pi
        trans_err = float(np.hypot(x_pred - x_curr, y_pred - y_curr))
        rot_err = th_pred - th_curr_wrapped
        rot_err = (rot_err + np.pi) % (2 * np.pi) - np.pi
        return trans_err, abs(float(np.degrees(rot_err)))
    none_trans_err, none_rot_err = _pose_errors(r_none, t_none)
    odo_trans_err, odo_rot_err = _pose_errors(r_odo, t_odo)
    raw_none_trans_err, raw_none_rot_err = _pose_errors(r_raw_none, t_raw_none)
    raw_odo_trans_err, raw_odo_rot_err = _pose_errors(r_raw_odo, t_raw_odo)
    best_key = min([
        ('none', none_trans_err, r_none, t_none),
        ('odo', odo_trans_err, r_odo, t_odo),
        ('raw_none', raw_none_trans_err, r_raw_none, t_raw_none),
        ('raw_odo', raw_odo_trans_err, r_raw_odo, t_raw_odo)
    ], key=lambda x: x[1])
    best_r = best_key[2]; best_t = best_key[3]
    best_trans_err, best_rot_err = _pose_errors(best_r, best_t)

    out = {
        'ok': True,
        'n_src': int(len(src_local)),
        'n_tgt': int(len(tgt_local)),
        'gt_R': r0, 'gt_t': t0,
        'src_local': src_local, 'tgt_local': tgt_local,
        'none': {
            'R': r_none, 't': t_none,
            'alpha_rad': _theta_from_r(r_none),
            'alpha_deg': _deg(_theta_from_r(r_none)),
            'rmse': float(errs_none[-1]) if errs_none.size > 0 else float('inf'),
            'iterations': int(len(hist_none)),
            'n_corr_last': int(hist_none[-1]['n_corr']) if len(hist_none) > 0 else 0,
            'pose_err_trans': none_trans_err,
            'pose_err_rot_deg': none_rot_err,
            'errs': errs_none,
            'hist': hist_none,
            'src_transformed': src_tf_none,
        },
        'odo': {
            'R': r_odo, 't': t_odo,
            'alpha_rad': _theta_from_r(r_odo),
            'alpha_deg': _deg(_theta_from_r(r_odo)),
            'rmse': float(errs_odo[-1]) if errs_odo.size > 0 else float('inf'),
            'iterations': int(len(hist_odo)),
            'n_corr_last': int(hist_odo[-1]['n_corr']) if len(hist_odo) > 0 else 0,
            'errs': errs_odo,
            'hist': hist_odo,
            'src_transformed': src_tf_odo,
        },
        'raw_none': {
            'R': r_raw_none, 't': t_raw_none,
            'alpha_rad': _theta_from_r(r_raw_none),
            'alpha_deg': _deg(_theta_from_r(r_raw_none)),
            'rmse': float(errs_raw_none[-1]) if errs_raw_none.size > 0 else float('inf'),
            'iterations': int(len(hist_raw_none)),
            'n_corr_last': int(hist_raw_none[-1]['n_corr']) if len(hist_raw_none) > 0 else 0,
            'errs': errs_raw_none,
            'hist': hist_raw_none,
            'src_transformed': src_tf_raw_none,
        },
        'raw_odo': {
            'R': r_raw_odo, 't': t_raw_odo,
            'alpha_rad': _theta_from_r(r_raw_odo),
            'alpha_deg': _deg(_theta_from_r(r_raw_odo)),
            'rmse': float(errs_raw_odo[-1]) if errs_raw_odo.size > 0 else float('inf'),
            'iterations': int(len(hist_raw_odo)),
            'n_corr_last': int(hist_raw_odo[-1]['n_corr']) if len(hist_raw_odo) > 0 else 0,
            'errs': errs_raw_odo,
            'hist': hist_raw_odo,
            'src_transformed': src_tf_raw_odo,
        },
        'best': {
            'R': best_r, 't': best_t,
            'pose_err_trans': best_trans_err,
            'pose_err_rot_deg': best_rot_err,
        },
    }
    return out


def run_icp_over_history(
    history: np.ndarray,
    lidar,
    env,
    *,
    step: int = 1,
    max_iterations: int = 40,
    tolerance: float = 1e-5,
    max_correspondence_distance: Optional[float] = None,
    use_scipy: bool = True,
    trim_fraction: Optional[float] = None,
    damping_enabled: bool = True,
    angle_thresh_deg: float = 7.5,
    struct_ratio_thresh: float = 0.03,
    damp_factor: float = 0.5,
    sliding_filter_enabled: bool = True,
    sliding_cos_threshold: float = 0.985,
    angle_balance_enabled: bool = True,
    angle_bin_deg: float = 10.0,
    angle_max_per_bin: int = 12,
    angle_prefer_far: bool = True,
    robust_enabled: bool = True,
    huber_c_factor: float = 1.5,
    dynamic_maxdist: bool = True,
    dynamic_factor: float = 2.0,
    dynamic_min: float = 0.20,
    dynamic_max: float = 0.50,
    progress_cb: Optional[callable] = None,
) -> List[Dict]:
    """Esegue ICP su coppie (k-1,k) a passi 'step' con filtro sliding opzionale.
    Ritorna una lista di risultati (dict) per ciascuna coppia.
    Se progress_cb è fornito viene chiamato come progress_cb(done, total) dopo ogni coppia."""
    n_total = int(len(history))  # rinominato da N
    results: List[Dict] = []
    step_i = int(max(1, step))
    total_pairs = len(range(1, n_total, step_i))
    done = 0
    for k in range(1, n_total, step_i):
        res = run_icp_pair_local(
            lidar, env, history[k-1], history[k],
            max_iterations=max_iterations,
            tolerance=tolerance,
            max_correspondence_distance=max_correspondence_distance,
            use_scipy=use_scipy,
            trim_fraction=trim_fraction,
            damping_enabled=damping_enabled,
            angle_thresh_deg=angle_thresh_deg,
            struct_ratio_thresh=struct_ratio_thresh,
            damp_factor=damp_factor,
            sliding_filter_enabled=sliding_filter_enabled,
            sliding_cos_threshold=sliding_cos_threshold,
            angle_balance_enabled=angle_balance_enabled,
            angle_bin_deg=angle_bin_deg,
            angle_max_per_bin=angle_max_per_bin,
            angle_prefer_far=angle_prefer_far,
            robust_enabled=robust_enabled,
            huber_c_factor=huber_c_factor,
            dynamic_maxdist=dynamic_maxdist,
            dynamic_factor=dynamic_factor,
            dynamic_min=dynamic_min,
            dynamic_max=dynamic_max,
        )
        res['k'] = int(k)
        results.append(res)
        done += 1
        if progress_cb is not None:
            try:
                progress_cb(done, total_pairs)
            except (TypeError, ValueError, RuntimeError):
                # Ignora callback malformate o errori runtime non critici
                pass
    return results
