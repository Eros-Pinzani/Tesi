"""Algoritmo ICP 2D (point-to-point) + utility per eseguirlo su scansioni LiDAR consecutive.

- Lavora in frame ROBOT (locale) come richiesto.
- Esegue due varianti: init_pose=None (nessuna inizializzazione) e init_pose da odometria (stima relativa fra pose consecutive).
- Non richiede SciPy; se presente, usa cKDTree per i nearest neighbors, altrimenti fallback O(N*M).
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple

# --------------------------- Algebra di base ---------------------------

def rot2d(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=float)


def pose_to_R_t(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converte una posa [x, y, theta] in (R 2x2, t 2,)."""
    x, y, th = map(float, pose)
    return rot2d(th), np.array([x, y], dtype=float)


def relative_local_transform(prev_pose: np.ndarray, curr_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Trasformazione che porta punti espressi nel frame del robot a tempo k (curr)
    nel frame del robot a tempo k-1 (prev): p_{k-1} = R_rel @ p_k + t_rel.

    Derivazione:
      p_w = Rk p_k + tk  ;  p_{k-1} = R_{k-1}^T (p_w - t_{k-1})
      => p_{k-1} = (R_{k-1}^T Rk) p_k + R_{k-1}^T (tk - t_{k-1})
    """
    R_prev, t_prev = pose_to_R_t(prev_pose)
    R_curr, t_curr = pose_to_R_t(curr_pose)
    R_rel = R_prev.T @ R_curr
    t_rel = R_prev.T @ (t_curr - t_prev)
    return R_rel, t_rel


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

def best_fit_transform_2d(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calcola (R 2x2, t 2,) che minimizza ||R A + t - B||_F, con corrispondenze A<->B.
    A, B: (N,2) numpy arrays
    """
    assert A.shape == B.shape and A.shape[1] == 2
    N = A.shape[0]
    if N == 0:
        return np.eye(2), np.zeros(2)
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t


def nearest_neighbors(src: np.ndarray, dst: np.ndarray, max_distance: Optional[float] = None, use_scipy: bool = True):
    """Per ogni punto src, trova il nearest neighbor in dst. Ritorna (idxs, dists, mask_inliers)."""
    if use_scipy:
        try:
            import importlib
            spatial = importlib.import_module('scipy.spatial')  # type: ignore
            KDTree = getattr(spatial, 'cKDTree', None)
            if KDTree is not None:
                tree = KDTree(np.asarray(dst, dtype=float))
                dists, idxs = tree.query(np.asarray(src, dtype=float), k=1)
                mask = (dists <= float(max_distance)) if (max_distance is not None) else np.ones_like(dists, dtype=bool)
                return idxs, dists, mask
        except Exception:
            pass  # fallback sotto
    # Fallback O(N*M)
    N = src.shape[0]
    idxs = np.empty(N, dtype=int)
    dists = np.empty(N, dtype=float)
    mask = np.ones(N, dtype=bool)
    for i in range(N):
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
    # Nuovi parametri robustezza & soglia dinamica
    robust_enabled: bool = True,
    huber_c_factor: float = 1.5,
    dynamic_maxdist: bool = True,
    dynamic_factor: float = 2.0,
    dynamic_min: float = 0.20,
    dynamic_max: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict], np.ndarray]:
    """ICP point-to-point 2D con filtro opzionale anti-sliding.

    Ritorna: (R_finale, t_finale, source_trasformata, history, errors)
    - history: lista di dict con chiavi {'iter', 'R', 't', 'mean_error', 'n_corr'}
    - errors: np.array degli RMSE per iterazione (vero RMSE)
    """
    assert source.ndim == 2 and source.shape[1] == 2 and target.ndim == 2 and target.shape[1] == 2
    src = np.asarray(source, dtype=float).copy()
    dst = np.asarray(target, dtype=float).copy()

    total_R = np.eye(2)
    total_t = np.zeros(2)
    if init_pose is not None:
        R0, t0 = init_pose
        src = (R0 @ src.T).T + t0.reshape(1, 2)
        total_R = R0 @ total_R
        total_t = R0 @ total_t + t0

    prev_error: Optional[float] = None
    history: List[Dict] = []

    for it in range(int(max_iterations)):
        idxs, dists, mask = nearest_neighbors(src, dst, max_correspondence_distance, use_scipy=use_scipy)

        # Soglia dinamica per corrispondenze (stringe il max_distance effettivo in base alla mediana)
        if dynamic_maxdist and mask.any():
            vd = dists[mask]
            if vd.size >= 6:
                med = float(np.median(vd))
                thr_dyn = max(dynamic_min, min(dynamic_max, dynamic_factor * med))
                mask = mask & (dists <= thr_dyn)

        # Trimming corrispondenze più lontane
        if mask.any() and trim_fraction is not None and 0.0 < float(trim_fraction) < 1.0:
            valid_d = dists[mask]
            if len(valid_d) >= 6:
                thr_trim = float(np.quantile(valid_d, float(trim_fraction)))
                mask = mask & (dists <= thr_trim)

        # Filtro sliding (solo se abilitato e abbastanza corrispondenze)
        if sliding_filter_enabled and mask.sum() >= min_after_sliding:
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
            except Exception:
                pass  # in caso di errore nel PCA salta il filtro

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
            centroid_A = valid_src.mean(axis=0)
            centroid_B = valid_dst.mean(axis=0)
            AA = valid_src - centroid_A
            BB = valid_dst - centroid_B
            H = AA.T @ BB
        else:
            w = weights.reshape(-1, 1)
            Wsum = float(np.sum(w)) or 1.0
            centroid_A = (valid_src * w).sum(axis=0) / Wsum
            centroid_B = (valid_dst * w).sum(axis=0) / Wsum
            AA = valid_src - centroid_A
            BB = valid_dst - centroid_B
            H = AA.T @ (BB * w)
        U, S, Vt = np.linalg.svd(H)
        R_delta_raw = Vt.T @ U.T
        if np.linalg.det(R_delta_raw) < 0:
            Vt[1, :] *= -1
            R_delta_raw = Vt.T @ U.T
        t_delta_raw = centroid_B - R_delta_raw @ centroid_A

        # Damping opzionale
        R_delta = R_delta_raw
        t_delta = t_delta_raw
        if damping_enabled:
            s_max = float(np.max(S)) if S.size > 0 else 1.0
            s_min = float(np.min(S)) if S.size > 0 else 1.0
            struct_ratio = (s_min / s_max) if s_max > 1e-12 else 1.0
            angle_raw = float(np.arctan2(R_delta_raw[1, 0], R_delta_raw[0, 0]))
            angle_thresh = float(np.deg2rad(angle_thresh_deg))
            if struct_ratio < float(struct_ratio_thresh) and abs(angle_raw) > angle_thresh:
                r = struct_ratio / float(struct_ratio_thresh)
                r = max(0.0, min(1.0, r))
                dyn_factor = float(damp_factor) + (1.0 - float(damp_factor)) * r
                angle_new = angle_raw * dyn_factor
                R_delta = rot2d(angle_new)
                t_delta = centroid_B - R_delta @ centroid_A

        # Aggiorna i punti src e accumula la trasformazione
        src = (R_delta @ src.T).T + t_delta.reshape(1, 2)
        total_t = R_delta @ total_t + t_delta
        total_R = R_delta @ total_R

        d_valid = dists[mask]
        rmse = float(np.sqrt(np.mean(d_valid * d_valid))) if d_valid.size > 0 else float('inf')
        history.append({'iter': it, 'R': R_delta, 't': t_delta, 'mean_error': rmse, 'n_corr': int(mask.sum())})
        if verbose:
            extra = ''
            if 'S' in locals():
                try:
                    sr = (float(np.min(S)) / float(np.max(S))) if (S.size > 0 and float(np.max(S)) > 1e-12) else 1.0
                except Exception:
                    sr = 1.0
                extra = f", struct_ratio={sr:.3f}"
            print(f"[ICP] Iter {it}: rmse={rmse:.6f}, n_corr={int(mask.sum())}{extra}")
        if prev_error is not None and abs(prev_error - rmse) < float(tolerance):
            if verbose:
                print(f"[ICP] Converged at iter {it} (Δrmse < tol).")
            break
        prev_error = rmse

    source_transformed = (total_R @ np.asarray(source).T).T + total_t.reshape(1, 2)
    errors = np.array([h['mean_error'] for h in history], dtype=float)
    return total_R, total_t, source_transformed, history, errors


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

    # Bilanciamento angolare per aumentare diversita' ed evitare sovra-rappresentazione di un solo bordo
    if angle_balance_enabled:
        tgt_local = _angle_uniform_subsample(tgt_local, bin_deg=angle_bin_deg, max_per_bin=angle_max_per_bin, prefer_far=angle_prefer_far)
        src_local = _angle_uniform_subsample(src_local, bin_deg=angle_bin_deg, max_per_bin=angle_max_per_bin, prefer_far=angle_prefer_far)
        if len(tgt_local) < 3 or len(src_local) < 3:
            return {
                'ok': False,
                'reason': 'after_balance_not_enough_points',
                'n_src': int(len(src_local)),
                'n_tgt': int(len(tgt_local)),
            }

    # (A) ICP senza inizializzazione
    R_none, t_none, src_tf_none, hist_none, errs_none = icp_point_to_point(
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

    # (B) ICP con inizializzazione odometrica (curr->prev nel frame locale di prev)
    R0, t0 = relative_local_transform(prev_pose, curr_pose)
    R_odo, t_odo, src_tf_odo, hist_odo, errs_odo = icp_point_to_point(
        src_local, tgt_local,
        init_pose=(R0, t0),
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

    # (C) ICP RAW (nudo e crudo): nessun filtro/tweak
    R_raw_none, t_raw_none, src_tf_raw_none, hist_raw_none, errs_raw_none = icp_point_to_point(
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
        angle_balance_enabled=False,
        robust_enabled=False,
        dynamic_maxdist=False,
    )
    R_raw_odo, t_raw_odo, src_tf_raw_odo, hist_raw_odo, errs_raw_odo = icp_point_to_point(
        src_local, tgt_local,
        init_pose=(R0, t0),
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_distance=None,
        trim_fraction=None,
        use_scipy=use_scipy,
        verbose=False,
        damping_enabled=False,
        sliding_filter_enabled=False,
        angle_balance_enabled=False,
        robust_enabled=False,
        dynamic_maxdist=False,
    )

    def _theta_from_R(R: np.ndarray) -> float:
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _deg(rad: float) -> float:
        return float(rad * 180.0 / np.pi)

    out = {
        'ok': True,
        'n_src': int(len(src_local)),
        'n_tgt': int(len(tgt_local)),
        'gt_R': R0, 'gt_t': t0,  # ground truth relativa
        'src_local': src_local, 'tgt_local': tgt_local,
        'none': {
            'R': R_none, 't': t_none,
            'alpha_rad': _theta_from_R(R_none),
            'alpha_deg': _deg(_theta_from_R(R_none)),
            'rmse': float(errs_none[-1]) if errs_none.size > 0 else float('inf'),
            'iterations': int(len(hist_none)),
            'n_corr_last': int(hist_none[-1]['n_corr']) if len(hist_none) > 0 else 0,
            'errs': errs_none,
            'hist': hist_none,
            'src_transformed': src_tf_none,
        },
        'odo': {
            'R': R_odo, 't': t_odo,
            'alpha_rad': _theta_from_R(R_odo),
            'alpha_deg': _deg(_theta_from_R(R_odo)),
            'rmse': float(errs_odo[-1]) if errs_odo.size > 0 else float('inf'),
            'iterations': int(len(hist_odo)),
            'n_corr_last': int(hist_odo[-1]['n_corr']) if len(hist_odo) > 0 else 0,
            'errs': errs_odo,
            'hist': hist_odo,
            'src_transformed': src_tf_odo,
        },
        'raw_none': {
            'R': R_raw_none, 't': t_raw_none,
            'alpha_rad': _theta_from_R(R_raw_none),
            'alpha_deg': _deg(_theta_from_R(R_raw_none)),
            'rmse': float(errs_raw_none[-1]) if errs_raw_none.size > 0 else float('inf'),
            'iterations': int(len(hist_raw_none)),
            'n_corr_last': int(hist_raw_none[-1]['n_corr']) if len(hist_raw_none) > 0 else 0,
            'errs': errs_raw_none,
            'hist': hist_raw_none,
            'src_transformed': src_tf_raw_none,
        },
        'raw_odo': {
            'R': R_raw_odo, 't': t_raw_odo,
            'alpha_rad': _theta_from_R(R_raw_odo),
            'alpha_deg': _deg(_theta_from_R(R_raw_odo)),
            'rmse': float(errs_raw_odo[-1]) if errs_raw_odo.size > 0 else float('inf'),
            'iterations': int(len(hist_raw_odo)),
            'n_corr_last': int(hist_raw_odo[-1]['n_corr']) if len(hist_raw_odo) > 0 else 0,
            'errs': errs_raw_odo,
            'hist': hist_raw_odo,
            'src_transformed': src_tf_raw_odo,
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
    N = int(len(history))
    results: List[Dict] = []
    step_i = int(max(1, step))
    total_pairs = len(range(1, N, step_i))
    done = 0
    for k in range(1, N, step_i):
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
            except Exception:
                pass
    return results
