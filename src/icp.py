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

def _estimate_normals_2d(points: np.ndarray, k: int = 8, use_scipy: bool = True) -> Optional[np.ndarray]:
    """Stima normali 2D (unit) per ciascun punto via PCA locale sui k vicini.
    Se ci sono meno di 3 punti, ritorna None. Usa SciPy KDTree se disponibile.
    Il verso della normale è arbitrario (±n); per p2l il segno non influisce (si usa r^2).
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 3:
        return None
    k_eff = max(3, min(k, n - 1))
    if use_scipy and _HAS_KDTREE:
        try:
            tree = _build_kdtree(pts)
            dists, idxs = tree.query(pts, k=k_eff+1)  # include il punto stesso
            normals = np.zeros((n, 2), dtype=float)
            for i in range(n):
                neigh_idx = idxs[i][1:] if np.ndim(idxs[i]) else []  # skip self
                if len(neigh_idx) < 2:
                    normals[i] = np.array([0.0, 0.0])
                    continue
                nb = pts[neigh_idx]
                cen = nb.mean(axis=0)
                cc = (nb - cen).T @ (nb - cen)
                try:
                    eigvals, eigvecs = np.linalg.eigh(cc)
                    # normale = autovettore associato a autovalore minore (perp alla direzione principale)
                    nvec = eigvecs[:, np.argmin(eigvals)]
                    nrm = float(np.linalg.norm(nvec)) or 1.0
                    normals[i] = nvec / nrm
                except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                    normals[i] = np.array([0.0, 0.0])
            # normalizza vettori nulli verso x
            z = np.linalg.norm(normals, axis=1) < 1e-9
            if np.any(z):
                normals[z] = np.array([1.0, 0.0])
            return normals
        except Exception:
            pass
    # Fallback brute-force: usa solo PCA sull'intero set (grossolano)
    cen = pts.mean(axis=0)
    cc = (pts - cen).T @ (pts - cen)
    try:
        eigvals, eigvecs = np.linalg.eigh(cc)
        nvec = eigvecs[:, np.argmin(eigvals)]
        nrm = float(np.linalg.norm(nvec)) or 1.0
        normals = np.tile(nvec / nrm, (n, 1))
        return normals
    except Exception:
        return None


def _p2l_increment(valid_src: np.ndarray, valid_dst: np.ndarray, normals: np.ndarray, *, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calcola un incremento (R_delta, t_delta) punto-linea con Gauss-Newton lineare.
    Approssimazione small-angle: R≈I + θJ, J=[[0,-1],[1,0]]. Parametri [tx, ty, θ].
    Restituisce (R_delta, t_delta, rmse_along_normals).
    """
    n = len(valid_src)
    if n < 3:
        return np.eye(2), np.zeros(2), float('inf')
    # Residui e jacobiani
    dif = (valid_src - valid_dst)  # al passo attuale (src già trasformata dalle iterazioni precedenti)
    nx = normals[:, 0]
    ny = normals[:, 1]
    # residuo lungo normale: r_i = n^T (p - q)
    r = nx * dif[:, 0] + ny * dif[:, 1]
    # Jacobiano: [n_x, n_y, (n_y * p_x - n_x * p_y)]
    j3 = ny * valid_src[:, 0] - nx * valid_src[:, 1]
    A = np.stack([nx, ny, j3], axis=1)  # N x 3
    # Regularizzazione Tikhonov lieve per stabilità numerica
    scale = float(np.median(np.linalg.norm(dif, axis=1))) if n > 0 else 1.0
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    lam_t = 1e-3 * (scale**2)
    lam_r = 1e-3
    L = np.diag([lam_t, lam_t, lam_r])
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        w = np.clip(w, 1e-6, 1e6)
        W = np.diag(w)
        AtW = A.T @ W
        H = AtW @ A + L
        b = - AtW @ r
    else:
        H = A.T @ A + L
        b = - A.T @ r
    try:
        delta = np.linalg.solve(H, b)
    except np.linalg.LinAlgError:
        try:
            delta = np.linalg.lstsq(H, b, rcond=None)[0]
        except Exception:
            return np.eye(2), np.zeros(2), float('inf')
    tx, ty, dth = map(float, delta)
    # Limita incremento per stabilità
    dth = float(np.clip(dth, -0.35, 0.35))
    # cap traslazione a un valore plausibile (3*scale o 0.6 m minimum cap)
    cap_t = max(0.6, 3.0 * scale)
    tx = float(np.clip(tx, -cap_t, cap_t))
    ty = float(np.clip(ty, -cap_t, cap_t))
    r_delta = rot2d(dth)
    t_delta = np.array([tx, ty], dtype=float)
    # RMSE lungo le normali dopo l'aggiornamento (approssimato linearmente)
    r_new = r + A @ np.array([tx, ty, dth])
    rmse_n = float(np.sqrt(np.mean(r_new * r_new)))
    return r_delta, t_delta, rmse_n


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
    mutual_filter_enabled: bool = True,
    p2l_enabled: bool = True,
    normals_k: int = 8,
    max_step_angle_deg: float = 3.0,
    max_step_trans: float = 0.5,
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
        # Mutual filter: tieni solo corrispondenze reciproche dst->src
        if mutual_filter_enabled and mask.any():
            try:
                # Calcola mappa inversa (solo per i candidati in mask per efficienza)
                valid_idx = np.where(mask)[0]
                if valid_idx.size > 0:
                    rev_src = dst[idxs[valid_idx]]
                    idxs_rev, _d_rev, _m_rev = nearest_neighbors(rev_src, src, max_distance=None, use_scipy=use_scipy)
                    # Accetta solo dove il vicino inverso punta allo stesso indice originale
                    ok_back = (idxs_rev == valid_idx)
                    full_mask = np.zeros_like(mask)
                    full_mask[valid_idx[ok_back]] = True
                    mask = mask & full_mask
            except Exception:
                pass
        # Adattività: se poche corrispondenze disabilita sliding/trim per non perdere informazione
        if adaptive_filters and mask.sum() < (2 * min_after_sliding):
            adaptive_sliding = False
            adaptive_trim = None

        # Soglia dinamica (MAD robusta sulla distanza) per stringere max_distance effettivo
        if dynamic_maxdist and mask.any():
            vd = dists[mask]
            if vd.size >= 6:
                med = float(np.median(vd))
                mad = float(np.median(np.abs(vd - med))) or (med * 0.1) or 1e-6
                thr_dyn = med + dynamic_factor * 1.4826 * mad
                thr_dyn = float(np.clip(thr_dyn, dynamic_min, dynamic_max))
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

        # Dopo aver ottenuto r_delta_raw/t_delta_raw e prima dell'aggiornamento, valuta P2L
        r_delta = r_delta_raw
        t_delta = t_delta_raw
        rmse_choice = None
        try:
            d_valid = dists[mask]
            rmse_choice = float(np.sqrt(np.mean(d_valid * d_valid))) if d_valid.size > 0 else float('inf')
        except Exception:
            rmse_choice = None
        if p2l_enabled:
            # Stima normali sui dst validi e calcola incremento P2L
            normals = _estimate_normals_2d(valid_dst, k=int(normals_k), use_scipy=use_scipy)
            if normals is not None:
                w_pl = None
                if robust_enabled and 'vd' in locals():
                    vd_loc = dists[mask]
                    if vd_loc.size >= 3:
                        med = float(np.median(vd_loc))
                        if med > 1e-12:
                            c = huber_c_factor * med
                            w = np.ones_like(vd_loc)
                            big = vd_loc > c
                            w[big] = c / (vd_loc[big] + 1e-12)
                            w_pl = w
                r_pl, t_pl, rmse_n = _p2l_increment(valid_src, valid_dst, normals, weights=w_pl)
                # Valuta RMSE euclidea dopo l'incremento P2L
                try:
                    src_after = (r_pl @ valid_src.T).T + t_pl.reshape(1, 2)
                    rmse_eu_p2l = float(np.sqrt(np.mean(np.sum((src_after - valid_dst)**2, axis=1))))
                except Exception:
                    rmse_eu_p2l = float('inf')
                # Heuristica di scelta
                use_p2l = False
                try:
                    s_max = float(np.max(s)) if 's' in locals() and len(s) > 0 else 1.0
                    s_min = float(np.min(s)) if 's' in locals() and len(s) > 0 else 1.0
                    struct_ratio = (s_min / s_max) if s_max > 1e-12 else 1.0
                except Exception:
                    struct_ratio = 1.0
                tnorm = float(np.linalg.norm(t_pl))
                cap_t_choice = max(0.6, 3.0 * (rmse_choice if (rmse_choice is not None and np.isfinite(rmse_choice)) else 0.1))
                if struct_ratio < float(struct_ratio_thresh) * 1.4:
                    use_p2l = True
                if (rmse_choice is not None) and np.isfinite(rmse_eu_p2l) and (rmse_eu_p2l < 0.95 * rmse_choice):
                    use_p2l = True
                # Escludi P2L se traslazione e' eccessiva o non finita
                if (not np.isfinite(tnorm)) or (tnorm > cap_t_choice):
                    use_p2l = False
                if use_p2l:
                    r_delta = r_pl
                    t_delta = t_pl
        # Hard caps su incremento
        try:
            # clamp rotazione
            ang = float(np.arctan2(r_delta[1, 0], r_delta[0, 0]))
            ang_cap = float(np.deg2rad(max(0.1, max_step_angle_deg)))
            if abs(ang) > ang_cap:
                ang = np.sign(ang) * ang_cap
                r_delta = rot2d(ang)
            # clamp traslazione
            tnorm = float(np.linalg.norm(t_delta))
            tcap = max(0.05, float(max_step_trans))
            if tnorm > tcap and np.isfinite(tnorm):
                t_delta = t_delta * (tcap / (tnorm + 1e-12))
        except Exception:
            pass
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


# --------------------------- Projective ICP (basato su indice raggio) ---------------------------

def _compute_normals_projective(idx: np.ndarray, pts: np.ndarray, window: int = 2) -> np.ndarray:
    n = len(idx)
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    normals = np.zeros((n, 2), dtype=float)
    for k in range(n):
        k0 = max(0, k - int(window))
        k1 = min(n - 1, k + int(window))
        if k1 == k0:
            normals[k] = np.array([1.0, 0.0])
            continue
        p0 = pts[k0]
        p1 = pts[k1]
        t = p1 - p0
        nt = float(np.linalg.norm(t))
        if nt < 1e-9 or not np.isfinite(nt):
            normals[k] = np.array([1.0, 0.0])
            continue
        t = t / nt
        nvec = np.array([-t[1], t[0]], dtype=float)
        normals[k] = nvec / (float(np.linalg.norm(nvec)) + 1e-12)
    return normals


def _icp_projective_once(src: np.ndarray, dst: np.ndarray, dst_normals: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Singolo step di ICP point-to-line 2D con Gauss-Newton linearizzato."""
    n = len(src)
    if n < 3:
        return np.eye(2), np.zeros(2), float('inf')
    dif = (src - dst)
    nx = dst_normals[:, 0]
    ny = dst_normals[:, 1]
    r = nx * dif[:, 0] + ny * dif[:, 1]
    j3 = ny * src[:, 0] - nx * src[:, 1]
    A = np.stack([nx, ny, j3], axis=1)

    # Regularizzazione ridotta per permettere movimenti più grandi
    scale = float(np.median(np.abs(r))) if n > 0 else 1.0
    scale = max(scale, 0.01)
    lam_t = 1e-5 * (scale**2)
    lam_r = 1e-5
    L = np.diag([lam_t, lam_t, lam_r])

    if weights is not None:
        w = np.clip(np.asarray(weights, dtype=float).reshape(-1), 1e-6, 1e6)
        W = np.diag(w)
        H = A.T @ W @ A + L
        b = - A.T @ W @ r
    else:
        H = A.T @ A + L
        b = - A.T @ r
    try:
        delta = np.linalg.solve(H, b)
    except np.linalg.LinAlgError:
        try:
            delta = np.linalg.lstsq(H, b, rcond=None)[0]
        except Exception:
            return np.eye(2), np.zeros(2), float('inf')

    tx, ty, dth = map(float, delta)

    # Caps più permissivi per consentire movimenti reali
    dth = float(np.clip(dth, -0.15, 0.15))  # ~8.6°
    tcap = max(0.5, 5.0 * scale)  # adattivo alla scala del problema
    nt = float(np.hypot(tx, ty))
    if nt > tcap and np.isfinite(nt):
        s = tcap / (nt + 1e-12)
        tx *= s; ty *= s

    R = rot2d(dth)
    t = np.array([tx, ty], dtype=float)

    # Calcola RMSE dopo l'applicazione della trasformazione
    src_a = (R @ src.T).T + t
    pr = dst_normals[:, 0] * (src_a[:, 0] - dst[:, 0]) + dst_normals[:, 1] * (src_a[:, 1] - dst[:, 1])
    rmse = float(np.sqrt(np.mean(pr * pr))) if pr.size > 0 else float('inf')

    if not np.isfinite(rmse):
        return np.eye(2), np.zeros(2), float('inf')

    return R, t, rmse


def icp_projective(
    idx_src: np.ndarray,
    src_local: np.ndarray,
    idx_dst: np.ndarray,
    dst_local: np.ndarray,
    *,
    max_iterations: int = 30,
    tolerance: float = 1e-5,
    robust: bool = True,
    huber_k: float = 1.5,
    depth_gate: float = 0.50,
    normals_window: int = 2,
    init_R: Optional[np.ndarray] = None,
    init_t: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """ICP proiettivo migliorato con inizializzazione opzionale e correzioni numeriche."""
    idx_src = np.asarray(idx_src, dtype=int)
    idx_dst = np.asarray(idx_dst, dtype=int)
    src_local = np.asarray(src_local, dtype=float)
    dst_local = np.asarray(dst_local, dtype=float)
    common, i_src, i_dst = np.intersect1d(idx_src, idx_dst, return_indices=True)
    if common.size < 3:
        return np.eye(2), np.zeros(2), {"ok": False, "reason": "not_enough_common_rays", "n_corr": int(common.size)}
    src = src_local[i_src].copy()
    dst = dst_local[i_dst]

    # Applica inizializzazione se fornita
    R_tot = np.eye(2) if init_R is None else np.asarray(init_R, dtype=float).copy()
    t_tot = np.zeros(2) if init_t is None else np.asarray(init_t, dtype=float).copy()

    if init_R is not None and init_t is not None:
        src = (init_R @ src.T).T + init_t.reshape(1, 2)

    # gate su profondità più permissivo
    rs = np.linalg.norm(src, axis=1)
    rd = np.linalg.norm(dst, axis=1)
    keep = np.abs(rs - rd) <= float(depth_gate)
    src = src[keep]; dst = dst[keep]; common = common[keep]
    if len(src) < 3:
        return R_tot, t_tot, {"ok": False, "reason": "not_enough_after_depth_gate", "n_corr": int(len(src))}

    normals = _compute_normals_projective(common, dst, window=int(normals_window))
    errors: List[float] = []
    prev_rmse = None
    converged = False

    for it in range(int(max_iterations)):
        weights = None
        if robust:
            dif = (src - dst)
            proj = normals[:, 0] * dif[:, 0] + normals[:, 1] * dif[:, 1]
            med = float(np.median(np.abs(proj))) if proj.size > 0 else 0.0
            c = huber_k * max(med, 0.01)
            w = np.ones_like(proj)
            big = np.abs(proj) > c
            if np.any(big):
                w[big] = c / (np.abs(proj[big]) + 1e-12)
            weights = w
        R_d, t_d, rmse = _icp_projective_once(src, dst, normals, weights)

        # Controlla se l'incremento è degenere
        if not np.isfinite(rmse) or rmse > 1e6:
            break

        src = (R_d @ src.T).T + t_d
        t_tot = R_d @ t_tot + t_d
        R_tot = R_d @ R_tot
        errors.append(float(rmse))

        if prev_rmse is not None and abs(prev_rmse - rmse) < float(tolerance):
            converged = True
            break
        prev_rmse = rmse

    return R_tot, t_tot, {"ok": True, "rmse": (errors[-1] if errors else float("inf")), "iterations": len(errors), "errors": np.array(errors, dtype=float), "n_corr": int(len(src)), "converged": converged}


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
    """Esegue ICP tra le scansioni consecutive con inizializzazione da odometria.

    Esegue due run: (A) con inizializzazione da odometria e (B) senza inizializzazione.
    Ritorna un dizionario con risultati e metriche per entrambe le varianti.

    Convenzione: ICP cerca T : source -> target, quindi src_transformed = T @ src + t.
    In questo caso: source=scan_k (curr), target=scan_{k-1} (prev).
    L'odometria fornisce la trasformazione relativa frame_k -> frame_{k-1},
    che è esattamente quella che l'ICP deve trovare.
    """
    # Estrai punti di impatto nel frame locale del sensore con indici di raggio
    idx_tgt, tgt_local = lidar.scan_hits_indexed(prev_pose, env, frame='local')  # target = k-1
    idx_src, src_local = lidar.scan_hits_indexed(curr_pose, env, frame='local')  # source = k
    if tgt_local is None or src_local is None or len(tgt_local) < 3 or len(src_local) < 3:
        return {
            'ok': False,
            'reason': 'not_enough_points',
            'n_src': 0 if src_local is None else int(len(src_local)),
            'n_tgt': 0 if tgt_local is None else int(len(tgt_local)),
        }

    # Calcola trasformazione relativa da odometria come buona inizializzazione
    # relative_local_transform(prev, curr) dà la trasf che porta punti da frame_k a frame_{k-1}
    R_odom, t_odom = relative_local_transform(prev_pose, curr_pose)

    # (A) ICP projective con inizializzazione da odometria
    Rp, tp, info_p = icp_projective(idx_src, src_local, idx_tgt, tgt_local,
                                    max_iterations=min(20, int(max_iterations)),
                                    tolerance=float(tolerance),
                                    robust=True, huber_k=1.5,
                                    depth_gate=0.60, normals_window=3,
                                    init_R=R_odom, init_t=t_odom)
    # (B) ICP projective senza inizializzazione
    Rr, tr, info_r = icp_projective(idx_src, src_local, idx_tgt, tgt_local,
                                    max_iterations=min(20, int(max_iterations)),
                                    tolerance=float(tolerance),
                                    robust=True, huber_k=1.5,
                                    depth_gate=0.60, normals_window=3,
                                    init_R=None, init_t=None)

    def _theta_from_r(r_mat: np.ndarray) -> float:
        return float(np.arctan2(r_mat[1, 0], r_mat[0, 0]))
    def _deg(rad: float) -> float:
        return float(rad * 180.0 / np.pi)

    # Plausibility checks: limita risultati assurdi (outlier)
    def _plausible(R: np.ndarray, t: np.ndarray, rmse: float) -> Tuple[np.ndarray, np.ndarray, float]:
        angle = abs(_theta_from_r(R))
        tnorm = float(np.linalg.norm(t))
        if not np.isfinite(angle) or not np.isfinite(tnorm):
            return np.eye(2), np.zeros(2), float('inf')
        # Soglie più permissive per movimenti reali
        if angle > np.deg2rad(15.0) or tnorm > 1.00:
            return np.eye(2), np.zeros(2), float('inf')
        return R, t, rmse

    Rp, tp, rp_rmse = _plausible(Rp, tp, float(info_p.get('rmse', float('inf'))))
    Rr, tr, rr_rmse = _plausible(Rr, tr, float(info_r.get('rmse', float('inf'))))
    info_p['rmse'] = rp_rmse
    info_r['rmse'] = rr_rmse

    # Conversione blocchi con schema compatibile con il resto del codice
    out = {
        'ok': True,
        'n_src': int(len(src_local)),
        'n_tgt': int(len(tgt_local)),
        'gt_R': R_odom, 'gt_t': t_odom,  # Ground truth dall'odometria
        'src_local': src_local, 'tgt_local': tgt_local,
        'none': {
            'R': Rp, 't': tp,
            'alpha_rad': _theta_from_r(Rp),
            'alpha_deg': _deg(_theta_from_r(Rp)),
            'rmse': float(info_p.get('rmse', float('inf'))),
            'iterations': int(info_p.get('iterations', 0)),
            'n_corr_last': int(info_p.get('n_corr', 0)),
            'pose_err_trans': None,
            'pose_err_rot_deg': None,
            'errs': info_p.get('errors', np.array([], dtype=float)),
            'hist': [],
            'src_transformed': (Rp @ np.asarray(src_local).T).T + tp.reshape(1, 2),
        },
        'raw_none': {
            'R': Rr, 't': tr,
            'alpha_rad': _theta_from_r(Rr),
            'alpha_deg': _deg(_theta_from_r(Rr)),
            'rmse': float(info_r.get('rmse', float('inf'))),
            'iterations': int(info_r.get('iterations', 0)),
            'n_corr_last': int(info_r.get('n_corr', 0)),
            'errs': info_r.get('errors', np.array([], dtype=float)),
            'hist': [],
            'src_transformed': (Rr @ np.asarray(src_local).T).T + tr.reshape(1, 2),
        },
    }
    # Best variant: preferisci quello con odometria se convergito, altrimenti il migliore RMSE
    if info_p.get('converged', False) and rp_rmse < float('inf'):
        best_block = out['none']
    elif out['none']['rmse'] <= out['raw_none']['rmse']:
        best_block = out['none']
    else:
        best_block = out['raw_none']
    out['best'] = {
        'R': best_block['R'],
        't': best_block['t'],
        'alpha_rad': best_block['alpha_rad'],
        'alpha_deg': best_block['alpha_deg'],
        'rmse': best_block['rmse'],
        'iterations': best_block['iterations'],
        'n_corr_last': best_block['n_corr_last'],
        'pose_err_trans': None,
        'pose_err_rot_deg': None,
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
