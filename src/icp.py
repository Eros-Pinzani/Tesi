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
    use_scipy: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict], np.ndarray]:
    """ICP point-to-point 2D.

    Ritorna: (R_finale, t_finale, source_trasformata, history, errors)
    - history: lista di dict con chiavi {'iter', 'R', 't', 'mean_error', 'n_corr'}
    - errors: np.array dei mean_error per iterazione
    """
    assert source.ndim == 2 and source.shape[1] == 2 and target.ndim == 2 and target.shape[1] == 2
    src = np.asarray(source, dtype=float).copy()
    dst = np.asarray(target, dtype=float).copy()

    # Applica posa iniziale se fornita
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
        valid_src = src[mask]
        valid_dst = dst[idxs[mask]]
        if len(valid_src) < 3:
            if verbose:
                print(f"[ICP] Iter {it}: corrispondenze insufficienti ({len(valid_src)}). Stop.")
            break
        R_delta, t_delta = best_fit_transform_2d(valid_src, valid_dst)
        # Aggiorna i punti src e accumula la trasformazione
        src = (R_delta @ src.T).T + t_delta.reshape(1, 2)
        total_t = R_delta @ total_t + t_delta
        total_R = R_delta @ total_R

        mean_error = float(np.mean(dists[mask])) if mask.any() else float('inf')
        history.append({'iter': it, 'R': R_delta, 't': t_delta, 'mean_error': mean_error, 'n_corr': int(mask.sum())})
        if verbose:
            print(f"[ICP] Iter {it}: mean_error={mean_error:.6f}, n_corr={int(mask.sum())}")
        if prev_error is not None and abs(prev_error - mean_error) < float(tolerance):
            if verbose:
                print(f"[ICP] Converged at iter {it} (Δerror < tol).")
            break
        prev_error = mean_error

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
) -> Dict:
    """Esegue ICP tra le scansioni LiDAR alle pose consecutive, entrambe in frame locale del robot.

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
    R_none, t_none, _, hist_none, errs_none = icp_point_to_point(
        src_local, tgt_local,
        init_pose=None,
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_distance=max_correspondence_distance,
        use_scipy=use_scipy,
        verbose=False,
    )

    # (B) ICP con inizializzazione odometrica (curr->prev nel frame locale di prev)
    R0, t0 = relative_local_transform(prev_pose, curr_pose)
    R_odo, t_odo, _, hist_odo, errs_odo = icp_point_to_point(
        src_local, tgt_local,
        init_pose=(R0, t0),
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_distance=max_correspondence_distance,
        use_scipy=use_scipy,
        verbose=False,
    )

    def _theta_from_R(R: np.ndarray) -> float:
        return float(np.arctan2(R[1, 0], R[0, 0]))

    out = {
        'ok': True,
        'n_src': int(len(src_local)),
        'n_tgt': int(len(tgt_local)),
        'none': {
            'R': R_none, 't': t_none, 'alpha': _theta_from_R(R_none),
            'rmse': float(errs_none[-1]) if errs_none.size > 0 else float('inf'),
            'iterations': int(len(hist_none)),
            'n_corr_last': int(hist_none[-1]['n_corr']) if len(hist_none) > 0 else 0,
        },
        'odo': {
            'R': R_odo, 't': t_odo, 'alpha': _theta_from_R(R_odo),
            'rmse': float(errs_odo[-1]) if errs_odo.size > 0 else float('inf'),
            'iterations': int(len(hist_odo)),
            'n_corr_last': int(hist_odo[-1]['n_corr']) if len(hist_odo) > 0 else 0,
        }
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
) -> List[Dict]:
    """Esegue ICP su coppie (k-1,k) a passi 'step' lungo la storia.
    Ritorna una lista di risultati (dict) per ciascuna coppia.
    """
    N = int(len(history))
    results: List[Dict] = []
    for k in range(1, N, int(max(1, step))):
        res = run_icp_pair_local(
            lidar, env, history[k-1], history[k],
            max_iterations=max_iterations,
            tolerance=tolerance,
            max_correspondence_distance=max_correspondence_distance,
            use_scipy=use_scipy,
        )
        res['k'] = int(k)
        results.append(res)
    return results
