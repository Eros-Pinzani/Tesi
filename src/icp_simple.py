"""
ICP Point-to-Point Semplice e Robusto
Implementazione pulita senza filtri complicati
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any


def compute_relative_transform_from_odometry(prev_pose: np.ndarray, curr_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola la trasformazione relativa tra due pose consecutive usando l'odometria.

    Args:
        prev_pose: [x, y, theta] al tempo k-1
        curr_pose: [x, y, theta] al tempo k

    Returns:
        R: Matrice di rotazione 2x2
        t: Vettore traslazione 2D
    """
    x_prev, y_prev, theta_prev = prev_pose[:3]
    x_curr, y_curr, theta_curr = curr_pose[:3]

    # Differenza angolare
    d_theta = theta_curr - theta_prev

    # Rotazione relativa
    cos_dt = np.cos(d_theta)
    sin_dt = np.sin(d_theta)
    R = np.array([[cos_dt, -sin_dt],
                  [sin_dt, cos_dt]], dtype=np.float64)

    # Traslazione nel frame precedente
    cos_prev = np.cos(theta_prev)
    sin_prev = np.sin(theta_prev)

    dx_world = x_curr - x_prev
    dy_world = y_curr - y_prev

    # Ruota nel frame locale di k-1
    t_x = cos_prev * dx_world + sin_prev * dy_world
    t_y = -sin_prev * dx_world + cos_prev * dy_world

    t = np.array([t_x, t_y], dtype=np.float64)

    return R, t


def find_nearest_neighbors(source: np.ndarray, target: np.ndarray, max_distance: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trova i nearest neighbors tra source e target.

    Args:
        source: Array Nx2 di punti source
        target: Array Mx2 di punti target
        max_distance: Distanza massima per considerare una corrispondenza

    Returns:
        source_matched: Punti source con corrispondenza valida
        target_matched: Punti target corrispondenti
    """
    if len(source) == 0 or len(target) == 0:
        return np.array([]), np.array([])

    # Calcola distanze euclidee per ogni punto source
    source_matched = []
    target_matched = []

    for i in range(len(source)):
        # Distanze da questo punto source a tutti i target
        distances = np.linalg.norm(target - source[i], axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        if min_dist < max_distance:
            source_matched.append(source[i])
            target_matched.append(target[min_idx])

    if len(source_matched) == 0:
        return np.array([]), np.array([])

    return np.array(source_matched), np.array(target_matched)


def compute_transformation_svd(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola la trasformazione ottimale usando SVD.
    Trova R e t tali che: target ≈ R * source + t

    Args:
        source: Array Nx2 di punti source
        target: Array Nx2 di punti target (corrispondenze)

    Returns:
        R: Matrice di rotazione 2x2
        t: Vettore traslazione 2D
    """
    # Centra i punti
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)

    source_centered = source - centroid_source
    target_centered = target - centroid_target

    # Matrice di covarianza H = target^T * source
    H = target_centered.T @ source_centered

    # SVD
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt

    # Assicura che sia una rotazione propria (det = +1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Traslazione: t = centroid_target - R * centroid_source
    t = centroid_target - R @ centroid_source

    return R, t


def compute_rmse(source: np.ndarray, target: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    """
    Calcola l'RMSE dopo aver applicato la trasformazione.

    Args:
        source: Punti source Nx2
        target: Punti target Nx2
        R: Rotazione 2x2
        t: Traslazione 2D

    Returns:
        RMSE
    """
    if len(source) == 0:
        return float('inf')

    transformed = (R @ source.T).T + t
    errors = np.linalg.norm(transformed - target, axis=1)
    return np.sqrt(np.mean(errors ** 2))


def icp_simple(source: np.ndarray,
               target: np.ndarray,
               init_R: Optional[np.ndarray] = None,
               init_t: Optional[np.ndarray] = None,
               max_iterations: int = 50,
               tolerance: float = 1e-6,
               max_correspondence_distance: float = 0.5) -> Dict[str, Any]:
    """
    ICP Point-to-Point semplice e robusto.

    Args:
        source: Punti source Nx2
        target: Punti target Mx2
        init_R: Rotazione iniziale (da odometria)
        init_t: Traslazione iniziale (da odometria)
        max_iterations: Numero massimo di iterazioni
        tolerance: Soglia di convergenza
        max_correspondence_distance: Distanza massima per corrispondenze

    Returns:
        Dictionary con risultati ICP
    """
    # Inizializzazione
    if init_R is None:
        R = np.eye(2, dtype=np.float64)
    else:
        R = np.array(init_R, dtype=np.float64).copy()

    if init_t is None:
        t = np.zeros(2, dtype=np.float64)
    else:
        t = np.array(init_t, dtype=np.float64).copy()

    # Applica trasformazione iniziale
    source_transformed = (R @ source.T).T + t

    prev_rmse = float('inf')
    errors_history = []
    iteration = 0
    source_matched = np.array([])
    target_matched = np.array([])

    for iteration in range(max_iterations):
        # 1. Find correspondences
        source_matched, target_matched = find_nearest_neighbors(
            source_transformed, target, max_correspondence_distance
        )

        # Verifica che ci siano abbastanza corrispondenze
        if len(source_matched) < 3:
            break

        # 2. Compute transformation
        R_iter, t_iter = compute_transformation_svd(source_matched, target_matched)

        # 3. Apply transformation
        source_transformed = (R_iter @ source_transformed.T).T + t_iter

        # 4. Update cumulative transformation
        R = R_iter @ R
        t = R_iter @ t + t_iter

        # 5. Compute RMSE
        rmse = compute_rmse(source_matched, target_matched, R_iter, t_iter)
        errors_history.append(rmse)

        # 6. Check convergence
        if abs(prev_rmse - rmse) < tolerance:
            break

        prev_rmse = rmse

    # Calcola angolo finale
    angle_rad = np.arctan2(R[1, 0], R[0, 0])
    angle_deg = np.degrees(angle_rad)

    return {
        'R': R,
        't': t,
        'iterations': iteration + 1,
        'rmse': prev_rmse if prev_rmse != float('inf') else 0.0,
        'angle_deg': angle_deg,
        'errors': np.array(errors_history),
        'n_correspondences': len(source_matched) if len(source_matched) > 0 else 0,
        'converged': iteration < max_iterations - 1
    }


def run_icp_pair(prev_pose: np.ndarray,
                 curr_pose: np.ndarray,
                 src_local: np.ndarray,
                 tgt_local: np.ndarray,
                 max_iterations: int = 50,
                 tolerance: float = 1e-6,
                 max_correspondence_distance: float = 0.5) -> Dict[str, Any]:
    """
    Esegue ICP su una coppia di scan con formato compatibile col codice esistente.

    Args:
        prev_pose: Pose al tempo k-1 [x, y, theta]
        curr_pose: Pose al tempo k [x, y, theta]
        src_local: Punti scan al tempo k nel frame locale
        tgt_local: Punti scan al tempo k-1 nel frame locale
        max_iterations: Massimo numero di iterazioni
        tolerance: Soglia di convergenza
        max_correspondence_distance: Distanza massima per corrispondenze

    Returns:
        Dictionary con risultati compatibili con il formato esistente
    """
    # Calcola trasformazione da odometria per inizializzazione
    R_odom, t_odom = compute_relative_transform_from_odometry(prev_pose, curr_pose)

    # ICP con inizializzazione da odometria
    result_init = icp_simple(
        src_local, tgt_local,
        init_R=R_odom, init_t=t_odom,
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_distance=max_correspondence_distance
    )

    # ICP senza inizializzazione (per confronto)
    result_raw = icp_simple(
        src_local, tgt_local,
        init_R=None, init_t=None,
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_distance=max_correspondence_distance
    )

    # Formato compatibile
    return {
        'ok': True,
        'k': 0,  # Sarà impostato dal chiamante
        'n_src': len(src_local),
        'n_tgt': len(tgt_local),
        'gt_R': R_odom,
        'gt_t': t_odom,
        'src_local': src_local,
        'tgt_local': tgt_local,
        'none': {  # Con inizializzazione
            'R': result_init['R'],
            't': result_init['t'],
            'alpha_rad': np.radians(result_init['angle_deg']),
            'alpha_deg': result_init['angle_deg'],
            'rmse': result_init['rmse'],
            'iterations': result_init['iterations'],
            'n_corr_last': result_init['n_correspondences'],
            'errors': result_init['errors'],
            'src_transformed': (result_init['R'] @ src_local.T).T + result_init['t'],
            'converged': result_init['converged']
        },
        'raw_none': {  # Senza inizializzazione
            'R': result_raw['R'],
            't': result_raw['t'],
            'alpha_rad': np.radians(result_raw['angle_deg']),
            'alpha_deg': result_raw['angle_deg'],
            'rmse': result_raw['rmse'],
            'iterations': result_raw['iterations'],
            'n_corr_last': result_raw['n_correspondences'],
            'errors': result_raw['errors'],
            'src_transformed': (result_raw['R'] @ src_local.T).T + result_raw['t'],
            'converged': result_raw['converged']
        }
    }


