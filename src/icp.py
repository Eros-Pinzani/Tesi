"""
ICP Point-to-Point - Implementazione Pulita e Robusta
Algoritmo ICP semplificato senza filtri complicati, solo le funzionalità essenziali.
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
        rotation_matrix: Matrice di rotazione 2x2
        translation_vector: Vettore traslazione 2D
    """
    # Estrae le coordinate e gli angoli dalle pose
    x_prev, y_prev, theta_prev = prev_pose[:3]
    x_curr, y_curr, theta_curr = curr_pose[:3]

    # Calcola la differenza angolare tra le due pose
    d_theta = theta_curr - theta_prev

    # Costruisce la matrice di rotazione relativa
    cos_dt = np.cos(d_theta)
    sin_dt = np.sin(d_theta)
    rotation_matrix = np.array([[cos_dt, -sin_dt],
                                [sin_dt, cos_dt]], dtype=np.float64)

    # Calcola la traslazione nel frame di riferimento precedente
    cos_prev = np.cos(theta_prev)
    sin_prev = np.sin(theta_prev)

    # Calcola lo spostamento nel frame globale
    dx_world = x_curr - x_prev
    dy_world = y_curr - y_prev

    # Trasforma lo spostamento nel frame locale della pose precedente
    t_x = cos_prev * dx_world + sin_prev * dy_world
    t_y = -sin_prev * dx_world + cos_prev * dy_world

    translation_vector = np.array([t_x, t_y], dtype=np.float64)

    return rotation_matrix, translation_vector


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
    # Verifica che entrambi gli array abbiano punti
    if len(source) == 0 or len(target) == 0:
        return np.array([]), np.array([])

    # Liste per raccogliere le corrispondenze valide
    source_matched = []
    target_matched = []

    # Per ogni punto source, trova il punto target più vicino
    for i in range(len(source)):
        # Calcola le distanze euclidee da questo punto a tutti i target
        distances = np.linalg.norm(target - source[i], axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        # Aggiunge la corrispondenza solo se la distanza è sotto la soglia
        if min_dist < max_distance:
            source_matched.append(source[i])
            target_matched.append(target[min_idx])

    # Se non ci sono corrispondenze valide, restituisce array vuoti
    if len(source_matched) == 0:
        return np.array([]), np.array([])

    return np.array(source_matched), np.array(target_matched)


def compute_transformation_svd(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola la trasformazione ottimale usando SVD.
    Trova rotation_matrix e translation_vector tali che: target ≈ rotation_matrix * source + translation_vector

    Args:
        source: Array Nx2 di punti source
        target: Array Nx2 di punti target (corrispondenze)

    Returns:
        rotation_matrix: Matrice di rotazione 2x2
        translation_vector: Vettore traslazione 2D
    """
    # Calcola i centroidi dei due insiemi di punti
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)

    # Centra i punti rispetto ai loro centroidi
    source_centered = source - centroid_source
    target_centered = target - centroid_target

    # Costruisce la matrice di covarianza H = target^T * source
    h_matrix = target_centered.T @ source_centered

    # Decomposizione SVD della matrice di covarianza
    u_matrix, _, vt_matrix = np.linalg.svd(h_matrix)
    rotation_matrix = u_matrix @ vt_matrix

    # Assicura che sia una rotazione propria (det = +1) e non una riflessione
    if np.linalg.det(rotation_matrix) < 0:
        u_matrix[:, -1] *= -1
        rotation_matrix = u_matrix @ vt_matrix

    # Calcola la traslazione: t = centroid_target - R * centroid_source
    translation_vector = centroid_target - rotation_matrix @ centroid_source

    return rotation_matrix, translation_vector


def compute_rmse(source: np.ndarray, target: np.ndarray, rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> float:
    """
    Calcola l'RMSE dopo aver applicato la trasformazione.

    Args:
        source: Punti source Nx2
        target: Punti target Nx2
        rotation_matrix: Rotazione 2x2
        translation_vector: Traslazione 2D

    Returns:
        RMSE
    """
    # Se non ci sono punti, restituisce infinito
    if len(source) == 0:
        return float('inf')

    # Applica la trasformazione ai punti source
    transformed = (rotation_matrix @ source.T).T + translation_vector

    # Calcola le distanze euclidee tra punti trasformati e target
    errors = np.linalg.norm(transformed - target, axis=1)

    # Restituisce la radice quadrata della media dei quadrati degli errori
    return np.sqrt(np.mean(errors ** 2))


def icp(source: np.ndarray,
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
    # Inizializza la matrice di rotazione
    if init_R is None:
        rotation_matrix = np.eye(2, dtype=np.float64)
    else:
        rotation_matrix = np.array(init_R, dtype=np.float64).copy()

    # Inizializza il vettore di traslazione
    if init_t is None:
        translation_vector = np.zeros(2, dtype=np.float64)
    else:
        translation_vector = np.array(init_t, dtype=np.float64).copy()

    # Applica la trasformazione iniziale ai punti source
    source_transformed = (rotation_matrix @ source.T).T + translation_vector

    # Inizializza le variabili per il loop iterativo
    prev_rmse = float('inf')
    errors_history = []
    iteration = 0
    n_correspondences = 0

    # Loop principale ICP
    for iteration in range(max_iterations):
        # Passo 1: Trova le corrispondenze tra source trasformato e target
        source_matched, target_matched = find_nearest_neighbors(
            source_transformed, target, max_correspondence_distance
        )

        # Verifica che ci siano abbastanza corrispondenze per calcolare la trasformazione
        if len(source_matched) < 3:
            break

        n_correspondences = len(source_matched)

        # Passo 2: Calcola la trasformazione ottimale usando SVD
        rotation_iter, translation_iter = compute_transformation_svd(source_matched, target_matched)

        # Passo 3: Applica la trasformazione ai punti source
        source_transformed = (rotation_iter @ source_transformed.T).T + translation_iter

        # Passo 4: Aggiorna la trasformazione cumulativa
        rotation_matrix = rotation_iter @ rotation_matrix
        translation_vector = rotation_iter @ translation_vector + translation_iter

        # Passo 5: Calcola l'RMSE per monitorare la convergenza
        rmse = compute_rmse(source_matched, target_matched, rotation_iter, translation_iter)
        errors_history.append(rmse)

        # Passo 6: Verifica la convergenza
        if abs(prev_rmse - rmse) < tolerance:
            break

        prev_rmse = rmse

    # Calcola l'angolo di rotazione finale dalla matrice
    angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    angle_deg = np.degrees(angle_rad)

    # Restituisce i risultati in un dizionario
    return {
        'R': rotation_matrix,
        't': translation_vector,
        'iterations': iteration + 1,
        'rmse': prev_rmse if prev_rmse != float('inf') else 0.0,
        'angle_deg': angle_deg,
        'errors': np.array(errors_history),
        'n_correspondences': n_correspondences,
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
    # Calcola la trasformazione da odometria per l'inizializzazione
    r_odom, t_odom = compute_relative_transform_from_odometry(prev_pose, curr_pose)

    # Esegue ICP FILTRATO con inizializzazione da odometria (più robusto)
    result_filtered = icp(
        src_local, tgt_local,
        init_R=r_odom, init_t=t_odom,
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_distance=max_correspondence_distance
    )

    # Esegue ICP RAW con inizializzazione a identità (meno robusto)
    result_raw = icp(
        src_local, tgt_local,
        init_R=np.eye(2), init_t=np.zeros(2),
        max_iterations=max_iterations,
        tolerance=tolerance,
        max_correspondence_distance=max_correspondence_distance
    )

    # Prepara il dizionario di output nel formato compatibile con il codice esistente
    return {
        'ok': True,
        'k': 0,  # Verrà impostato dal chiamante
        'n_src': len(src_local),
        'n_tgt': len(tgt_local),
        'gt_R': r_odom,
        'gt_t': t_odom,
        'src_local': src_local,
        'tgt_local': tgt_local,
        'none': {  # Risultati ICP filtrato
            'R': result_filtered['R'],
            't': result_filtered['t'],
            'alpha_rad': np.radians(result_filtered['angle_deg']),
            'alpha_deg': result_filtered['angle_deg'],
            'rmse': result_filtered['rmse'],
            'iterations': result_filtered['iterations'],
            'n_corr_last': result_filtered['n_correspondences'],
            'errors': result_filtered['errors'],
            'src_transformed': (result_filtered['R'] @ src_local.T).T + result_filtered['t'],
            'converged': result_filtered['converged']
        },
        'raw_none': {  # Risultati ICP raw
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

def run_icp_over_history(history, lidar, env, step=1):
    """
    Wrapper per compatibilità con visualizer.py

    Questa funzione esiste per mantenere la compatibilità con il codice legacy.
    Le traiettorie ICP vengono calcolate in main.py e passate direttamente al viewer.
    """
    # Converte la history in array numpy
    history = np.asarray(history, dtype=float)
    n = len(history)

    # Inizializza le traiettorie con la stessa dimensione della history
    traj_init = np.zeros((n, 3), dtype=float)
    traj_raw = np.zeros((n, 3), dtype=float)

    # Imposta la prima pose uguale per entrambe le traiettorie
    traj_init[0] = history[0].copy()
    traj_raw[0] = history[0].copy()

    # Loop semplificato per compatibilità
    for k in range(step, n, step):
        # Estrae le pose consecutive
        prev_pose = history[k-step]
        curr_pose = history[k]

        # Acquisisce gli scan nelle pose correnti
        prev_scan = lidar.scan_hits(prev_pose, env, frame='local')
        curr_scan = lidar.scan_hits(curr_pose, env, frame='local')

        # Verifica che ci siano abbastanza punti negli scan
        if len(prev_scan) < 10 or len(curr_scan) < 10:
            # Se non ci sono abbastanza punti, copia semplicemente la pose corrente
            traj_init[k] = curr_pose.copy()
            traj_raw[k] = curr_pose.copy()
            continue

        # Copia la pose corrente (nessun calcolo ICP effettivo)
        traj_init[k] = curr_pose.copy()
        traj_raw[k] = curr_pose.copy()

    return traj_init, traj_raw