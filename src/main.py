from robot import Robot
from trajectory_generator import TrajectoryGenerator
from simulator import Simulator
import visualizer
from visualizer import _interp_pose  # import esplicito per uso in bisezione collisione
import math  # Per calcolo di 2πR/v
from environment import Environment  # per visualizzare bounds e ostacoli
import numpy as np  # per calcolare bounds dalle traiettorie
from typing import List, Optional, Tuple
from environment_presets import setup_environments_per_trajectory
from lidar import Lidar  # sensore LiDAR


def build_simulator() -> Simulator:
    """Crea un simulatore con un robot di default."""
    return Simulator(robot=Robot())


def reset_robot_default(sim: Simulator, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
    """Reimposta il robot del simulatore alla posa iniziale di default (x,y,theta)."""
    sim.reset_robot(x=x, y=y, theta=theta)


# ------------------ Collisione via LiDAR ------------------

def _support_distance_rect(delta: float, a: float, b: float) -> float:
    """Distanza dal centro alla frontiera di un rettangolo axis-aligned (semiassi a=half-length, b=half-width)
    lungo la direzione con angolo delta nel frame del robot. Formula del supporto: a*|cos δ| + b*|sin δ|."""
    c = abs(math.cos(delta))
    s = abs(math.sin(delta))
    return a * c + b * s


def _lidar_clearance_measure(pose, lidar: Lidar, env: Environment, body_length: float, body_width: float) -> float:
    """Ritorna la minima differenza (range - supporto_rettangolo) sui raggi del LiDAR per la posa.
    Se <= 0 si considera contatto (il corpo tocca l'ostacolo)."""
    # semi-dimensioni del rettangolo corpo (metri)
    a = 0.5 * float(body_length)
    b = 0.5 * float(body_width)
    # Angoli relativi dei raggi nel frame del robot (come in Lidar.scan)
    half = 0.5 * float(lidar.angle_span)
    rel_angles = np.linspace(-half, half, num=lidar.n_rays, endpoint=True)
    # Scansione attuale
    _pts, ranges = lidar.scan(pose, env, return_ranges=True)
    # Misura di clearance: range meno distanza bordo corpo su ciascun raggio
    supports = np.array([_support_distance_rect(float(da), a, b) for da in rel_angles], dtype=float)
    diffs = ranges - supports
    return float(np.min(diffs))


def _first_collision_via_lidar(history: np.ndarray, env: Environment, lidar: Lidar, *, body_length: float = 0.40, body_width: float = 0.20, iters: int = 14) -> Tuple[Optional[int], Optional[float]]:
    """Trova primo contatto via LiDAR lungo la storia: ritorna (k, alpha) con k il primo indice in cui c'è contatto
    e alpha la frazione in (k-1,k] in cui la misura di clearance attraversa 0 (bisezione su pose interpolate).
    Se contatto a frame 0: (0, 0.0). Se nessun contatto: (None, None)."""
    N = len(history)
    if N <= 0:
        return None, None
    # Misura iniziale
    m0 = _lidar_clearance_measure(history[0], lidar, env, body_length, body_width)
    if m0 <= 0.0:
        return 0, 0.0
    # Cerca primo frame con misura <= 0
    k_hit = None
    for k in range(1, N):
        mk = _lidar_clearance_measure(history[k], lidar, env, body_length, body_width)
        if mk <= 0.0:
            k_hit = k
            break
    if k_hit is None:
        return None, None
    # Bisezione tra (k-1, k]
    lo, hi = 0.0, 1.0
    p0 = history[k_hit - 1]
    p1 = history[k_hit]
    for _ in range(max(1, int(iters))):
        mid = 0.5 * (lo + hi)
        pose_mid = _interp_pose(p0, p1, mid)
        mm = _lidar_clearance_measure(pose_mid, lidar, env, body_length, body_width)
        if mm <= 0.0:
            hi = mid
        else:
            lo = mid
    return int(k_hit), float(hi)


# ------------------ Fine collisione via LiDAR ------------------


def _env_bounds_diag(env: Environment) -> float:
    try:
        x0, y0, x1, y1 = env.bounds.bounds  # type: ignore[union-attr]
        w = float(x1 - x0)
        h = float(y1 - y0)
        return float((w*w + h*h) ** 0.5)
    except Exception:
        return 10.0


def _build_lidars_for_cases(envs: List[Environment], titles: List[str]) -> List[Lidar]:
    """Crea una lista di Lidar per singolo caso con r_max adattivo per non coprire sempre tutti gli ostacoli.
    Strategia: r_max = fattore * diagonale dei bounds, con fattori più piccoli per i casi rettilinei."""
    lidars: List[Lidar] = []
    for idx, (env, title) in enumerate(zip(envs, titles)):
        diag = _env_bounds_diag(env)
        # Fattori per caso: più conservativi sui rettilinei
        if idx in (0, 1):  # rettilinei
            factor = 0.35
        elif idx in (2, 3):  # circolari
            factor = 0.50
        elif idx == 4:  # otto
            factor = 0.45
        else:  # random walk
            factor = 0.55
        r_max = max(1.0, factor * diag)
        # Numero raggi: meno densi per rettilinei per rendere meno capillare
        n_rays = 160 if idx in (0, 1) else 240
        lidar = Lidar(n_rays=n_rays, angle_span=2*math.pi, r_max=r_max, angle_offset=0.0, add_noise=False)
        lidars.append(lidar)
    return lidars


def main():
    dt = 0.05       # Passo temporale di integrazione (Eulero)

    # Parametri base di riferimento
    v_ref = 0.5
    radius_ref = 2.0
    v_min_ref = 0.2
    v_max_ref = 0.8
    omega_std_ref = 0.5

    tg = TrajectoryGenerator()                 # Generatore delle traiettorie
    sim = build_simulator()                    # Simulatore con robot iniziale di default

    histories = []      # Lista delle storie [x,y,theta] per ogni traiettoria (complete)
    titles = []         # Titoli da mostrare nel carosello
    commands_list = []  # Lista parallela dei comandi (v, omega) per ogni traiettoria (complete)

    # 1) Rettilinea (v costante)
    T_straight = 20.0
    v = v_ref
    vs, omegas = tg.straight(v=v, T=T_straight, dt=dt)
    reset_robot_default(sim)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Rettilinea (v costante)")

    # 2) Rettilinea (v variabile)
    T_straight_var = 20.0
    v_min, v_max = v_min_ref, v_max_ref
    vs, omegas = tg.straight_var_speed(v_min=v_min, v_max=v_max, T=T_straight_var, dt=dt, phase=0.0)
    reset_robot_default(sim)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Rettilinea (v variabile)")

    # 3) Circolare (v costante) — 1 giro intero
    v = v_ref
    R = radius_ref
    period = (2.0 * math.pi * R) / max(v, 1e-9)
    n_steps = max(1, int(round(period / dt)))
    T_circle = n_steps * dt
    vs, omegas = tg.circle(v=v, radius=R, T=T_circle, dt=dt)
    reset_robot_default(sim)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Circolare (v costante)")

    # 4) Circolare (v variabile) — 1 giro intero
    v_min, v_max = v_min_ref, v_max_ref
    v_mid = 0.5 * (v_min + v_max)
    period_var = (2.0 * math.pi * R) / max(v_mid, 1e-9)
    n_steps_var = max(1, int(round(period_var / dt)))
    T_circle_var = n_steps_var * dt
    vs, omegas = tg.circle_var_speed(v_min=v_min, v_max=v_max, radius=R, T=T_circle_var, dt=dt, phase=0.0)
    reset_robot_default(sim)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Circolare (v variabile)")

    # 5) Traiettoria a 8 — ciclo completo
    v = v_ref
    period_eight = (4.0 * math.pi * R) / max(v, 1e-9)
    n_steps_eight = max(2, int(round(period_eight / dt)))
    if n_steps_eight % 2 == 1:
        n_steps_eight += 1
    T_eight = (n_steps_eight - 1e-9) * dt
    vs, omegas = tg.eight(v=v, radius=R, T=T_eight, dt=dt)
    reset_robot_default(sim)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Traiettoria a 8")

    # 6) Random walk
    T_rw = 40.0
    v_mean = v_ref
    omega_std = omega_std_ref
    vs, omegas = tg.random_walk(v_mean=v_mean, omega_std=omega_std, T=T_rw, dt=dt, seed=42)
    reset_robot_default(sim)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Random walk")

    # Costruisci ambienti specifici per ciascuna traiettoria (usando le storie complete)
    envs = setup_environments_per_trajectory(histories, titles)

    # Istanzia LiDAR per-caso con portata adattiva
    lidars = _build_lidars_for_cases(envs, titles)

    # Calcola indice e frazione del primo impatto via LiDAR per ogni traiettoria
    stop_indices: List[Optional[int]] = []
    stop_fractions: List[Optional[float]] = []
    for hist, env, lid in zip(histories, envs, lidars):
        kcol, frac = _first_collision_via_lidar(hist, env, lid, body_length=0.40, body_width=0.20)
        stop_indices.append(kcol)
        stop_fractions.append(frac)

    # Passi per disegnare la posa del robot (in ordine dei casi)
    show_steps = [80, 80, 40, 40, 120, 120]

    # Salva immagini complete (nessun overlay errore)
    visualizer.save_trajectories_images(histories, titles, show_orient_every=show_steps, environment=envs, fit_to='environment')

    # Salva scansioni ogni 2 secondi per ciascuna traiettoria usando i lidar per-caso
    for hist, title, env, lid in zip(histories, titles, envs, lidars):
        visualizer.save_lidar_scans_images(hist, title, lid, env, dt, interval_s=2.0, fit_to='environment')

    # Mostra carosello con raggi e stop su collisione (via LiDAR)
    visualizer.show_trajectories_carousel(
        histories,
        titles,
        show_orient_every=show_steps,
        save_each=False,
        commands_list=commands_list,
        dts=dt,
        show_info=True,
        environment=envs,
        fit_to='environment',
        stop_indices=stop_indices,
        stop_fractions=stop_fractions,
        lidar=lidars,
        show_lidar=True,
    )


if __name__ == "__main__":
    main()
