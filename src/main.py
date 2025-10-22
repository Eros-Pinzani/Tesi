from robot import Robot
from trajectory_generator import TrajectoryGenerator
from simulator import Simulator
import visualizer
from visualizer import _interp_pose  # import esplicito per uso in bisezione collisione
import math  # Per calcolo di 2πR/v
from environment import Environment  # per visualizzare bounds e ostacoli
import numpy as np  # per calcolare bounds dalle traiettorie
from typing import List, Optional, Tuple
from shapely.geometry import Polygon  # per collisione rettangolare
from messages import COLLISION_TRAJECTORY
from environment_presets import setup_environments_per_trajectory


def build_simulator() -> Simulator:
    """Crea un simulatore con un robot di default."""
    return Simulator(robot=Robot())


def reset_robot_default(sim: Simulator, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
    """Reimposta il robot del simulatore alla posa iniziale di default (x,y,theta)."""
    sim.reset_robot(x=x, y=y, theta=theta)


def _rect_polygon_from_pose(pose, body_length: float, body_width: float) -> Polygon:
    """Costruisce il poligono rettangolare orientato del robot a partire da (x,y,theta)."""
    x, y, theta = map(float, pose)
    L = float(body_length)
    W = float(body_width)
    hx, hy = 0.5 * L, 0.5 * W
    local = np.array([[+hx, +hy], [-hx, +hy], [-hx, -hy], [+hx, -hy]], dtype=float)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([[c, -s], [s, c]], dtype=float)
    world = (local @ R.T) + np.array([x, y], dtype=float)
    return Polygon(world.tolist())


def _first_collision_along_segment(p0, p1, env: Environment, body_length: float, body_width: float, *, iters: int = 18) -> Optional[float]:
    """Se p1 collide e p0 no, trova via bisezione la frazione alpha in (0,1] del primo contatto.
    Ritorna None se non c'è collisione nel segmento."""
    union = env.obstacles_union()
    if union is None:
        return None
    def collides(pose) -> bool:
        return _rect_polygon_from_pose(pose, body_length, body_width).intersects(union)
    c0 = collides(p0)
    c1 = collides(p1)
    if not c1:
        return None
    if c0:
        return 0.0  # già in collisione all'inizio
    lo, hi = 0.0, 1.0
    for _ in range(max(1, int(iters))):
        mid = 0.5 * (lo + hi)
        pose_mid = _interp_pose(p0, p1, mid)
        if collides(pose_mid):
            hi = mid
        else:
            lo = mid
    return hi


def _find_collision_index_with_fraction(history: np.ndarray, env: Environment, body_length: float = 0.40, body_width: float = 0.20) -> Tuple[Optional[int], Optional[float]]:
    """Ritorna (k, alpha) dove k è il primo indice di frame in collisione e alpha la frazione nel segmento (k-1,k].
    Se collisione al frame 0, ritorna (0, 0.0). Se nessuna collisione, (None, None)."""
    union = env.obstacles_union()
    if union is None:
        return None, None
    # Check frame 0
    if _rect_polygon_from_pose(history[0], body_length, body_width).intersects(union):
        return 0, 0.0
    # Cerca il primo frame k che collide
    for k in range(1, len(history)):
        if _rect_polygon_from_pose(history[k], body_length, body_width).intersects(union):
            # Bisezione tra k-1 e k per frazione precisa
            alpha = _first_collision_along_segment(history[k-1], history[k], env, body_length, body_width)
            if alpha is None:
                alpha = 1.0
            return k, float(alpha)
    return None, None


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

    # Calcola indice e frazione del primo impatto per ogni traiettoria (rettangolo orientato)
    stop_indices: List[Optional[int]] = []
    stop_fractions: List[Optional[float]] = []
    error_messages: List[Optional[str]] = []
    for hist, env in zip(histories, envs):
        kcol, frac = _find_collision_index_with_fraction(hist, env, body_length=0.40, body_width=0.20)
        stop_indices.append(kcol)
        stop_fractions.append(frac)
        error_messages.append(COLLISION_TRAJECTORY if kcol is not None else None)

    # Passi per disegnare la posa del robot (in ordine dei casi)
    show_steps = [80, 80, 40, 40, 120, 120]

    # Salva immagini complete (nessun overlay errore)
    visualizer.save_trajectories_images(histories, titles, show_orient_every=show_steps, environment=envs, fit_to='environment')

    # Mostra carosello: riproduce e si ferma alla collisione mostrando il messaggio
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
        error_messages=error_messages,
        stop_indices=stop_indices,
        stop_fractions=stop_fractions,
    )


if __name__ == "__main__":
    main()
