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
import argparse
from icp import run_icp_over_history  # nuovo: esecuzione ICP su storia
import time  # per ETA nella barra di progresso
from tqdm import tqdm as _tqdm  # progress bar esterna con ETA

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
        if idx == 0:  # Rettilinea v costante: aumenta r_max e n_rays per avere più hit
            factor = 0.55
        elif idx == 1:  # Rettilinea v variabile
            factor = 0.40
        elif idx in (2, 3):  # circolari
            factor = 0.50
        elif idx == 4:  # otto
            factor = 0.45
        else:  # random walk
            factor = 0.55
        r_max = max(1.0, factor * diag)
        # Numero raggi: più densi per il caso 0
        if idx == 0:
            n_rays = 240
        elif idx == 1:
            n_rays = 180
        else:
            n_rays = 240
        lidar = Lidar(n_rays=n_rays, angle_span=2*math.pi, r_max=r_max, angle_offset=0.0, add_noise=False)
        lidars.append(lidar)
    return lidars


def main():
    parser = argparse.ArgumentParser(description="Simulatore traiettorie + salvatore immagini")
    parser.add_argument("--skip-collision", action="store_true", help="Salta il calcolo collisioni per avvio piu' rapido")
    parser.add_argument("--skip-viewer", action="store_true", help="Non aprire il viewer interattivo")
    parser.add_argument("--scan-interval", type=float, default=1.0, help="Intervallo tra scansioni LiDAR salvate [s]")
    parser.add_argument("--viewer-lidar-every", type=int, default=4, help="Aggiorna LiDAR nel viewer ogni N frame (default 4)")
    parser.add_argument("--run-icp", action="store_true", help="Esegui ICP su coppie (k-1,k) in frame locale e stampa confronto init=None vs init=odo")
    args = parser.parse_args()

    # Pulisci vecchie immagini per evitare accumulo: trajectories, scans, scans_polar
    try:
        visualizer.cleanup_output_images(subfolders=("trajectories", "scans", "scans_polar"), remove_root=False)
    except Exception as e:
        print(f"[main] Avviso: impossibile pulire cartelle immagini: {e}")

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

    # Passi per disegnare la posa del robot (in ordine dei casi)
    show_steps = [80, 80, 40, 40, 120, 120]

    # --------- Barra di progresso unica (tqdm) per tutti i salvataggi ---------
    # Calcola numero totale di immagini da salvare: traiettorie + (scans + polari) per ciascun caso
    step_idx = max(1, int(round(float(args.scan_interval) / max(1e-9, float(dt)))))
    total_steps = len(histories)  # una per traiettoria
    for hist in histories:
        N = int(len(hist))
        scans_count = (0 if N <= 0 else ((N - 1) // step_idx + 1))
        total_steps += 2 * scans_count  # scans punti + scans polari

    def _run_all_saves(cb):
        # Salva immagini di traiettoria (usa progress globale)
        visualizer.save_trajectories_images(
            histories, titles,
            show_orient_every=show_steps,
            environment=envs,
            fit_to='environment',
            progress_cb=cb,
            quiet=True,
        )
        # Salva scansioni (punti) e polari per ciascun caso (usa progress globale)
        for hist, title, env, lid in zip(histories, titles, envs, lidars):
            visualizer.save_lidar_scans_images(
                hist, title, lid, env, dt,
                interval_s=float(args.scan_interval),
                fit_to='environment',
                progress_cb=cb,
                quiet=True,
            )
            visualizer.save_lidar_polar_images(
                hist, title, lid, env, dt,
                interval_s=float(args.scan_interval),
                include_misses=True,
                progress_cb=cb,
                quiet=True,
            )

    if _tqdm is not None:
        with _tqdm(total=total_steps, desc="Salvataggio immagini", unit="img", ncols=90) as pbar:
            cb = lambda _cur, _tot: pbar.update(1)
            _run_all_saves(cb)
    else:
        # Fallback ASCII con ETA
        start_t = time.time()
        state = {"done": 0}
        width = 36
        def _eta(sec: float) -> str:
            m, s = divmod(int(round(max(0.0, sec))), 60)
            return f"{m:02d}:{s:02d}"
        def cb(_cur, _tot):
            state["done"] += 1
            done = min(state["done"], total_steps)
            frac = done / max(1, total_steps)
            filled = int(round(width * frac))
            bar = '#' * filled + '-' * (width - filled)
            elapsed = time.time() - start_t
            per_step = elapsed / max(1, done)
            remain = per_step * max(0, total_steps - done)
            print(f"\rSalvataggio immagini [{bar}] {done}/{total_steps}  ETA {_eta(remain)}", end='', flush=True)
            if done >= total_steps:
                print()
        _run_all_saves(cb)
    # --------- Fine barra di progresso unica ---------

    # Calcola collisioni via LiDAR solo se richiesto
    stop_indices = [None] * len(histories)
    stop_fractions = [None] * len(histories)
    if not args.skip_collision:
        stop_indices = []
        stop_fractions = []
        for hist, env, lid in zip(histories, envs, lidars):
            kcol, frac = _first_collision_via_lidar(hist, env, lid, body_length=0.40, body_width=0.20)
            stop_indices.append(kcol)
            stop_fractions.append(frac)

    # Mostra carosello (con raggi) solo se non saltato
    if not args.skip_viewer:
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
            lidar_every=int(max(1, args.viewer_lidar_every)),
         )

    # (Opzionale) Esegui ICP in frame locale per confrontare init=None vs init odometrica
    if args.run_icp:
        print("\n========== ICP (frame locale) ==========")
        # Legenda dei campi stampati
        print(
            "Legenda:\n"
            "- Coppia N: scansioni consecutive (k-1, k) nel frame locale del robot\n"
            "- errore medio (rmse) [init=None]: RMSE usando posa iniziale nulla\n"
            "- errore medio (rmse) [init=odometria]: RMSE usando posa iniziale da odometria\n"
            "- numero iterazioni ICP [..]: iterazioni eseguite dall'algoritmo ICP\n"
            "- angolo di rotazione alpha [..] (deg): rotazione stimata tra le due scansioni (in gradi)\n"
        )
        # Parametri ICP uniformi per tutti i casi (damping meno invasivo)
        trim_fraction = 0.7
        damping_enabled = True
        angle_thresh_deg = 10.0   # prima 7.5
        struct_ratio_thresh = 0.02  # prima 0.03: scatta meno spesso
        damp_factor = 0.7        # prima 0.5: riduzione più blanda
        for idx, (hist, title, env, lid) in enumerate(zip(histories, titles, envs, lidars)):
            print(f"\nCaso {idx+1}: {title}")
            icp_results = run_icp_over_history(
                hist, lid, env,
                step=1,
                max_iterations=40,
                tolerance=1e-5,
                max_correspondence_distance=0.5,
                use_scipy=True,
                trim_fraction=trim_fraction,
                damping_enabled=damping_enabled,
                angle_thresh_deg=angle_thresh_deg,
                struct_ratio_thresh=struct_ratio_thresh,
                damp_factor=damp_factor,
            )
            for res in icp_results[0:5]:
                if not res.get('ok', False):
                    print(f" Coppia {res['k']:4d}: punti insufficienti (src={res.get('n_src')}, tgt={res.get('n_tgt')})")
                    continue
                rn = res['none']; ro = res['odo']
                print(
                    f" Coppia {res['k']:4d}: "
                    f"rmse[None]={rn['rmse']:.4f}, rmse[Odo]={ro['rmse']:.4f}, "
                    f"it[None]={rn['iterations']}, it[Odo]={ro['iterations']}, "
                    f"alpha[None]={rn['alpha_deg']:.4f} deg, alpha[Odo]={ro['alpha_deg']:.4f} deg"
                )


if __name__ == "__main__":
    main()
