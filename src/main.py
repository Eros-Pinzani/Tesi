from robot import Robot
from trajectory_generator import TrajectoryGenerator
from simulator import Simulator
import visualizer
from environment_presets import setup_environments_per_trajectory
from lidar import Lidar  # sensore LiDAR
import argparse
from icp import run_icp_over_history  # nuovo: esecuzione ICP su storia
from icp import relative_local_transform  # nuovo: GT relativo per confronto pose
import time  # per ETA nella barra di progresso
from tqdm import tqdm as _tqdm  # progress bar esterna con ETA
# import sys  # per rilevare TTY e usare bold ANSI (non più necessario, grassetto forzato)
from icp_plots import (
    save_concept_correspondences,
    save_alignment_overlays,
    save_convergence_curves,
    save_motion_arrows,
    save_raw_vs_filtered,
)
from typing import List, Optional, Tuple
from environment import Environment
import re
import math
import numpy as np
import sys, os
import datetime as _dt
# Nuovo: inizializza colorama per garantire rendering ANSI su Windows
try:
    from colorama import init as _colorama_init
except ImportError:
    _colorama_init = None
else:
    _colorama_init()

# Helper slugify locale (evita warning su uso di funzione privata) e precompila regex
_slugify_re = re.compile(r'[^a-z0-9_\-]')

# Tee per duplicare stdout/stderr su file e console
class _Tee:
    def __init__(self, primary, secondary):
        self._primary = primary  # tipicamente console (già wrappata da colorama)
        self._secondary = secondary  # file
        # Propaga alcune proprietà utili
        self.encoding = getattr(primary, 'encoding', 'utf-8')

    def write(self, data):
        try:
            self._primary.write(data)
        finally:
            self._secondary.write(data)
        return len(data)

    def flush(self):
        try:
            self._primary.flush()
        finally:
            self._secondary.flush()

    def isatty(self):
        return bool(getattr(self._primary, 'isatty', lambda: False)())

    def fileno(self):
        if hasattr(self._primary, 'fileno'):
            return self._primary.fileno()
        raise OSError('fileno non disponibile')


def _slugify_local(text: str) -> str:
    base = (text or '').lower().strip() or 'case'
    base = re.sub(r'\s+', '_', base)
    return _slugify_re.sub('', base)


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


def _lidar_clearance_measure(pose: np.ndarray, lidar: Lidar, env: Environment, body_length: float, body_width: float) -> float:
    """Ritorna la minima differenza (range - supporto_rettangolo) sui raggi del LiDAR per la posa.
    Se <= 0 si considera contatto (il corpo tocca l'ostacolo)."""
    # semi-dimensioni del rettangolo corpo (metri)
    a = 0.5 * float(body_length)
    b = 0.5 * float(body_width)
    # Angoli relativi dei raggi nel frame del robot (come in Lidar.scan)
    half = 0.5 * float(lidar.angle_span)
    rel_angles = np.linspace(-half, half, num=lidar.n_rays, endpoint=True)
    # Scansione attuale
    _, ranges = lidar.scan(pose, env, return_ranges=True)
    # Misura di clearance: range meno distanza bordo corpo su ciascun raggio
    supports = np.array([_support_distance_rect(float(da), a, b) for da in rel_angles], dtype=float)
    diffs = ranges - supports
    return float(np.min(diffs))


def _first_collision_via_lidar(history: np.ndarray, env: Environment, lidar: Lidar, *, body_length: float = 0.40, body_width: float = 0.20, iters: int = 14) -> Tuple[Optional[int], Optional[float]]:
    """Trova primo contatto via LiDAR lungo la storia: ritorna (k, alpha) con k il primo indice in cui c'è contatto
    e alpha la frazione in (k-1,k] in cui la misura di clearance attraversa 0 (bisezione su pose interpolate).
    Se contatto a frame 0: (0, 0.0). Se nessun contatto: (None, None)."""
    n = len(history)
    if n <= 0:
        return None, None
    # Misura iniziale
    m0 = _lidar_clearance_measure(history[0], lidar, env, body_length, body_width)
    if m0 <= 0.0:
        return 0, 0.0
    # Cerca primo frame con misura <= 0
    k_hit = None
    for k in range(1, n):
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
        pose_mid = _interp_pose_local(p0, p1, mid)
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
    except (AttributeError, TypeError, ValueError):
        return 10.0


def _build_lidars_for_cases(envs: List[Environment], titles: List[str]) -> List[Lidar]:
    """Crea una lista di Lidar per singolo caso con r_max adattivo per non coprire sempre tutti gli ostacoli.
    Strategia: r_max = fattore * diagonale dei bounds, con fattori più piccoli per i casi rettilinei."""
    lidars: List[Lidar] = []
    for idx, (env, _unused_title) in enumerate(zip(envs, titles)):
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
        # Micro-ritocchi: più copertura e densità raggi per casi 4 (idx==3) e 5 (idx==4)
        if idx in (3, 4):
            factor = 0.60
        r_max = max(1.0, factor * diag)
        # Numero raggi: più densi per i casi 3 e 4
        if idx == 0:
            n_rays = 240
        elif idx == 1:
            n_rays = 180
        elif idx in (3, 4):
            n_rays = 300
        else:
            n_rays = 240
        lidar = Lidar(n_rays=n_rays, angle_span=2*math.pi, r_max=r_max, angle_offset=0.0, add_noise=False)
        lidars.append(lidar)
    return lidars


def _interp_pose_local(p0: np.ndarray, p1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolazione lineare (x,y,theta) con wrapping di theta in [-pi,pi)."""
    a = float(max(0.0, min(1.0, alpha)))
    x0, y0, t0 = map(float, p0)
    x1, y1, t1 = map(float, p1)
    dx = x1 - x0
    dy = y1 - y0
    dth = (t1 - t0 + math.pi) % (2.0 * math.pi) - math.pi
    x = x0 + a * dx
    y = y0 + a * dy
    th = t0 + a * dth
    th = (th + math.pi) % (2.0 * math.pi) - math.pi
    return np.array([x, y, th], dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Simulatore traiettorie + salvatore immagini")
    parser.add_argument("--skip-collision", action="store_true", help="Salta il calcolo collisioni per avvio piu' rapido")
    parser.add_argument("--skip-viewer", action="store_true", help="Non aprire il viewer interattivo")
    parser.add_argument("--scan-interval", type=float, default=1.0, help="Intervallo tra scansioni LiDAR salvate [s]")
    parser.add_argument("--viewer-lidar-every", type=int, default=4, help="Aggiorna LiDAR nel viewer ogni N frame (default 4)")
    parser.add_argument("--run-icp", action="store_true", help="Esegui ICP su coppie (k-1,k) in frame locale e stampa confronto ICP filtrato vs RAW")
    parser.add_argument("--viewer-icp-grid", action="store_true", help="[DEPRECATO] Usa --viewer-mode grid al posto di questo flag")
    parser.add_argument("--skip-icp", action="store_true", help="Non eseguire l'ICP prima dell'apertura del viewer")
    parser.add_argument("--viewer-mode", choices=["grid","carousel"], default="carousel", help="Seleziona viewer: 'grid' (ICP a 5 pannelli) o 'carousel' (standard) - default: carousel")
    parser.add_argument("--quiet", action="store_true", help="Se presente sopprime stampe durante salvataggio immagini (default: stampe attive)")
    parser.add_argument("--no-icp-verbose", dest="icp_verbose", action="store_false", help="Disabilita stampe dettagliate ICP durante l'esecuzione principale")
    parser.set_defaults(icp_verbose=True)
    args = parser.parse_args()

    # Attiva tee su file per duplicare l'output della console in un .txt della sessione
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    _log_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    try:
        os.makedirs(_log_dir, exist_ok=True)
    except OSError:
        # fallback alla root del progetto se non riesce a creare logs/
        _log_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
    _ts = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    _log_path = os.path.join(_log_dir, f"run_output_{_ts}.txt")
    _log_file = open(_log_path, 'w', encoding='utf-8', newline='')
    sys.stdout = _Tee(_orig_stdout, _log_file)
    sys.stderr = _Tee(_orig_stderr, _log_file)
    print(f"[log] Duplico l'output della console su: {_log_path}")

    try:
        # Compatibilità: se viene passato il flag deprecato, forza viewer_mode a 'grid'
        if getattr(args, "viewer_icp_grid", False):
            print("[AVVISO] --viewer-icp-grid è deprecato; usa --viewer-mode grid. Forzo viewer_mode=grid.")
            args.viewer_mode = "grid"

        # Pulisci vecchie immagini per evitare accumulo: trajectories, scans, scans_polar
        try:
            visualizer.cleanup_output_images(subfolders=("trajectories", "scans", "scans_polar", "icp"), remove_root=False)
        except OSError as e:
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
        t_straight = 20.0  # durata rettilinea costante
        v = v_ref
        vs, omegas = tg.straight(v=v, T=t_straight, dt=dt)
        reset_robot_default(sim)
        histories.append(sim.run_from_sequence(vs, omegas, dt))
        commands_list.append(sim.commands)
        titles.append("Rettilinea (v costante)")

        # 2) Rettilinea (v variabile)
        t_straight_var = 20.0  # durata rettilinea a velocita' variabile
        v_min, v_max = v_min_ref, v_max_ref
        vs, omegas = tg.straight_var_speed(v_min=v_min, v_max=v_max, T=t_straight_var, dt=dt, phase=0.0)
        reset_robot_default(sim)
        histories.append(sim.run_from_sequence(vs, omegas, dt))
        commands_list.append(sim.commands)
        titles.append("Rettilinea (v variabile)")

        # 3) Circolare (v costante) — 1 giro intero
        v = v_ref
        r_ref = radius_ref
        period = (2.0 * math.pi * r_ref) / max(v, 1e-9)
        n_steps = max(1, int(round(period / dt)))
        t_circle = n_steps * dt
        vs, omegas = tg.circle(v=v, radius=r_ref, T=t_circle, dt=dt)
        reset_robot_default(sim)
        histories.append(sim.run_from_sequence(vs, omegas, dt))
        commands_list.append(sim.commands)
        titles.append("Circolare (v costante)")

        # 4) Circolare (v variabile) — 1 giro intero
        v_min, v_max = v_min_ref, v_max_ref
        v_mid = 0.5 * (v_min + v_max)
        period_var = (2.0 * math.pi * r_ref) / max(v_mid, 1e-9)
        n_steps_var = max(1, int(round(period_var / dt)))
        t_circle_var = n_steps_var * dt
        vs, omegas = tg.circle_var_speed(v_min=v_min, v_max=v_max, radius=r_ref, T=t_circle_var, dt=dt, phase=0.0)
        reset_robot_default(sim)
        histories.append(sim.run_from_sequence(vs, omegas, dt))
        commands_list.append(sim.commands)
        titles.append("Circolare (v variabile)")

        # 5) Traiettoria a 8 — ciclo completo
        v = v_ref
        period_eight = (4.0 * math.pi * r_ref) / max(v, 1e-9)
        n_steps_eight = max(2, int(round(period_eight / dt)))
        if n_steps_eight % 2 == 1:
            n_steps_eight += 1
        t_eight = (n_steps_eight - 1e-9) * dt
        vs, omegas = tg.eight(v=v, radius=r_ref, T=t_eight, dt=dt)
        reset_robot_default(sim)
        histories.append(sim.run_from_sequence(vs, omegas, dt))
        commands_list.append(sim.commands)
        titles.append("Traiettoria a 8")

        # 6) Random walk
        t_rw = 40.0  # durata random walk
        v_mean = v_ref
        omega_std = omega_std_ref
        vs, omegas = tg.random_walk(v_mean=v_mean, omega_std=omega_std, T=t_rw, dt=dt, seed=42)
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
            n = int(len(hist))
            scans_count = (0 if n <= 0 else ((n - 1) // step_idx + 1))
            total_steps += 2 * scans_count  # scans punti + scans polari

        def _run_all_saves(progress_cb_fn):
            # Salva immagini di traiettoria (usa progress globale)
            visualizer.save_trajectories_images(
                histories, titles,
                show_orient_every=show_steps,
                environment=envs,
                fit_to='environment',
                progress_cb=progress_cb_fn,
                quiet=args.quiet,
            )
            # Salva scansioni (punti) e polari per ciascun caso (usa progress globale)
            for save_hist, save_title, save_env, save_lid in zip(histories, titles, envs, lidars):
                 visualizer.save_lidar_scans_images(
                    save_hist, save_title, save_lid, save_env, dt,
                    interval_s=float(args.scan_interval),
                    fit_to='environment',
                    progress_cb=progress_cb_fn,
                    quiet=args.quiet,
                )
                 visualizer.save_lidar_polar_images(
                    save_hist, save_title, save_lid, save_env, dt,
                    interval_s=float(args.scan_interval),
                    include_misses=True,
                    progress_cb=progress_cb_fn,
                    quiet=args.quiet,
                )

        if _tqdm is not None:
            with _tqdm(total=total_steps, desc="Salvataggio immagini", unit="img", ncols=90) as pbar:
                progress_cb = lambda _cur, _tot: pbar.update(1)
                _run_all_saves(progress_cb)
        else:
            start_t = time.time(); state = {"done": 0}; width = 36
            def _eta(sec: float) -> str:
                m, s = divmod(int(round(max(0.0, sec))), 60); return f"{m:02d}:{s:02d}"
            def _ascii_cb(_c, _t):
                state["done"] += 1
                done = min(state["done"], total_steps)
                progress_fraction = done / max(1, total_steps)
                filled = int(round(width * progress_fraction))
                bar = '#' * filled + '-' * (width - filled)
                elapsed = time.time() - start_t
                per_step = elapsed / max(1, done)
                remain = per_step * max(0, total_steps - done)
                print(f"\rSalvataggio immagini [{bar}] {done}/{total_steps}  ETA {_eta(remain)}", end='', flush=True)
                if done >= total_steps:
                    print()
            _run_all_saves(_ascii_cb)
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

        # Esegui ICP per tutti i casi prima di aprire il viewer (se non saltato), e salva grafici ICP post-process
        if not args.skip_icp:
            print("\n[Esecuzione ICP] Avvio calcolo ICP su tutte le traiettorie...")
            icp_all_cases = []
            # Parametri ICP globali con piccoli aggiustamenti per alcuni casi
            trim_fraction = 0.6
            damping_enabled = True
            angle_thresh_deg = 10.0
            struct_ratio_thresh = 0.02
            damp_factor = 0.75
            sliding_filter_enabled = True
            angle_balance_enabled = True
            angle_bin_deg = 8.0
            angle_prefer_far = True
            for idx, (case_hist, case_title, case_env, case_lid) in enumerate(zip(histories, titles, envs, lidars)):
                if idx in (3, 4):
                    _maxcorr = 0.38; _sliding_cos = 0.99; _angle_max_bin = 24
                else:
                    _maxcorr = 0.40; _sliding_cos = 0.985; _angle_max_bin = 18
                # Progress bar per-caso
                case_pbar = None
                case_cb = None
                if _tqdm is not None:
                    _step = 1
                    total_pairs = max(0, len(range(1, len(case_hist), max(1, _step))))
                    case_pbar = _tqdm(
                        total=total_pairs,
                        desc=f"ICP – {case_title}",
                        unit="pair",
                        ncols=90,
                        leave=False,
                        bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}]"
                    )
                    case_cb = lambda _d, _t, p=case_pbar: p.update(1)
                try:
                    res_list = run_icp_over_history(
                        case_hist, case_lid, case_env,
                        step=1,
                        max_iterations=40,
                        tolerance=1e-5,
                        max_correspondence_distance=_maxcorr,
                        use_scipy=True,
                        trim_fraction=trim_fraction,
                        damping_enabled=damping_enabled,
                        angle_thresh_deg=angle_thresh_deg,
                        struct_ratio_thresh=struct_ratio_thresh,
                        damp_factor=damp_factor,
                        sliding_filter_enabled=sliding_filter_enabled,
                        sliding_cos_threshold=_sliding_cos,
                        angle_balance_enabled=angle_balance_enabled,
                        angle_bin_deg=angle_bin_deg,
                        angle_max_per_bin=_angle_max_bin,
                        angle_prefer_far=angle_prefer_far,
                        progress_cb=case_cb,
                    )
                finally:
                    if case_pbar is not None:
                        case_pbar.close()
                icp_all_cases.append(res_list)
            print("[Esecuzione ICP] Completata. Salvo grafici ICP riassuntivi...")
            # Salvataggio grafici riassuntivi per ogni caso (concept, overlay, convergence, arrows, raw_vs_filtered)
            try:
                visualizer.ensure_icp_dirs('concept', 'overlays', 'convergence', 'arrows', 'raw_vs_filtered')
            except OSError:
                pass
            # Salva anche risultati ICP in JSON per il viewer (solo trasformazioni)
            import json
            def _to_list(mat):
                return [[float(x) for x in row] for row in np.asarray(mat, dtype=float)]
            def _vec_list(v):
                vv = np.asarray(v, dtype=float).reshape(-1)
                return [float(x) for x in vv[:2]]
            icp_json = {"cases": []}
            for case_title, case_results in zip(titles, icp_all_cases):
                pairs = []
                for r in case_results:
                    if not r.get('ok'): continue
                    k = int(r.get('k', 0))
                    blk_none = r.get('none'); blk_raw = r.get('raw_none')
                    if not isinstance(blk_none, dict) or not isinstance(blk_raw, dict):
                        continue
                    pairs.append({
                        'k': k,
                        'none': {'R': _to_list(blk_none.get('R')), 't': _vec_list(blk_none.get('t'))},
                        'raw_none': {'R': _to_list(blk_raw.get('R')), 't': _vec_list(blk_raw.get('t'))},
                    })
                slug = (case_title.lower().replace(' ', '_').replace('à','a').replace('è','e').replace('é','e'))
                icp_json['cases'].append({'title': case_title, 'slug': slug, 'pairs': pairs})
            try:
                base_dir = os.path.join(os.path.dirname(__file__), '..', 'img', 'icp')
                os.makedirs(base_dir, exist_ok=True)
                out_json_path = os.path.join(base_dir, 'icp_results.json')
                with open(out_json_path, 'w', encoding='utf-8') as fjson:
                    json.dump(icp_json, fjson, ensure_ascii=False, indent=0)
                print(f"[ICP] Risultati serializzati in: {out_json_path}")
            except OSError as _e_json:
                print(f"[ICP] Avviso: non riesco a salvare icp_results.json: {_e_json}")
            def _select_icp_representative(case_results: List[dict]) -> Optional[dict]:
                cand = [res_item for res_item in case_results if res_item.get('ok')]
                if not cand:
                    return None
                def _score(res_item: dict) -> float:
                    rot_deg = 0.0
                    gt_r = res_item.get('gt_R')
                    if gt_r is not None:
                        r_mat = np.asarray(gt_r)
                        if r_mat.shape == (2, 2):
                            rot_deg = abs(float(np.degrees(np.arctan2(r_mat[1, 0], r_mat[0, 0]))))
                    trans = 0.0
                    gt_t = res_item.get('gt_t')
                    if gt_t is not None:
                        t = np.asarray(gt_t).reshape(-1)
                        if t.size >= 2:
                            trans = float(np.linalg.norm(t[:2]))
                    imp = 0.0
                    raw_n = res_item.get('raw_none'); none_f = res_item.get('none')
                    if isinstance(raw_n, dict) and isinstance(none_f, dict):
                        rr = raw_n.get('rmse'); rf = none_f.get('rmse')
                        if isinstance(rr, (int, float)) and isinstance(rf, (int, float)):
                            diff = float(rr) - float(rf)
                            imp = diff if diff > 0.0 else 0.0
                    return 0.6*rot_deg + 0.3*trans + 0.1*imp
                return max(cand, key=_score)
            per_case_imgs = 5
            total_icp_imgs = per_case_imgs * len(histories)
            if _tqdm is not None:
                with _tqdm(total=total_icp_imgs, desc="Grafici ICP", unit="img", ncols=90) as pbar_icp:
                    for plot_title, plot_res in zip(titles, icp_all_cases):
                        rep = _select_icp_representative(plot_res)
                        if rep is None:
                            pbar_icp.update(per_case_imgs)
                            continue
                        base_slug = _slugify_local(plot_title)
                        save_concept_correspondences(rep, f"Corrispondenze – {plot_title}", visualizer.icp_out_path('concept', f"{base_slug}_concept.png")); pbar_icp.update(1)
                        save_alignment_overlays(rep, f"Overlay – {plot_title}", visualizer.icp_out_path('overlays', f"{base_slug}_overlays.png")); pbar_icp.update(1)
                        save_convergence_curves(rep, f"Convergenza – {plot_title}", visualizer.icp_out_path('convergence', f"{base_slug}_convergence.png")); pbar_icp.update(1)
                        save_motion_arrows(rep, f"Δ Pose – {plot_title}", visualizer.icp_out_path('arrows', f"{base_slug}_arrows.png")); pbar_icp.update(1)
                        save_raw_vs_filtered(rep, f"RAW vs Filtrato – {plot_title}", visualizer.icp_out_path('raw_vs_filtered', f"{base_slug}_raw_vs_filtered.png")); pbar_icp.update(1)
            else:
                for plot_title, plot_res in zip(titles, icp_all_cases):
                    rep = _select_icp_representative(plot_res)
                    if rep is None:
                        continue
                    base_slug = _slugify_local(plot_title)
                    save_concept_correspondences(rep, f"Corrispondenze – {plot_title}", visualizer.icp_out_path('concept', f"{base_slug}_concept.png"))
                    save_alignment_overlays(rep, f"Overlay – {plot_title}", visualizer.icp_out_path('overlays', f"{base_slug}_overlays.png"))
                    save_convergence_curves(rep, f"Convergenza – {plot_title}", visualizer.icp_out_path('convergence', f"{base_slug}_convergence.png"))
                    save_motion_arrows(rep, f"Δ Pose – {plot_title}", visualizer.icp_out_path('arrows', f"{base_slug}_arrows.png"))
                    save_raw_vs_filtered(rep, f"RAW vs Filtrato – {plot_title}", visualizer.icp_out_path('raw_vs_filtered', f"{base_slug}_raw_vs_filtered.png"))

            # Nuovo: stampa legenda + riepilogo ICP su console, a meno che non si usi anche --run-icp (evita duplicati)
            if getattr(args, 'icp_verbose', False) and not getattr(args, 'run_icp', False):
                print("\n========== ICP ==========")
                # Legenda
                print(
                    "Legenda:\n"
                    "- Coppia N: scansioni consecutive (k-1, k) nel frame locale del robot\n"
                    "- rmse[ICP]: RMSE della variante filtrata (con trimming / sliding / damping) senza init speciale\n"
                    "- rmse[RAW]: RMSE della variante RAW (nessun filtro / damping)\n"
                    "- it[ICP], it[RAW]: iterazioni eseguite\n"
                    "- alpha[ICP], alpha[RAW]: rotazione stimata (deg) nel frame locale\n"
                    "- Pose: confronto Δx, Δy, α (deg) tra GT, ICP filtrato e RAW (tutte nel frame k-1)\n"
                )
                bold = "\033[1m"; reset = "\033[0m"
                def _case_title(idx_case: int, title: str) -> str:
                    return f"{bold}CASO {idx_case}: {title.upper()}{reset}"
                for idxc, (case_hist, case_title, case_results) in enumerate(zip(histories, titles, icp_all_cases), start=1):
                    print(f"\n{_case_title(idxc, case_title)}", flush=True)
                    for res in case_results:
                        if not res.get('ok', False):
                            print(f"Coppia {res.get('k')}: punti insufficienti (src={res.get('n_src')}, tgt={res.get('n_tgt')})")
                            continue
                        rn = res['none']
                        rrn = res.get('raw_none')
                        prefix = f"Coppia {res['k']}: "
                        indent = " " * len(prefix)
                        if rrn:
                            print(prefix + f"rmse[ICP]={rn['rmse']:.4f}, rmse[RAW]={rrn['rmse']:.4f}")
                            print(indent + f"it[ICP]={rn['iterations']}, it[RAW]={rrn['iterations']}")
                            print(indent + f"alpha[ICP]={rn['alpha_deg']:.4f} deg, alpha[RAW]={rrn['alpha_deg']:.4f} deg")
                        else:
                            print(prefix + f"rmse[ICP]={rn['rmse']:.4f}")
                            print(indent + f"it[ICP]={rn['iterations']}")
                            print(indent + f"alpha[ICP]={rn['alpha_deg']:.4f} deg")
                        # Pose
                        k = int(res['k'])
                        if 1 <= k < len(case_hist):
                            prev_pose = case_hist[k-1]; curr_pose = case_hist[k]
                            r_gt, t_gt = relative_local_transform(prev_pose, curr_pose)
                            def _ang_deg(rm):
                                return 0.0 if rm is None else float(np.degrees(np.arctan2(rm[1, 0], rm[0, 0])))
                            gt_ax = float(t_gt[0]); gt_ay = float(t_gt[1]); gt_ad = _ang_deg(r_gt)
                            n_ax = float(rn['t'][0]); n_ay = float(rn['t'][1]); n_ad = float(rn['alpha_deg'])
                            print("    Pose:")
                            print(f"      {'Reali:':<16}Δx={gt_ax:+.3f} m, Δy={gt_ay:+.3f} m, α={gt_ad:+.4f} deg")
                            print(f"      {'ICP:':<16}Δx={n_ax:+.3f} m, Δy={n_ay:+.3f} m, α={n_ad:+.4f} deg")
                            if rrn:
                                rn_ax = float(rrn['t'][0]); rn_ay = float(rrn['t'][1]); rn_ad = float(rrn['alpha_deg'])
                                print(f"      {'RAW:':<16}Δx={rn_ax:+.3f} m, Δy={rn_ay:+.3f} m, α={rn_ad:+.4f} deg")

        # Mostra viewer dopo tutti i salvataggi e (opzionale) ICP
        if not args.skip_viewer:
            if args.viewer_mode == "grid":
                visualizer.show_trajectories_icp_grid(
                    histories,
                    titles,
                    environment=envs,
                    lidar=lidars,
                    dts=dt,
                    commands_list=commands_list,
                    fit_to='environment',
                    show_info=True,
                    error_messages=[None]*len(histories),
                    stop_indices=stop_indices,
                    stop_fractions=stop_fractions,
                )
            else:
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
            print("\n========== ICP ==========")
            # Titoli dei casi
            bold = "\033[1m"; reset = "\033[0m"
            def _case_title(idx_case: int, title: str) -> str:
                return f"{bold}CASO {idx_case}: {title.upper()}{reset}"
            # Legenda semplificata (solo due varianti)
            print(
                "Legenda:\n"
                "- Coppia N: scansioni consecutive (k-1, k) nel frame locale del robot\n"
                "- rmse[ICP]: RMSE della variante filtrata (con trimming / sliding / damping) senza init speciale\n"
                "- rmse[RAW]: RMSE della variante RAW (nessun filtro / damping)\n"
                "- it[ICP], it[RAW]: iterazioni eseguite\n"
                "- alpha[ICP], alpha[RAW]: rotazione stimata (deg) nel frame locale\n"
                "- Pose: confronto Δx, Δy, α (deg) tra GT, ICP filtrato e RAW (tutte nel frame k-1)\n"
            )
            # Parametri ICP uniformi per tutti i casi (damping meno invasivo)
            trim_fraction = 0.6
            damping_enabled = True
            angle_thresh_deg = 10.0
            struct_ratio_thresh = 0.02
            damp_factor = 0.75
            sliding_filter_enabled = True
            # Nuovo: bilanciamento angolare (favorisce punti lontani per aumentare parallasse)
            angle_balance_enabled = True
            angle_bin_deg = 8.0
            angle_prefer_far = True
            icp_all_cases = []
            for idx, (case_hist, case_title, case_env, case_lid) in enumerate(zip(histories, titles, envs, lidars)):
                print(f"\n{_case_title(idx+1, case_title)}", flush=True)
                # Parametri per-caso (micro-ritocchi): casi 4 e 5 (idx 3 e 4)
                if idx in (3, 4):
                    _maxcorr = 0.38
                    _sliding_cos = 0.99
                    _angle_max_bin = 24
                else:
                    _maxcorr = 0.40
                    _sliding_cos = 0.985
                    _angle_max_bin = 18
                # Progress bar per-caso
                case_pbar = None
                case_cb = None
                if _tqdm is not None:
                    _step = 1
                    total_pairs = max(0, len(range(1, len(case_hist), max(1, _step))))
                    case_pbar = _tqdm(
                        total=total_pairs,
                        desc="",
                        unit="pair",
                        ncols=90,
                        leave=False,
                        bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}]"
                    )
                    case_cb = lambda _d, _t, p=case_pbar: p.update(1)
                try:
                    icp_results = run_icp_over_history(
                        case_hist, case_lid, case_env,
                        step=1,
                        max_iterations=40,
                        tolerance=1e-5,
                        max_correspondence_distance=_maxcorr,
                        use_scipy=True,
                        trim_fraction=trim_fraction,
                        damping_enabled=damping_enabled,
                        angle_thresh_deg=angle_thresh_deg,
                        struct_ratio_thresh=struct_ratio_thresh,
                        damp_factor=damp_factor,
                        sliding_filter_enabled=sliding_filter_enabled,
                        sliding_cos_threshold=_sliding_cos,
                        angle_balance_enabled=angle_balance_enabled,
                        angle_bin_deg=angle_bin_deg,
                        angle_max_per_bin=_angle_max_bin,
                        angle_prefer_far=angle_prefer_far,
                        progress_cb=case_cb,
                    )
                finally:
                    if case_pbar is not None:
                        case_pbar.close()
                icp_all_cases.append(icp_results)
                # Stampa TUTTE le coppie, calcolando comunque tutte le coppie
                _iter_results = icp_results
                for res in _iter_results:
                    if not res.get('ok', False):
                        print(f"Coppia {res['k']}: punti insufficienti (src={res.get('n_src')}, tgt={res.get('n_tgt')})")
                        continue
                    rn = res['none']
                    rrn = res.get('raw_none')
                    prefix = f"Coppia {res['k']}: "
                    indent = " " * len(prefix)
                    if rrn:
                        print(prefix + f"rmse[ICP]={rn['rmse']:.4f}, rmse[RAW]={rrn['rmse']:.4f}")
                        print(indent + f"it[ICP]={rn['iterations']}, it[RAW]={rrn['iterations']}")
                        print(indent + f"alpha[ICP]={rn['alpha_deg']:.4f} deg, alpha[RAW]={rrn['alpha_deg']:.4f} deg")
                    else:
                        print(prefix + f"rmse[ICP]={rn['rmse']:.4f}")
                        print(indent + f"it[ICP]={rn['iterations']}")
                        print(indent + f"alpha[ICP]={rn['alpha_deg']:.4f} deg")
                    # Pose
                    k = int(res['k'])
                    if 1 <= k < len(case_hist):
                        prev_pose = case_hist[k-1]; curr_pose = case_hist[k]
                        r_gt, t_gt = relative_local_transform(prev_pose, curr_pose)
                        def _ang_deg(rm):
                            return 0.0 if rm is None else float(np.degrees(np.arctan2(rm[1, 0], rm[0, 0])))
                        gt_ax = float(t_gt[0]); gt_ay = float(t_gt[1]); gt_ad = _ang_deg(r_gt)
                        n_ax = float(rn['t'][0]); n_ay = float(rn['t'][1]); n_ad = float(rn['alpha_deg'])
                        print("    Pose:")
                        print(f"      {'Reali:':<16}Δx={gt_ax:+.3f} m, Δy={gt_ay:+.3f} m, α={gt_ad:+.4f} deg")
                        print(f"      {'ICP:':<16}Δx={n_ax:+.3f} m, Δy={n_ay:+.3f} m, α={n_ad:+.4f} deg")
                        if rrn:
                            rn_ax = float(rrn['t'][0]); rn_ay = float(rrn['t'][1]); rn_ad = float(rrn['alpha_deg'])
                            print(f"      {'RAW:':<16}Δx={rn_ax:+.3f} m, Δy={rn_ay:+.3f} m, α={rn_ad:+.4f} deg")
            # ====== Salvataggio grafici ICP post-process (1,2,3,9,14) con progress bar ======
            try:
                visualizer.ensure_icp_dirs('concept', 'overlays', 'convergence', 'arrows', 'raw_vs_filtered')
            except OSError:
                pass
            # Salva anche risultati ICP in JSON per il viewer (solo trasformazioni)
            import json
            def _to_list(mat):
                return [[float(x) for x in row] for row in np.asarray(mat, dtype=float)]
            def _vec_list(v):
                vv = np.asarray(v, dtype=float).reshape(-1)
                return [float(x) for x in vv[:2]]
            icp_json = {"cases": []}
            for case_title, case_results in zip(titles, icp_all_cases):
                pairs = []
                for r in case_results:
                    if not r.get('ok'): continue
                    k = int(r.get('k', 0))
                    blk_none = r.get('none'); blk_raw = r.get('raw_none')
                    if not isinstance(blk_none, dict) or not isinstance(blk_raw, dict):
                        continue
                    pairs.append({
                        'k': k,
                        'none': {'R': _to_list(blk_none.get('R')), 't': _vec_list(blk_none.get('t'))},
                        'raw_none': {'R': _to_list(blk_raw.get('R')), 't': _vec_list(blk_raw.get('t'))},
                    })
                slug = (case_title.lower().replace(' ', '_').replace('à','a').replace('è','e').replace('é','e'))
                icp_json['cases'].append({'title': case_title, 'slug': slug, 'pairs': pairs})
            try:
                base_dir = os.path.join(os.path.dirname(__file__), '..', 'img', 'icp')
                os.makedirs(base_dir, exist_ok=True)
                out_json_path = os.path.join(base_dir, 'icp_results.json')
                with open(out_json_path, 'w', encoding='utf-8') as fjson:
                    json.dump(icp_json, fjson, ensure_ascii=False, indent=0)
                print(f"[ICP] Risultati serializzati in: {out_json_path}")
            except OSError as _e_json:
                print(f"[ICP] Avviso: non riesco a salvare icp_results.json: {_e_json}")
            def _select_icp_representative(case_results: List[dict]) -> Optional[dict]:
                cand = [res_item for res_item in case_results if res_item.get('ok')]
                if not cand:
                    return None
                def _score(res_item: dict) -> float:
                    rot_deg = 0.0
                    gt_r = res_item.get('gt_R')
                    if gt_r is not None:
                        r_mat = np.asarray(gt_r)
                        if r_mat.shape == (2, 2):
                            rot_deg = abs(float(np.degrees(np.arctan2(r_mat[1, 0], r_mat[0, 0]))))
                    trans = 0.0
                    gt_t = res_item.get('gt_t')
                    if gt_t is not None:
                        t = np.asarray(gt_t).reshape(-1)
                        if t.size >= 2:
                            trans = float(np.linalg.norm(t[:2]))
                    imp = 0.0
                    raw_n = res_item.get('raw_none'); none_f = res_item.get('none')
                    if isinstance(raw_n, dict) and isinstance(none_f, dict):
                        rr = raw_n.get('rmse'); rf = none_f.get('rmse')
                        if isinstance(rr, (int, float)) and isinstance(rf, (int, float)):
                            diff = float(rr) - float(rf)
                            imp = diff if diff > 0.0 else 0.0
                    return 0.6*rot_deg + 0.3*trans + 0.1*imp
                return max(cand, key=_score)
            per_case_imgs = 4  # concept, overlays, convergence, raw_vs_filtered; 'arrows' mantiene ma senza Odo
            total_icp_imgs = per_case_imgs * len(histories)
            if _tqdm is not None:
                with _tqdm(total=total_icp_imgs, desc="Grafici ICP", unit="img", ncols=90) as pbar_icp:
                    for plot_title, plot_res in zip(titles, icp_all_cases):
                        rep = _select_icp_representative(plot_res)
                        if rep is None:
                            pbar_icp.update(per_case_imgs)
                            continue
                        base_slug = _slugify_local(plot_title)
                        save_concept_correspondences(rep, f"Corrispondenze – {plot_title}", visualizer.icp_out_path('concept', f"{base_slug}_concept.png")); pbar_icp.update(1)
                        save_alignment_overlays(rep, f"Overlay – {plot_title}", visualizer.icp_out_path('overlays', f"{base_slug}_overlays.png")); pbar_icp.update(1)
                        save_convergence_curves(rep, f"Convergenza – {plot_title}", visualizer.icp_out_path('convergence', f"{base_slug}_convergence.png")); pbar_icp.update(1)
                        save_raw_vs_filtered(rep, f"RAW vs Filtrato – {plot_title}", visualizer.icp_out_path('raw_vs_filtered', f"{base_slug}_raw_vs_filtered.png")); pbar_icp.update(1)
            else:
                for plot_title, plot_res in zip(titles, icp_all_cases):
                    rep = _select_icp_representative(plot_res)
                    if rep is None:
                        continue
                    base_slug = _slugify_local(plot_title)
                    save_concept_correspondences(rep, f"Corrispondenze – {plot_title}", visualizer.icp_out_path('concept', f"{base_slug}_concept.png"))
                    save_alignment_overlays(rep, f"Overlay – {plot_title}", visualizer.icp_out_path('overlays', f"{base_slug}_overlays.png"))
                    save_convergence_curves(rep, f"Convergenza – {plot_title}", visualizer.icp_out_path('convergence', f"{base_slug}_convergence.png"))
                    save_raw_vs_filtered(rep, f"RAW vs Filtrato – {plot_title}", visualizer.icp_out_path('raw_vs_filtered', f"{base_slug}_raw_vs_filtered.png"))

    finally:
        # Ripristina stream originali e chiudi il file di log
        try:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr
        finally:
            try:
                _log_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()

