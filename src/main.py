from robot import Robot
from trajectory_generator import TrajectoryGenerator
from simulator import Simulator
import visualizer
from environment_presets import setup_environments_per_trajectory
from lidar import Lidar  # sensore LiDAR
import argparse
from icp import run_icp_pair, compute_relative_transform_from_odometry
import time  # per ETA nella barra di progresso
from tqdm import tqdm as _tqdm  # progress bar esterna con ETA
from icp_plots import (
    save_concept_correspondences,
    save_convergence_curves,
    save_motion_arrows,
    save_raw_vs_filtered,
    save_error_over_time,
)
from typing import List, Optional, Tuple, Dict
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

# Piccolo helper: wrapping angolare in [-pi,pi)
_def_pi = math.pi
_def_2pi = 2.0 * math.pi

def _wrap_angle(a: float) -> float:
    return (float(a) + _def_pi) % _def_2pi - _def_pi


def _apply_world_transform(traj: Optional[np.ndarray], base_pose: np.ndarray) -> Optional[np.ndarray]:
    """Applica la trasformazione di mondo (R0,t0) derivata dalla prima posa reale base_pose = (x0,y0,theta0)
    a una traiettoria (x,y,theta) locale ricostruita dal log. Ritorna una nuova array, o None se traj è None.
    """
    if traj is None:
        return None
    arr = np.asarray(traj, dtype=float).copy()
    if arr.ndim != 2 or arr.shape[1] < 3:
        return arr
    x0, y0, th0 = map(float, base_pose[:3])
    c, s = math.cos(th0), math.sin(th0)
    r0 = np.array([[c, -s], [s, c]], dtype=float)
    arr_xy = arr[:, :2] @ r0.T + np.array([x0, y0], dtype=float)
    arr_th = np.array([_wrap_angle(th0 + t) for t in arr[:, 2]], dtype=float)
    out = arr.copy()
    out[:, 0:2] = arr_xy
    out[:, 2] = arr_th
    return out

# Regex per sopprimere nel file le righe di progress bar e "Salvataggio immagini"
# - Righe che iniziano con "Salvataggio immagini" (tqdm o ASCII fallback)
# - Righe di tqdm con percentuale e barra (es. " 42%|####...") ovunque nella riga
_prog_re_salva = re.compile(r"^\s*Salvataggio immagini\b")
_prog_re_tqdm = re.compile(r"\b\d{1,3}%\|")

# Tee per duplicare stdout/stderr su file e console, filtrando le progress nel file
class _Tee:
    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary
        self.encoding = getattr(primary, 'encoding', 'utf-8')
        self._buf = ''
        self._suppressed_last = False  # evita righe vuote dopo soppressione

    @staticmethod
    def _should_suppress_line(line: str) -> bool:
        if not line:
            return False
        s = line.lstrip('\r')
        if _prog_re_salva.match(s):
            return True
        if _prog_re_tqdm.search(s):
            return True
        return False

    def _write_to_secondary_filtered(self, data: str) -> None:
        # Normalizza solo CRLF in LF; lasciamo i CR come separatori di update trattandoli come fine linea
        text = data.replace('\r\n', '\n')
        # Spezza su CR per catturare aggiornamenti in-place senza introdurre '\n' spurii
        parts = text.split('\r')
        for idxp, part in enumerate(parts):
            self._buf += part
            # processa linee complete terminate da \n
            while True:
                nl = self._buf.find('\n')
                if nl == -1:
                    break
                line = self._buf[:nl]
                self._buf = self._buf[nl+1:]
                if self._should_suppress_line(line):
                    self._suppressed_last = True
                    continue
                # scarta righe vuote immediatamente dopo soppressione
                if self._suppressed_last and line.strip() == '':
                    # mantieni flag finché non arriva una riga non vuota
                    continue
                self._secondary.write(line + '\n')
                self._suppressed_last = False
            # Se non è l'ultimo pezzo, abbiamo avuto un CR che segnala aggiornamento riga: tratta il contenuto accumulato come linea completa
            if idxp < len(parts) - 1:
                line_cr = self._buf
                self._buf = ''
                if self._should_suppress_line(line_cr):
                    self._suppressed_last = True
                    continue
                if self._suppressed_last and (line_cr.strip() == ''):
                    continue
                # Scrive la linea derivata da CR senza aggiungere newline extra (usa \n per chiudere la linea corrente)
                self._secondary.write(line_cr)
                self._suppressed_last = False

    def write(self, data):
        try:
            self._primary.write(data)
        finally:
            try:
                self._write_to_secondary_filtered(str(data))
                # flush immediato del file per non perdere dati in caso di terminazione improvvisa
                if hasattr(self._secondary, 'flush'):
                    self._secondary.flush()
            except (IOError, OSError, AttributeError):
                # Gestisce errori di I/O o problemi con l'oggetto file
                self._secondary.write(str(data))
                try:
                    self._secondary.flush()
                except (IOError, OSError, AttributeError):
                    pass
        return len(data)

    def flush(self):
        if self._buf:
            rem = self._buf
            self._buf = ''
            if not self._should_suppress_line(rem):
                if not (self._suppressed_last and rem.strip() == ''):
                    self._secondary.write(rem)
            # se era soppressa o vuota dopo soppressione, non scrivere nulla
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


def apply_loop_closure_correction(trajectory: np.ndarray, is_circular: bool = False, closure_threshold: float = 0.3) -> np.ndarray:
    """
    Applica correzione di loop closure per traiettorie circolari.
    Se il punto finale è vicino all'inizio, distribuisce l'errore lungo tutta la traiettoria.

    Args:
        trajectory: Array Nx3 [x, y, theta]
        is_circular: Se True, forza la chiusura del loop
        closure_threshold: Distanza massima per considerare il loop chiuso (metri)

    Returns:
        Traiettoria corretta
    """
    if len(trajectory) < 10 or not is_circular:
        return trajectory

    # Controlla se il loop è quasi chiuso
    start = trajectory[0, :2]
    end = trajectory[-1, :2]
    distance = np.linalg.norm(end - start)

    if distance > closure_threshold:
        # Loop non abbastanza vicino, non correggere
        return trajectory

    # Calcola errore totale
    error_xy = start - end
    error_theta = trajectory[0, 2] - trajectory[-1, 2]

    # Normalizza errore angolare a [-pi, pi]
    while error_theta > np.pi:
        error_theta -= 2 * np.pi
    while error_theta < -np.pi:
        error_theta += 2 * np.pi

    # Distribuisci la correzione linearmente lungo la traiettoria
    corrected = trajectory.copy()
    n = len(trajectory)

    for i in range(1, n):
        # Frazione del percorso completato
        alpha = float(i) / float(n - 1)

        # Applica correzione proporzionale
        corrected[i, 0] += alpha * error_xy[0]
        corrected[i, 1] += alpha * error_xy[1]
        corrected[i, 2] += alpha * error_theta

    return corrected


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


# ===== Parser ICP da log (usa ESATTAMENTE i valori stampati) =====
_re_case_hdr = re.compile(r"^\s*CASO\s+(\d+):\s*(.+)$")
_re_icp_pose = re.compile(r"^\s*ICP:\s*Δx=([+\-]?[0-9]+(?:\.[0-9]+)?)\s*m,\s*Δy=([+\-]?[0-9]+(?:\.[0-9]+)?)\s*m,\s*α=([+\-]?[0-9]+(?:\.[0-9]+)?)\s*deg\s*$")
_ansi_re = re.compile(r"\x1b\[[0-9;]*m")

# Nuovo: regex generica per tre etichette (Reali, ICP, RAW)
_re_pose_labeled = re.compile(
    r"^\s*(Reali:|ICP:|RAW:)\s*Δx=([+\-]?\d+(?:\.\d+)?)\s*m,\s*Δy=([+\-]?\d+(?:\.\d+)?)\s*m,\s*α=([+\-]?\d+(?:\.\d+)?)\s*deg\s*$"
)

def _accumulate_icp_deltas_to_traj(deltas: List[Tuple[float, float, float]]) -> np.ndarray:
    """Dati Δ pose locali (dx [m], dy [m], alpha_deg [deg]) nel frame k-1,
    integra in una traiettoria globale partendo da (0,0,0) senza trasformazioni extra."""
    n = len(deltas)
    hist = np.zeros((n + 1, 3), dtype=float)
    x = 0.0; y = 0.0; th = 0.0
    hist[0] = [x, y, th]
    for i, (dx, dy, a_deg) in enumerate(deltas, start=1):
        a = math.radians(float(a_deg))
        # Trasforma l'incremento locale (dx,dy) nel mondo ruotandolo dell'orientamento corrente
        c, s = math.cos(th), math.sin(th)
        gx = c * dx - s * dy
        gy = s * dx + c * dy
        x += gx
        y += gy
        th = (th + a + math.pi) % (2.0 * math.pi) - math.pi
        hist[i] = [x, y, th]
    return hist


def _parse_icp_trajectories_from_log(log_path: str, n_cases: int) -> List[Optional[np.ndarray]]:
    """[DEPRECATO] Mantiene la vecchia API: ritorna solo ICP filtrato.
    Usata per compatibilità, delega al parser completo e prende la serie 'icp'."""
    triplets = parse_icp_triplets_from_log(log_path, n_cases)
    out: List[Optional[np.ndarray]] = []
    for case in triplets:
        out.append(case.get('icp'))
    return out


def parse_icp_triplets_from_log(log_path: str, n_cases: int) -> List[Dict[str, Optional[np.ndarray]]]:
    """Parsa il file di log corrente e ricostruisce per ciascun CASO le traiettorie
    usando ESATTAMENTE i Δ stampati per: Reali, ICP filtrato ("ICP:"), RAW ("RAW:").
    Ritorna una lista per-caso di dizionari: {'real': np.ndarray|None, 'icp': np.ndarray|None, 'raw': np.ndarray|None}.
    Ogni traiettoria parte da (0,0,0)."""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError:
        return [{"real": None, "icp": None, "raw": None} for _ in range(int(n_cases))]

    # Accumula Δ per caso e per etichetta
    per_case: List[Dict[str, List[Tuple[float, float, float]]]] = [
        {"real": [], "icp": [], "raw": []} for _ in range(int(n_cases))
    ]
    cur_case_idx: Optional[int] = None

    for raw_line in lines:
        clean = _ansi_re.sub('', raw_line)
        line = clean.rstrip('\n')
        m_hdr = _re_case_hdr.match(line)
        if m_hdr:
            try:
                idx = int(m_hdr.group(1))
                cur_case_idx = idx - 1 if 1 <= idx <= n_cases else None
            except (ValueError, IndexError):
                # Gestisce errori di conversione o accesso al gruppo
                cur_case_idx = None
            continue
        if cur_case_idx is None:
            continue
        m_pose = _re_pose_labeled.match(line)
        if not m_pose:
            continue
        label = m_pose.group(1)
        try:
            dx = float(m_pose.group(2))
            dy = float(m_pose.group(3))
            a_deg = float(m_pose.group(4))
        except (ValueError, IndexError):
            # Gestisce errori di conversione float o accesso ai gruppi
            continue
        if label.startswith('Reali'):
            per_case[cur_case_idx]['real'].append((dx, dy, a_deg))
        elif label.startswith('ICP'):
            per_case[cur_case_idx]['icp'].append((dx, dy, a_deg))
        elif label.startswith('RAW'):
            per_case[cur_case_idx]['raw'].append((dx, dy, a_deg))

    # Costruisci traiettorie
    out: List[Dict[str, Optional[np.ndarray]]] = []
    for cs in per_case:
        item: Dict[str, Optional[np.ndarray]] = {}
        for k in ('real', 'icp', 'raw'):
            deltas = cs.get(k, [])
            item[k] = _accumulate_icp_deltas_to_traj(deltas) if deltas else None
        out.append(item)
    return out
# ===== Fine parser ICP da log =====

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
    parser.add_argument("--viewer-log-align-world", action="store_true", help="Allinea le traiettorie ricostruite dal LOG (Reali/RAW/ICP) al mondo usando la prima posa reale del caso")
    # parser.add_argument("--viewer-from-log-icp", action="store_true", help="Usa le traiettorie ICP ricostruite ESATTAMENTE dal log corrente nel viewer")
    parser.set_defaults(icp_verbose=True)
    args = parser.parse_args()

    # Se viene lanciato senza argomenti, attiva automaticamente tutte le funzionalità
    if len(sys.argv) == 1:
        # Attiva automaticamente ICP e viewer in modalità carousel (animato)
        args.run_icp = True
        args.skip_viewer = False
        args.viewer_mode = "carousel"  # Modalità animata
        args.viewer_log_align_world = True

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
    _log_file = open(_log_path, 'w', encoding='utf-8', newline='', buffering=1)
    sys.stdout = _Tee(_orig_stdout, _log_file)
    sys.stderr = _Tee(_orig_stderr, _log_file)

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

        # Parametri base di riferimento - AUMENTATI per migliorare ICP
        v_ref = 1.0         # Era 0.5 - raddoppiato per movimenti più grandi
        radius_ref = 1.5    # Era 2.0 - ridotto per avere curve più strette (più feature)
        v_min_ref = 0.4     # Era 0.2
        v_max_ref = 1.6     # Era 0.8
        omega_std_ref = 0.8 # Era 0.5

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

        # 5) Traiettoria a 8 — ciclo completo con raggio maggiore per separare i lobi
        # POSIZIONAMENTO SPECIALE: centrata per evitare di uscire dai bounds
        v = v_ref
        r_eight = 1.8  # Raggio ridotto a 1.8m per stare nei bounds
        period_eight = (4.0 * math.pi * r_eight) / max(v, 1e-9)
        n_steps_eight = max(2, int(round(period_eight / dt)))
        if n_steps_eight % 2 == 1:
            n_steps_eight += 1
        t_eight = (n_steps_eight - 1e-9) * dt
        vs, omegas = tg.eight(v=v, radius=r_eight, T=t_eight, dt=dt)
        # Reset con posizione iniziale standard (verticale per default)
        reset_robot_default(sim)
        histories.append(sim.run_from_sequence(vs, omegas, dt))
        commands_list.append(sim.commands)
        titles.append("Traiettoria a 8")

        # 6) Random walk
        t_rw = 40.0  # durata random walk
        v_mean = v_ref
        omega_std = omega_std_ref
        vs, omegas = tg.random_walk(v_mean=v_mean, omega_std=omega_std, T=t_rw, dt=dt, seed=456)
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

            for idx, (case_hist, case_title, case_env, case_lid) in enumerate(zip(histories, titles, envs, lidars)):
                # Imposta max_correspondence_distance in base al tipo di traiettoria
                if idx == 0:  # rettilinea v costante
                    _maxcorr = 0.25
                elif idx == 1:  # rettilinea v variabile - OTTIMIZZATO
                    _maxcorr = 0.30  # Leggermente più alto per accelerazioni
                elif idx == 2:  # circolare v costante
                    _maxcorr = 0.20
                elif idx == 3:  # circolare v variabile
                    _maxcorr = 0.20
                elif idx == 4:  # traiettoria a 8 - AUMENTATO per geometria complessa
                    _maxcorr = 0.35  # Più alta per gestire transizione tra lobi
                else:  # random walk - AUMENTATO ANCORA per ICP RAW
                    _maxcorr = 0.60  # MOLTO alta per gestire qualsiasi disallineamento

                # Imposta intervallo di scansione per ICP in base al tipo di traiettoria
                # Per traiettorie complesse (a 8, rettilinea v variabile, random walk), usa 10 Hz (0.1 secondi)
                if idx == 4:  # traiettoria a 8
                    icp_scan_interval = 0.1  # 10 Hz (ogni 0.1 secondi)
                elif idx == 1:  # rettilinea v variabile - AUMENTATE LE SCANSIONI
                    icp_scan_interval = 0.1  # 10 Hz (ogni 0.1 secondi) invece di 0.05
                elif idx == 5:  # random walk - AUMENTATE LE SCANSIONI
                    icp_scan_interval = 0.1  # 10 Hz (ogni 0.1 secondi) per più feature
                else:
                    icp_scan_interval = dt  # usa dt (0.05s) per processare ogni frame consecutivo

                # Calcola step_idx per questo caso specifico
                _step = max(1, int(round(icp_scan_interval / max(1e-9, float(dt)))))

                # Progress bar per-caso
                case_pbar = None
                if _tqdm is not None:
                    total_pairs = max(0, len(range(_step, len(case_hist), _step)))
                    case_pbar = _tqdm(
                        total=total_pairs,
                        desc=f"ICP – {case_title}",
                        unit="pair",
                        ncols=90,
                        leave=False,
                        bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}]"
                    )

                # Esegui ICP su coppie con intervallo specifico
                res_list = []
                for k in range(_step, len(case_hist), _step):
                    prev_idx = k - _step
                    prev_pose = case_hist[prev_idx]
                    curr_pose = case_hist[k]

                    # Genera scansioni LiDAR in frame locale
                    prev_local = case_lid.scan_hits(prev_pose, case_env, frame='local')
                    curr_local = case_lid.scan_hits(curr_pose, case_env, frame='local')

                    if len(prev_local) < 10 or len(curr_local) < 10:
                        res_list.append({
                            'ok': False,
                            'k': k,
                            'prev_k': prev_idx,
                            'n_src': len(curr_local),
                            'n_tgt': len(prev_local)
                        })
                        if case_pbar is not None:
                            case_pbar.update(1)
                        continue

                    # Esegui ICP con il nuovo algoritmo semplice
                    # Per traiettorie complesse, usa più iterazioni per convergenza migliore
                    if idx == 5:  # random walk
                        _max_iter = 150
                        _tolerance = 1e-7
                    elif idx == 1:  # rettilinea v variabile
                        _max_iter = 60
                        _tolerance = 1e-6
                    else:
                        _max_iter = 50
                        _tolerance = 1e-6

                    result = run_icp_pair(
                        prev_pose, curr_pose,
                        curr_local, prev_local,  # source=k, target=k-_step
                        max_iterations=_max_iter,
                        tolerance=_tolerance,
                        max_correspondence_distance=_maxcorr
                    )
                    result['k'] = k
                    result['prev_k'] = prev_idx
                    res_list.append(result)

                    if case_pbar is not None:
                        case_pbar.update(1)

                if case_pbar is not None:
                    case_pbar.close()

                icp_all_cases.append(res_list)
            print("[Esecuzione ICP] Completata. Salvo grafici ICP riassuntivi...")
            # Salvataggio grafici riassuntivi per ogni caso (concept, convergence, arrows, raw_vs_filtered, error_over_time)
            try:
                visualizer.ensure_icp_dirs('concept', 'convergence', 'arrows', 'raw_vs_filtered', 'error_over_time')
            except OSError:
                pass

            def _select_icp_representative(results_list: List[dict]) -> Optional[dict]:
                cand = [res_item for res_item in results_list if res_item.get('ok')]
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
            per_case_imgs = 5  # concept, convergence, arrows, raw_vs_filtered, error_over_time
            total_icp_imgs = per_case_imgs * len(histories)
            if _tqdm is not None:
                with _tqdm(total=total_icp_imgs, desc="Grafici ICP", unit="img", ncols=90) as pbar_icp:
                    for idx_case, (plot_title, plot_res, case_hist) in enumerate(zip(titles, icp_all_cases, histories)):
                        rep = _select_icp_representative(plot_res)
                        if rep is None:
                            pbar_icp.update(per_case_imgs)
                            continue
                        base_slug = _slugify_local(plot_title)
                        save_concept_correspondences(rep, f"Corrispondenze – {plot_title}", visualizer.icp_out_path('concept', f"{base_slug}_concept.png")); pbar_icp.update(1)
                        save_convergence_curves(rep, f"Convergenza – {plot_title}", visualizer.icp_out_path('convergence', f"{base_slug}_convergence.png")); pbar_icp.update(1)
                        save_motion_arrows(rep, f"Δ Pose – {plot_title}", visualizer.icp_out_path('arrows', f"{base_slug}_arrows.png")); pbar_icp.update(1)
                        save_raw_vs_filtered(rep, f"RAW vs Filtrato – {plot_title}", visualizer.icp_out_path('raw_vs_filtered', f"{base_slug}_raw_vs_filtered.png")); pbar_icp.update(1)
                        pbar_icp.update(1)  # Placeholder per error_over_time (verrà generato dopo)
            else:
                for idx_case, (plot_title, plot_res, case_hist) in enumerate(zip(titles, icp_all_cases, histories)):
                    rep = _select_icp_representative(plot_res)
                    if rep is None:
                        continue
                    base_slug = _slugify_local(plot_title)
                    save_concept_correspondences(rep, f"Corrispondenze – {plot_title}", visualizer.icp_out_path('concept', f"{base_slug}_concept.png"))
                    save_convergence_curves(rep, f"Convergenza – {plot_title}", visualizer.icp_out_path('convergence', f"{base_slug}_convergence.png"))
                    save_motion_arrows(rep, f"Δ Pose – {plot_title}", visualizer.icp_out_path('arrows', f"{base_slug}_arrows.png"))
                    save_raw_vs_filtered(rep, f"RAW vs Filtrato – {plot_title}", visualizer.icp_out_path('raw_vs_filtered', f"{base_slug}_raw_vs_filtered.png"))

            # Stampa legenda + riepilogo ICP su console quando run_icp è attivo
            if getattr(args, 'run_icp', False):
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
                def _case_title(case_idx: int, case_title_str: str) -> str:
                    return f"{bold}CASO {case_idx}: {case_title_str.upper()}{reset}"
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
                        prev_k = res.get('prev_k', k-1)  # usa prev_k se disponibile, altrimenti k-1 per retrocompatibilità
                        if 0 <= prev_k < len(case_hist) and k < len(case_hist):
                            prev_pose = case_hist[prev_k]; curr_pose = case_hist[k]
                            r_gt, t_gt = compute_relative_transform_from_odometry(prev_pose, curr_pose)
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

        # ===== Ricostruzione traiettorie da LOG corrente (se richiesto) =====
        icp_histories_from_log: Optional[List[np.ndarray]] = None
        icp_raw_from_log: Optional[List[np.ndarray]] = None
        icp_filt_from_log: Optional[List[np.ndarray]] = None
        if not args.skip_icp:
            # Assicura che le stampe ICP siano state flushate su file prima di leggere
            try:
                sys.stdout.flush(); sys.stderr.flush()
            except (IOError, OSError):
                # Gestisce errori di flush degli stream
                pass
            triplets = parse_icp_triplets_from_log(_log_path, n_cases=len(titles))
            icp_histories_from_log = []
            icp_raw_from_log = []
            icp_filt_from_log = []
            for idx, item in enumerate(triplets):
                real = item.get('real')
                icp = item.get('icp')
                raw = item.get('raw')
                # ognuno può mancare: inserisci fallback minimale
                real_f = real if isinstance(real, np.ndarray) and real.size > 0 else np.zeros((1,3), dtype=float)
                raw_f = raw if isinstance(raw, np.ndarray) and raw.size > 0 else np.zeros((1,3), dtype=float)
                icp_f = icp if isinstance(icp, np.ndarray) and icp.size > 0 else np.zeros((1,3), dtype=float)

                # Applica loop closure correction SOLO per traiettorie circolari (idx 2, 3)
                # NON applicare al caso 4 (otto) perché ha una forma diversa
                is_circular = idx in (2, 3)
                if is_circular and len(icp_f) > 10:
                    icp_f = apply_loop_closure_correction(icp_f, is_circular=True, closure_threshold=0.5)
                    raw_f = apply_loop_closure_correction(raw_f, is_circular=True, closure_threshold=0.5)

                # Allineamento opzionale al mondo: usa la prima posa reale del caso simulato
                if getattr(args, 'viewer_log_align_world', False) and idx < len(histories) and len(histories[idx]) > 0:
                    base = np.asarray(histories[idx][0], dtype=float)
                    real_f = _apply_world_transform(real_f, base)
                    raw_f = _apply_world_transform(raw_f, base)
                    icp_f = _apply_world_transform(icp_f, base)
                icp_histories_from_log.append(real_f)
                icp_raw_from_log.append(raw_f)
                icp_filt_from_log.append(icp_f)

            # Salva grafici degli errori nel tempo per ogni caso
            print("[Grafici Errori] Salvataggio grafici errori ICP nel tempo...")
            for idx, (real_traj, icp_traj, raw_traj, title) in enumerate(zip(icp_histories_from_log, icp_filt_from_log, icp_raw_from_log, titles)):
                if real_traj is not None and icp_traj is not None and raw_traj is not None:
                    base_slug = _slugify_local(title)
                    save_error_over_time(
                        real_traj, icp_traj, raw_traj,
                        title,
                        visualizer.icp_out_path('error_over_time', f"{base_slug}_error_over_time.png"),
                        dt=dt[idx] if isinstance(dt, list) else dt
                    )
        # ===== Fine ricostruzione =====

        # Mostra viewer dopo tutti i salvataggi e (opzionale) ICP
        if not args.skip_viewer:
            # Scegli set di traiettorie per il viewer
            # Se abbiamo ricostruito dal log: usa real(icp_histories_from_log) come pannello "Reale – ... (ICP da log)"
            # e passa raw/filtrato al viewer per i due pannelli ICP
            if icp_histories_from_log is not None:
                viewer_histories = icp_histories_from_log
                viewer_titles = titles
                viewer_cmds = None
                viewer_raw = icp_raw_from_log
                viewer_filt = icp_filt_from_log
            else:
                viewer_histories = histories
                viewer_titles = titles
                viewer_cmds = commands_list
                viewer_raw = None
                viewer_filt = None

            if args.viewer_mode == "grid":
                visualizer.show_trajectories_icp_grid(
                    viewer_histories,
                    viewer_titles,
                    environment=envs,
                    fit_to='environment',
                )
            else:
                visualizer.show_trajectories_carousel(
                    viewer_histories,
                    viewer_titles,
                    show_orient_every=show_steps,
                    _save_each=False,
                    _commands_list=viewer_cmds,
                    _dts=dt,
                    _show_info=True,
                    environment=envs,
                    _fit_to='environment',
                    _stop_indices=stop_indices,
                    _stop_fractions=stop_fractions,
                    lidar=lidars,
                    _show_lidar=True,
                    _lidar_every=int(max(1, args.viewer_lidar_every)),
                    icp_raw_histories=viewer_raw,
                    icp_filt_histories=viewer_filt,
                )


    finally:
        # Ripristina stream originali e chiudi il file di log
        try:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr
        finally:
            try:
                _log_file.close()
            except (IOError, OSError):
                # Gestisce errori durante la chiusura del file
                pass


if __name__ == "__main__":
    main()
