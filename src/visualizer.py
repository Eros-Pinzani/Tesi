"""Visualizer di traiettorie del robot

Funzionalità principali:
- Disegno della traiettoria con simboli del robot a intervalli regolari.
- Salvataggio immagini statiche in PNG nella cartella img/ con nomi basati sul titolo e timestamp.
- Viewer interattivo “carousel” con pulsanti (precedente/play/successivo) e
  pannello informazioni opzionale. Nel viewer i simboli del robot lungo la traiettoria compaiono
  progressivamente quando il robot mobile raggiunge quelle posizioni; le immagini salvate invece
  li mostrano tutti, come in precedenza.

Note implementative:
- Le routine di disegno del robot sono centralizzate (draw_robot), e funzioni di supporto calcolano
  dimensioni coerenti con l’estensione della traiettoria.
- Il viewer usa un timer di Matplotlib per far avanzare i frame; gli “artisti” grafici creati
  per il robot mobile vengono rimossi e ricreati ogni frame per un aggiornamento pulito.
- Per rendere snello il codice, le icone dei pulsanti usano simboli Unicode (compatibili su Windows)
  al posto di patch disegnate manualmente.
"""

# Classe che ha il compito di plottare le traiettorie, disegnare il robot in alcuni istanti e salvare figure

import matplotlib.pyplot as plt  # API principale per creare figure/assi e tracciare linee/frecce
from matplotlib.patches import Circle, Rectangle  # Primitive grafiche 2D per centro/ruote e corpo del robot
import numpy as np  # Calcolo numerico: array, trigonometria, differenze, range
from pathlib import Path  # Percorsi portabili per cartella img/ e file PNG
import re  # Normalizzazione del titolo in un nome file sicuro (slugify minimo)
from datetime import datetime  # Timestamp per nomi univoci ed evitare sovrascritture
from matplotlib.widgets import Button  # Pulsanti UI per navigazione e Play/Pausa
from matplotlib import transforms as mtransforms  # Trasformazioni affini: rotazione/traslazione delle patch
from typing import Optional, List, Sequence, Union  # Annotazioni di tipo per migliori suggerimenti e linting
from matplotlib.text import Text  # Artista di testo (pannello info, legenda)
from contextlib import suppress  # Ignora eccezioni non critiche in operazioni best-effort
from matplotlib.artist import Artist  # Tipo base di tutti gli elementi disegnabili (patch, arrow, ecc.)
from environment import Environment  # Per disegnare confini e ostacoli
from lidar import Lidar  # Tipo del sensore per visualizzazione raggi
import shutil  # Per pulire cartelle di output delle immagini


# Helper per rimuovere in sicurezza un artista matplotlib (gestisce None ed eccezioni)
def _safe_remove_artist(artist: Optional[object]):
    """Prova a rimuovere un artista (patch/annotazione ecc.) ignorando errori e None."""
    if artist is not None and hasattr(artist, 'remove'):
        with suppress(Exception):
            artist.remove()  # type: ignore[attr-defined]


def _rect_dims_from_radius(robot_radius: float):
    """Deriva dimensioni del rettangolo dal parametro di scala robot_radius.
    - width (lato corto, fronte): ~2× robot_radius
    - length (lato lungo, direzione di marcia): ~4× robot_radius
    Ritorna (width, length)."""
    width = 2.0 * robot_radius
    length = 4.0 * robot_radius
    return width, length


def _wheel_params(robot_radius: float):
    """Parametri delle rotelle a partire dalla scala del robot.
    Ritorna (wheel_radius, offset_out) dove offset_out è l'offset del centro ruota
    verso l'esterno rispetto alla fiancata (in coordinate locali)."""
    wheel_radius = 0.22 * robot_radius
    offset_out = 0.15 * robot_radius
    return wheel_radius, offset_out


def draw_robot(ax, state, robot_radius=0.1, color='tab:blue', dir_len=None, arrow_color='orange', center_color='orange',
               wheel_facecolor='white', wheel_edgecolor='k') -> List[Artist]:
    """Disegna il robot come rettangolo orientato con freccia e ruote.

    Parametri principali:
    - ax: axes Matplotlib su cui disegnare
    - state: [x, y, theta] posa del robot nel mondo
    - robot_radius: scala complessiva (controlla dimensioni corpo/ruote/freccia)
    - color, arrow_color, center_color: colori per corpo, freccia, pallino centrale

    Ritorna: lista degli artisti creati (utile per rimuoverli al frame successivo).
    """
    x, y, th = state
    artists: List[Artist] = []

    # Corpo rettangolare: lato lungo allineato con l'orientamento (theta)
    width, length = _rect_dims_from_radius(robot_radius)

    # Definisco il rettangolo nel frame locale (centro = 0) e applico rotazione+traslazione
    rect = Rectangle((-length/2.0, -width/2.0), length, width, linewidth=1.0, facecolor=color, alpha=0.3, edgecolor='k', zorder=3)
    trans = mtransforms.Affine2D().rotate(th).translate(x, y) + ax.transData
    rect.set_transform(trans)
    ax.add_patch(rect)
    artists.append(rect)

    # Rotelle: quattro cerchi vicino alle estremità dei lati lunghi (sempre disegnate)
    w_r, w_off = _wheel_params(robot_radius)
    wheel_long_frac = 0.8  # posizione lungo il lato lungo (80% della semi-lunghezza)
    x_off = wheel_long_frac * (length / 2.0)
    corners = [
        ( +x_off, +width/2.0 + w_off),  # lato superiore, estremità destra
        ( -x_off, +width/2.0 + w_off),  # lato superiore, estremità sinistra
        ( +x_off, -width/2.0 - w_off),  # lato inferiore, estremità destra
        ( -x_off, -width/2.0 - w_off),  # lato inferiore, estremità sinistra
    ]
    for cx, cy in corners:
        wheel = Circle((cx, cy), w_r, facecolor=wheel_facecolor, edgecolor=wheel_edgecolor, linewidth=1.0, zorder=4)
        wheel.set_transform(trans)
        ax.add_patch(wheel)
        artists.append(wheel)

    # Pallino centrale (rende evidente il centro del corpo)
    center_r = 0.25 * robot_radius
    center = Circle((0.0, 0.0), center_r, fill=True, color=center_color, ec='none', zorder=4)
    center.set_transform(trans)
    ax.add_patch(center)
    artists.append(center)

    # Freccia di orientamento (punta nella direzione di marcia)
    if dir_len is None:
        dir_len = 3.0 * robot_radius  # lunghezza default della freccia
    dx = dir_len * np.cos(th)
    dy = dir_len * np.sin(th)
    arr = ax.arrow(
        x,  # punto di partenza (posizione del robot)
        y,
        dx,  # componente x della freccia
        dy,  # componente y della freccia
        head_width=0.3 * robot_radius,
        head_length=0.4 * robot_radius,
        fc=arrow_color,
        ec=arrow_color,
        length_includes_head=True,
        zorder=4,
    )
    # ax.arrow ritorna un artista (FancyArrow) che posso rimuovere in seguito
    if isinstance(arr, Artist):
        artists.append(arr)

    return artists


def _default_save_path(title: str, *, subfolder: Optional[str] = None) -> Path:
    """Costruisce il percorso di salvataggio in img/ (o sotto-cartella) con titolo normalizzato + timestamp.
    - subfolder: percorso relativo dentro img/ (es. 'trajectories' o 'scans/rettilinea_v_costante')
    """
    project_root = Path(__file__).resolve().parents[1]
    img_dir = project_root / 'img'
    if subfolder:
        img_dir = img_dir / subfolder
    img_dir.mkdir(parents=True, exist_ok=True)
    base = _slugify(title)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return img_dir / f"{base}_{stamp}.png"


def _slugify(text: str) -> str:
    """Normalizza un testo per uso in nomi file/cartelle: minuscole, _ e - consentiti."""
    base = (text or '').lower().strip() or 'traiettoria'
    base = re.sub(r'\s+', '_', base)
    base = re.sub(r'[^a-z0-9_\-]', '', base)
    return base


# ----------------------- Pulizia output immagini -----------------------

def cleanup_output_images(*, subfolders: Sequence[str] = ("trajectories", "scans", "scans_polar"), remove_root: bool = False) -> None:
    """Elimina le immagini generate in precedenza sotto img/ per avere solo gli output dell'ultimo run.

    - subfolders: sottocartelle di img/ da pulire. Di default: trajectories, scans, scans_polar.
    - remove_root: se True, elimina l'intera cartella img/ e la ricrea vuota.
    """
    project_root = Path(__file__).resolve().parents[1]
    img_dir = project_root / 'img'
    if not img_dir.exists():
        return
    try:
        if remove_root:
            shutil.rmtree(img_dir, ignore_errors=True)
            img_dir.mkdir(parents=True, exist_ok=True)
        else:
            for sub in subfolders:
                target = img_dir / sub
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                    target.mkdir(parents=True, exist_ok=True)
        print(f"Pulizia immagini completata in: {img_dir}")
    except Exception as e:
        # Non bloccare l'esecuzione in caso di problemi di file system
        print(f"[cleanup_output_images] Avviso: non sono riuscito a pulire completamente {img_dir}: {e}")


# ----------------------- Fine pulizia output immagini -----------------------


def _robot_scale_from_history(history):
    """Deriva una scala per robot/freccia dall'estensione della traiettoria.
    Ritorna (robot_radius, dir_len)."""
    x_range = float(np.ptp(history[:, 0]))  # ampiezza su x
    y_range = float(np.ptp(history[:, 1]))  # ampiezza su y
    ref = max(x_range, y_range, 1.0)  # evita raggio nullo
    robot_radius = max(0.02, 0.012 * ref)  # raggio proporzionale all'estensione
    dir_len = 2.5 * robot_radius  # lunghezza freccia proporzionale al raggio
    return robot_radius, dir_len


def _compute_axes_limits_with_glyphs(history, step, r_robot, d_arrow, env: Optional[Environment] = None, *, fit_to: str = 'trajectory'):
    """Calcola i limiti degli assi includendo corpo, ruote, punte freccia.

    fit_to:
    - 'trajectory' (default): adatta i limiti alla traiettoria (più stretto, niente dezoom).
    - 'environment': estende i limiti per includere anche i bounds dell'ambiente.
    """
    xs = history[:, 0]
    ys = history[:, 1]
    # Estensione base della traiettoria
    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    # Margine legato al corpo + ruote esterne
    width, length = _rect_dims_from_radius(r_robot)
    w_r, w_off = _wheel_params(r_robot)
    body_half_diag = 0.5 * float(np.hypot(length, width))
    wheels_extra = float(w_off + w_r)
    extent_radius = body_half_diag + wheels_extra

    x_min -= extent_radius
    x_max += extent_radius
    y_min -= extent_radius
    y_max += extent_radius

    # Includi punte delle frecce valutate a intervalli e sempre quella finale
    n = len(history)
    step = max(1, int(step))
    indices = list(range(0, n, step))
    if (n - 1) not in indices and n > 0:
        indices.append(n - 1)
    for i in indices:
        x, y, th = map(float, history[i])
        tip_x = float(x + d_arrow * np.cos(th))
        tip_y = float(y + d_arrow * np.sin(th))
        x_min = min(x_min, tip_x)
        x_max = max(x_max, tip_x)
        y_min = min(y_min, tip_y)
        y_max = max(y_max, tip_y)

    # Opzionale: includi i bounds dell'ambiente solo se richiesto
    if fit_to == 'environment' and env is not None and getattr(env, 'bounds', None) is not None:
        try:
            bx, by = env.bounds.exterior.xy  # type: ignore[attr-defined]
            bx_min, bx_max = float(np.min(bx)), float(np.max(bx))
            by_min, by_max = float(np.min(by)), float(np.max(by))
            x_min = min(x_min, bx_min)
            x_max = max(x_max, bx_max)
            y_min = min(y_min, by_min)
            y_max = max(y_max, by_max)
        except Exception:
            pass

    # Piccolo margine finale per aria attorno al disegno
    pad = 0.02 * max(x_max - x_min, y_max - y_min, 1.0)
    return x_min - pad, x_max + pad, y_min - pad, y_max + pad


# Helper privato per disegnare una singola traiettoria statica sugli axes
# Centralizza la logica ripetuta in plot_trajectory, show_trajectories_carousel e save_trajectories_images
# Restituisce (r_robot, d_arrow) calcolati per la traiettoria

def _plot_static_trajectory_on_axes(
    ax,
    hist: np.ndarray,
    step: int,
    title: Optional[str] = None,
    include_title: bool = True,
    include_axis_labels: bool = True,
    *,
    draw_glyphs: bool = True,
    environment: Optional[Environment] = None,
    fit_to: str = 'trajectory',
):
    """Disegna lo sfondo dell'ambiente (opzionale), la linea della traiettoria e (opzionalmente) i robot statici sparsi.

    - draw_glyphs=False è usato nel viewer interattivo per non mostrare i robot statici
      finché non vengono “rivelati” durante la riproduzione.
    - fit_to controlla se i limiti assi si adattano alla sola traiettoria (default) o includono i bounds dell'ambiente.
    Ritorna (r_robot, d_arrow).
    """
    n = len(hist)
    step = max(1, int(step))

    # Disegna l'ambiente in background, se fornito (bounds e ostacoli)
    if environment is not None:
        environment.plot(ax=ax)

    # Traccia la traiettoria (linea nera) sopra lo sfondo
    ax.plot(hist[:, 0], hist[:, 1], '-', linewidth=1.5, color='k', zorder=2)
    # Scala robot/freccia coerente con l’estensione
    r_robot, d_arrow = _robot_scale_from_history(hist)

    if draw_glyphs:
        # Disegna i simboli del robot a intervalli regolari
        for i in range(0, n, step):
            if i == 0:
                body_col, arr_col, ctr_col = 'green', 'orange', 'green'  # partenza
            elif i == n - 1:
                body_col, arr_col, ctr_col = 'red', 'orange', 'red'      # arrivo
            else:
                body_col, arr_col, ctr_col = 'tab:blue', 'orange', 'orange'  # punti intermedi
            draw_robot(ax, hist[i], robot_radius=r_robot, dir_len=d_arrow, color=body_col, arrow_color=arr_col, center_color=ctr_col)
        # Assicura il disegno della posa finale anche se non multipla di step
        if n > 0 and ((n - 1) % step != 0 or n == 1):
            draw_robot(ax, hist[-1], robot_radius=r_robot, dir_len=d_arrow, color='red', arrow_color='orange', center_color='red')

    # Limiti assi calcolati in base alla scelta di fit
    x0, x1, y0, y1 = _compute_axes_limits_with_glyphs(hist, step, r_robot, d_arrow, env=environment, fit_to=fit_to)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Aspetto e labeling
    ax.set_aspect('equal', 'box')
    ax.grid(False)
    if include_axis_labels:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    if include_title and title is not None:
        ax.set_title(title)

    return r_robot, d_arrow


def _build_info_text(
    hist: np.ndarray,
    k_pose: int,
    dt: float,
    commands: Optional[np.ndarray] = None,
    *,
    use_cmd_of_prev: bool = True,
    show_next_pose: bool = False,
) -> str:
    """Crea il testo del pannello info (tempo, velocità, posa).

    - Se sono forniti comandi [v, w], vengono usati; altrimenti v e w sono stimati da differenze finite.
    - show_next_pose permette di mostrare la posa successiva (utile dopo un ridisegno statico).
    """
    N = int(len(hist) if hist is not None else 0)
    dt = float(max(dt, 1e-9))  # evita divisioni per zero
    k_pose = int(max(0, min(k_pose, max(N - 1, 0))))

    # v, w: da comandi se disponibili, altrimenti stimati dal moto tra due pose
    if commands is not None and len(commands) > 0:
        cmd_idx = (k_pose - 1) if use_cmd_of_prev else k_pose
        cmd_idx = int(max(0, min(cmd_idx, len(commands) - 1)))
        v_k = float(commands[cmd_idx][0])
        w_k = float(commands[cmd_idx][1])
    else:
        if N >= 2:
            k2 = int(max(1, min(k_pose, N - 1)))
            k1 = k2 - 1
            dx = float(hist[k2][0] - hist[k1][0])
            dy = float(hist[k2][1] - hist[k1][1])
            dth = float(hist[k2][2] - hist[k1][2])
            v_k = (dx**2 + dy**2) ** 0.5 / dt
            dth = (dth + np.pi) % (2 * np.pi) - np.pi  # normalizza in [-π, π)
            w_k = dth / dt
        else:
            v_k = 0.0
            w_k = 0.0

    # Tempo e posa (corrente o successiva)
    t_k = float(k_pose) * dt
    if show_next_pose and N > 0:
        pose_idx = int(min(k_pose + 1, N - 1))
    else:
        pose_idx = int(k_pose)

    if N > 0:
        x_k, y_k, th_k = map(float, hist[pose_idx])
    else:
        x_k = y_k = th_k = 0.0

    info_text = (
        f"t={t_k:.2f} s\n"
        f"v={v_k:.2f} m/s,  ω={w_k:.2f} rad/s\n"
        f"x={x_k:.2f} m,  y={y_k:.2f} m,  α={th_k:.2f} rad"
    )
    return info_text


def _update_info_artist(fig, info_artist: Optional[Text], info_text: str) -> Text:
    """Aggiorna il box info (rimuove il precedente se esiste e crea un nuovo fig.text)."""
    if info_artist is not None:
        _safe_remove_artist(info_artist)
    return fig.text(
        0.98,   # allineato a destra
        0.96,   # alto
        info_text,
        ha='right',
        va='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'),
    )


def _update_error_artist(fig, err_artist: Optional[Text], msg: Optional[str]) -> Optional[Text]:
    """Mostra un messaggio di errore in alto al centro; se msg è None rimuove l'artista."""
    if err_artist is not None:
        _safe_remove_artist(err_artist)
        err_artist = None
    if msg:
        err_artist = fig.text(
            0.5, 0.98, msg,
            ha='center', va='top', fontsize=11, color='crimson', fontweight='bold',
        )
    return err_artist


def _draw_lidar_rays(ax, origin_xy, lidar_points: np.ndarray, *, ray_color: str = 'tab:red', hit_marker_color: str = 'tab:red', alpha: float = 0.35) -> List[Artist]:
    """Disegna i raggi LiDAR come segmenti dall'origine ai punti misurati; ritorna gli artisti per pulizia."""
    x0, y0 = map(float, origin_xy[:2])
    arts: List[Artist] = []
    # Linee dei raggi
    for px, py in lidar_points:
        ln = ax.plot([x0, float(px)], [y0, float(py)], color=ray_color, alpha=alpha, linewidth=0.8, zorder=1.5)[0]
        arts.append(ln)
    # Marker sui punti di impatto (leggeri)
    scat = ax.scatter(lidar_points[:, 0], lidar_points[:, 1], s=5, c=hit_marker_color, alpha=min(1.0, alpha + 0.20), zorder=2)
    if isinstance(scat, Artist):
        arts.append(scat)
    return arts


def _draw_lidar_hits(ax, hit_points: Optional[np.ndarray], *, marker_color: str = 'tab:red', alpha: float = 0.7, size: float = 10.0) -> List[Artist]:
    """Disegna SOLO i punti di impatto LiDAR come un unico scatter; ritorna la lista con l'artista creato."""
    arts: List[Artist] = []
    if hit_points is None or len(hit_points) == 0:
        return arts
    scat = ax.scatter(hit_points[:, 0], hit_points[:, 1], s=size, c=marker_color, alpha=alpha, zorder=2)
    if isinstance(scat, Artist):
        arts.append(scat)
    return arts


def plot_trajectory(history, show_orient_every=20, title="Traiettoria del robot", save_path=None, *, environment: Optional[Environment] = None, fit_to: str = 'trajectory', error_message: Optional[str] = None):
    """Plotta una singola traiettoria e (opzionalmente) salva l'immagine PNG.

    Nota: l'overlay d'errore non viene mostrato nelle immagini statiche; il messaggio appare solo nel viewer al momento della collisione.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    step = max(1, int(show_orient_every))
    _plot_static_trajectory_on_axes(ax, history, step=step, title=title, include_title=True, include_axis_labels=True, environment=environment, fit_to=fit_to)
    out_path = Path(save_path) if save_path else _default_save_path(title, subfolder='trajectories')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.show()


def _interp_pose(p0, p1, alpha: float):
    """Interpolazione lineare di posa (x,y,theta) con wrapping angolare.
    alpha in [0,1]."""
    alpha = float(max(0.0, min(1.0, alpha)))
    x0, y0, t0 = map(float, p0)
    x1, y1, t1 = map(float, p1)
    dx = x1 - x0
    dy = y1 - y0
    # differenza angolare normalizzata in [-pi, pi)
    dth = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi
    x = x0 + alpha * dx
    y = y0 + alpha * dy
    th = t0 + alpha * dth
    # normalizza
    th = (th + np.pi) % (2 * np.pi) - np.pi
    return np.array([x, y, th], dtype=float)


def show_trajectories_carousel(
    histories,
    titles,
    show_orient_every=20,
    save_each=False,
    commands_list=None,
    dts=None,
    show_info=False,
    show_legend=True,
    *,
    environment: Optional[Union[Environment, Sequence[Optional[Environment]]]] = None,
    fit_to: str = 'trajectory',
    error_messages: Optional[Sequence[Optional[str]]] = None,
    stop_indices: Optional[Sequence[Optional[int]]] = None,
    stop_fractions: Optional[Sequence[Optional[float]]] = None,
    lidar: Optional[Union[Lidar, Sequence[Optional[Lidar]]]] = None,
    show_lidar: bool = True,
    lidar_every: int = 1,
):
    """Viewer interattivo per più traiettorie con pulsanti e Play/Pausa.

    - error_messages: messaggi opzionali da mostrare SOLO quando si raggiunge la collisione.
    - stop_indices: indice (per-traiettoria) a cui fermare il player (collisione); None => nessun blocco.
    - stop_fractions: frazione temporale tra stop_indices-1 e stop_indices dove avviene l'impatto (0..1].
    - lidar: singolo sensore o lista per-traiettoria; se presente, disegna i raggi del frame corrente.
    - lidar_every: aggiorna la visualizzazione LiDAR ogni N frame (default 1 = ogni frame).
    """
    assert len(histories) == len(titles) and len(histories) > 0, "Liste vuote o di diversa lunghezza"
    if isinstance(show_orient_every, (list, tuple, np.ndarray)):
        assert len(show_orient_every) == len(histories), "show_orient_every deve avere stessa lunghezza di delle traiettorie"
    if commands_list is not None:
        assert len(commands_list) == len(histories), "commands_list deve avere stessa lunghezza di histories"
    if error_messages is not None:
        assert len(error_messages) == len(histories), "error_messages deve avere stessa lunghezza di histories"
    if stop_indices is not None:
        assert len(stop_indices) == len(histories), "stop_indices deve avere stessa lunghezza di histories"
    if stop_fractions is not None:
        assert len(stop_fractions) == len(histories), "stop_fractions deve avere stessa lunghezza di histories"
    if isinstance(lidar, (list, tuple)):
        assert len(lidar) == len(histories), "lidar (lista) deve avere stessa lunghezza di histories"

    # Normalizza dts a lista per uso uniforme
    if dts is None:
        dts_resolved = [1.0] * len(histories)
    elif isinstance(dts, (list, tuple, np.ndarray)):
        assert len(dts) == len(histories), "dts deve avere stessa lunghezza di histories"
        dts_resolved = [float(x) for x in dts]
    else:
        dts_resolved = [float(dts)] * len(histories)

    def _resolve_show_every(idx: int) -> int:
        """Ritorna lo step da usare per la traiettoria idx (singolo valore o per-traiettoria)."""
        if isinstance(show_orient_every, (list, tuple, np.ndarray)):
            return max(1, int(show_orient_every[idx]))
        return max(1, int(show_orient_every))

    def _resolve_env(idx: int) -> Optional[Environment]:
        """Ritorna l'Environment per la traiettoria idx (singolo o per-traiettoria)."""
        if environment is None:
            return None
        if isinstance(environment, (list, tuple)):
            assert len(environment) == len(histories), "environment (lista) deve avere stessa lunghezza di histories"
            return environment[idx]
        return environment

    def _resolve_lidar(idx: int) -> Optional[Lidar]:
        if lidar is None:
            return None
        if isinstance(lidar, (list, tuple)):
            return lidar[idx]
        return lidar

    # Figura principale (spazio extra sotto per i pulsanti)
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.18)

    # Legenda esplicativa in alto a sinistra (se richiesta)
    if show_legend:
        legend_text = (
            "Legenda:\n"
            "t: tempo [s]\n"
            "v: velocità lineare [m/s]\n"
            "ω: velocità angolare [rad/s]\n"
            "x, y: posizione [m]\n"
            "α: orientamento [rad]"
        )
        fig.text(
            0.02, 0.96, legend_text,
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'),
        )

    # Stato del viewer
    state = {"idx": 0, "show_info": bool(show_info), "playing": False, "frame": 0, "end_mark_drawn": False}
    info_artist: Optional[Text] = None
    moving_artists: List[Artist] = []
    moving_lidar_artists: List[Artist] = []
    err_artist: Optional[Text] = None
    static_start_artists: List[Artist] = []  # fantasma start (verde)
    static_end_artists: List[Artist] = []    # fantasma end (rosso), mostrato solo a fine traiettoria

    def _clear_artists(lst):
        """Rimuove e svuota in sicurezza una lista di artisti."""
        if not lst:
            return
        for art in lst:
            with suppress(Exception):
                art.remove()
        lst.clear()

    # Timer per avanzamento automatico; l'intervallo verrà aggiornato in base al dt della traiettoria corrente
    timer = fig.canvas.new_timer(interval=int(dts_resolved[0] * 1000))

    def _set_timer_interval_for_current():
        """Aggiorna l'intervallo del timer in ms in base al dt della traiettoria corrente."""
        cur_dt = max(1e-6, float(dts_resolved[state["idx"]]))
        interval_ms = int(round(cur_dt * 1000))
        with suppress(Exception):
            timer.interval = interval_ms
        with suppress(Exception):
            if hasattr(timer, 'set_interval'):
                timer.set_interval(interval_ms)

    def _clear_robot_moving():
        nonlocal moving_artists
        _clear_artists(moving_artists)

    def _clear_lidar_moving():
        nonlocal moving_lidar_artists
        _clear_artists(moving_lidar_artists)

    def _clear_static_start():
        nonlocal static_start_artists
        _clear_artists(static_start_artists)

    def _clear_static_end():
        nonlocal static_end_artists
        _clear_artists(static_end_artists)

    def _draw_static_start(hist: np.ndarray):
        """Disegna il robot statico di partenza (verde)."""
        nonlocal static_start_artists
        _clear_static_start()
        if hist is None or len(hist) == 0:
            return
        r_robot, d_arrow = _robot_scale_from_history(hist)
        static_start_artists += draw_robot(ax, hist[0], robot_radius=r_robot, dir_len=d_arrow, color='green', arrow_color='orange', center_color='green')

    def _draw_static_end(hist: np.ndarray):
        """Disegna il robot statico di arrivo (rosso)."""
        nonlocal static_end_artists
        _clear_static_end()
        if hist is None or len(hist) == 0:
            return
        r_robot, d_arrow = _robot_scale_from_history(hist)
        static_end_artists += draw_robot(ax, hist[-1], robot_radius=r_robot, dir_len=d_arrow, color='red', arrow_color='orange', center_color='red')

    def _draw_lidar_for_pose(pose_k, *, force: bool = True):
        nonlocal moving_lidar_artists
        if not show_lidar:
            return
        env_cur = _resolve_env(state["idx"])
        lid = _resolve_lidar(state["idx"])  # per-traiettoria o singolo
        if env_cur is None or lid is None:
            return
        try:
            # Solo hit reali: disegna raggi+marker esclusivamente verso i punti di impatto
            hit_pts = lid.scan_hits(pose_k, env_cur, frame='world')
            if hit_pts is not None and len(hit_pts) > 0:
                _clear_lidar_moving()
                moving_lidar_artists = _draw_lidar_rays(ax, pose_k, hit_pts)
            elif force:
                # nessun punto: pulisci se richiesto
                _clear_lidar_moving()
        except Exception:
            pass

    def _draw_moving_at(k: int, *, update_lidar: bool = True):
        """Disegna il robot mobile alla posa k. LiDAR aggiornato opzionalmente."""
        nonlocal moving_artists
        _clear_robot_moving()
        hist = histories[state["idx"]]
        k = int(max(0, min(k, len(hist) - 1)))
        r_robot, d_arrow = _robot_scale_from_history(hist)
        # Colore in base al frame corrente: verde (start), blu (intermedio), rosso (fine)
        is_first = (k == 0)
        is_last = (k == len(hist) - 1) if len(hist) > 0 else False
        if is_first:
            body_col = 'green'
            center_col = 'green'
        elif is_last:
            body_col = 'red'
            center_col = 'red'
        else:
            body_col = 'tab:blue'
            center_col = 'orange'
        moving_artists = draw_robot(ax, hist[k], robot_radius=r_robot, dir_len=d_arrow, color=body_col, arrow_color='orange', center_color=center_col)
        # Mantieni visibile lo start; mostra l'end solo se siamo all'ultimo frame
        _draw_static_start(hist)
        if is_last and not state["end_mark_drawn"]:
            _draw_static_end(hist)
            state["end_mark_drawn"] = True
        if update_lidar:
            _draw_lidar_for_pose(hist[k], force=False)

    # Nota: nessun "fantasma" nel viewer; lasciati solo nelle immagini salvate.

    # Centralizza la logica di disegno per la traiettoria corrente

    def draw_current():
        nonlocal info_artist, err_artist
        state["playing"] = False
        with suppress(Exception):
            timer.stop()
        _clear_robot_moving()
        _clear_lidar_moving()
        _clear_static_start()
        _clear_static_end()
        ax.clear()
        hist = histories[state["idx"]]
        title = titles[state["idx"]]
        n = len(hist)
        step = _resolve_show_every(state["idx"])
        env_cur = _resolve_env(state["idx"])
        # Pulisci eventuale errore precedente
        err_artist = _update_error_artist(fig, err_artist, None)

        _plot_static_trajectory_on_axes(ax, hist, step=step, title=title, include_title=True, include_axis_labels=True, draw_glyphs=False, environment=env_cur, fit_to=fit_to)

        if state["show_info"]:
            idxc = state["idx"]
            dt_cur = dts_resolved[idxc]
            last_draw_idx = ((n - 1) // step) * step if n > 0 else 0
            cmds = commands_list[idxc] if commands_list is not None else None
            info_text = _build_info_text(hist, k_pose=int(last_draw_idx), dt=float(dt_cur), commands=cmds, use_cmd_of_prev=False, show_next_pose=True)
            info_artist = _update_info_artist(fig, info_artist, info_text)
        else:
            _safe_remove_artist(info_artist)
            info_artist = None

        state["frame"] = 0
        # Disegna lo start verde; l'end rosso comparirà solo alla fine
        _draw_static_start(hist)
        state["end_mark_drawn"] = False
        # Robot mobile al frame 0 + LiDAR se conforme a lidar_every
        should_update_lidar = (0 % max(1, int(lidar_every)) == 0)
        _draw_moving_at(0, update_lidar=should_update_lidar)
        _set_timer_interval_for_current()
        fig.canvas.draw_idle()

        if save_each:
            fig_save, ax_save = plt.subplots(figsize=(7, 7))
            _plot_static_trajectory_on_axes(ax_save, hist, step=step, title=title, include_title=True, include_axis_labels=True, draw_glyphs=True, environment=env_cur, fit_to=fit_to)
            out_path = _default_save_path(title, subfolder='trajectories')
            fig_save.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig_save)

    def _stop_if_collision_reached(next_k: int) -> bool:
        """Se c'è una collisione definita e la raggiungiamo, ferma il player, mostra il messaggio e ritorna True."""
        nonlocal err_artist, moving_artists, moving_lidar_artists
        idxc = state["idx"]
        if stop_indices is None:
            return False
        stop_k = stop_indices[idxc]
        if stop_k is None:
            return False
        if next_k >= int(stop_k):
            # Fermati esattamente al frame di collisione; disegna alla posa interpolata appena prima dell'impatto
            kcol = int(stop_k)
            hist = histories[idxc]
            # Frazione di collisione (se fornita), altrimenti 1.0 (urto esattamente su kcol)
            frac = 1.0 if stop_fractions is None or stop_fractions[idxc] is None else float(stop_fractions[idxc])
            # Disegna alla posa tra kcol-1 e kcol (o esattamente su kcol se kcol==0)
            if kcol >= 1:
                alpha_safe = max(0.0, min(1.0, frac - 1e-3))  # un pelo prima dell'impatto per evitare compenetrazione visiva
                pose = _interp_pose(hist[kcol - 1], hist[kcol], alpha_safe)
                # Aggiorna moving alla posa interpolata
                _clear_robot_moving()
                r_robot, d_arrow = _robot_scale_from_history(hist)
                moving_artists = draw_robot(ax, pose, robot_radius=r_robot, dir_len=d_arrow, color='tab:blue', arrow_color='orange', center_color='orange')
                _draw_lidar_for_pose(pose)
                # In collisione non disegnare l'end rosso
                _draw_static_start(hist)
                state["frame"] = kcol  # stato logico fermo a kcol
            else:
                # Collisione alla posa iniziale
                _clear_robot_moving()
                r_robot, d_arrow = _robot_scale_from_history(hist)
                moving_artists = draw_robot(ax, hist[0], robot_radius=r_robot, dir_len=d_arrow, color='tab:blue', arrow_color='orange', center_color='orange')
                _draw_lidar_for_pose(hist[0])
                _draw_static_start(hist)
                state["frame"] = 0
            # Mostra messaggio d'errore per questa traiettoria
            default_msg = "Ostacolo lungo la traiettoria"
            msg = (error_messages[idxc] if (error_messages is not None) else default_msg)
            err_artist = _update_error_artist(fig, err_artist, msg)
            # Metti in pausa e aggiorna pulsante
            state["playing"] = False
            with suppress(Exception):
                timer.stop()
            fig.canvas.draw_idle()
            return True
        return False

    def _on_timer():
        nonlocal info_artist, err_artist
        idxc = state["idx"]
        hist = histories[idxc]
        n = len(hist)
        k_next = state["frame"] + 1
        # Controlla collisione prima di avanzare
        if _stop_if_collision_reached(k_next):
            return
        if k_next >= n:
            state["playing"] = False
            with suppress(Exception):
                timer.stop()
            _set_play_label('▶ Play')
            return
        # Avanza frame e aggiorna
        state["frame"] = k_next
        # Robot ogni frame
        _draw_moving_at(k_next, update_lidar=False)
        # LiDAR solo ogni N frame
        if (int(k_next) % max(1, int(lidar_every)) == 0):
            _draw_lidar_for_pose(hist[k_next], force=True)
        if state["show_info"]:
            with suppress(Exception):
                dt_cur = float(dts_resolved[idxc])
                cmds = commands_list[idxc] if commands_list is not None else None
                info_text = _build_info_text(hist, k_pose=int(k_next), dt=dt_cur, commands=cmds, use_cmd_of_prev=True, show_next_pose=False)
                info_artist = _update_info_artist(fig, info_artist, info_text)
        fig.canvas.draw_idle()

    timer.add_callback(_on_timer)

    # Pulsanti con icone Unicode (compatibili su Windows)
    ax_prev = fig.add_axes((0.12, 0.05, 0.18, 0.08))
    btn_prev = Button(ax_prev, '◀◀ Precedente')

    ax_play = fig.add_axes((0.34, 0.05, 0.18, 0.08))
    btn_play = Button(ax_play, '▶ Play')

    ax_next = fig.add_axes((0.56, 0.05, 0.18, 0.08))
    btn_next = Button(ax_next, 'Successivo ▶▶')

    def _set_play_label(text: str):
        with suppress(Exception):
            btn_play.label.set_text(text)

    def _navigate(delta: int):
        """Cambia traiettoria (delta=-1 precedente, +1 successiva) e ridisegna."""
        state["idx"] = (state["idx"] + int(delta)) % len(histories)
        state["playing"] = False
        with suppress(Exception):
            timer.stop()
        _set_play_label('▶ Play')
        draw_current()

    def on_play(_event):
        """Toggle Play/Pausa: avvia/ferma il timer e aggiorna l'etichetta del pulsante."""
        # Se già raggiunta collisione, resta in pausa
        if stop_indices is not None:
            stop_k = stop_indices[state["idx"]]
            if stop_k is not None and state["frame"] >= int(stop_k):
                return
        if not state["playing"]:
            state["playing"] = True
            _set_timer_interval_for_current()
            with suppress(Exception):
                timer.start()
            _set_play_label('▮▮ Pausa')
        else:
            state["playing"] = False
            with suppress(Exception):
                timer.stop()
            _set_play_label('▶ Play')

    # Collega i pulsanti
    btn_prev.on_clicked(lambda _event: _navigate(-1))
    btn_play.on_clicked(on_play)
    btn_next.on_clicked(lambda _event: _navigate(+1))

    # Disegna subito la prima traiettoria e mostra
    draw_current()
    plt.show()


# ----------------------- API di salvataggio immagini -----------------------

def save_trajectories_images(
    histories,
    titles,
    show_orient_every=20,
    *,
    environment: Optional[Union[Environment, Sequence[Optional[Environment]]]] = None,
    fit_to: str = 'trajectory',
    error_messages: Optional[Sequence[Optional[str]]] = None,
    progress_cb: Optional[callable] = None,
    quiet: bool = True,
):
    """Salva PNG per ciascuna traiettoria, con simboli del robot (inclusi start verde, intermedi blu e end rosso).

    - histories: lista di array (N_i,3) per ciascuna traiettoria
    - titles: lista di titoli per i file
    - show_orient_every: passo con cui disegnare i simboli lungo la traiettoria (può essere lista per-caso)
    - environment: singolo Environment o lista per-caso; se fornito, disegna bounds/ostacoli sullo sfondo
    - fit_to: 'trajectory' o 'environment'
    - progress_cb: funzione opzionale (cur, total) chiamata dopo ogni salvataggio
    - quiet: True per non stampare i path dei file salvati
    """
    assert len(histories) == len(titles) and len(histories) > 0, "Liste vuote o di diversa lunghezza"
    total = len(histories)
    if isinstance(show_orient_every, (list, tuple, np.ndarray)):
        assert len(show_orient_every) == len(histories), "show_orient_every deve avere stessa lunghezza di histories"

    def _resolve_show_every(idx: int) -> int:
        if isinstance(show_orient_every, (list, tuple, np.ndarray)):
            return max(1, int(show_orient_every[idx]))
        return max(1, int(show_orient_every))

    def _resolve_env(idx: int) -> Optional[Environment]:
        if environment is None:
            return None
        if isinstance(environment, (list, tuple)):
            assert len(environment) == len(histories), "environment (lista) deve avere stessa lunghezza di histories"
            return environment[idx]
        return environment

    for i, (hist, title_str) in enumerate(zip(histories, titles), start=1):
        fig, ax = plt.subplots(figsize=(7, 7))
        step = _resolve_show_every(i - 1)
        env_cur = _resolve_env(i - 1)
        # Disegno statico completo con "fantasmi"
        _plot_static_trajectory_on_axes(
            ax, hist, step=step, title=None, include_title=False, include_axis_labels=False,
            draw_glyphs=True, environment=env_cur, fit_to=fit_to,
        )
        out_path = _default_save_path(title_str, subfolder='trajectories')
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        if callable(progress_cb):
            with suppress(Exception):
                progress_cb(i, total)
        plt.close(fig)


def save_lidar_scans_images(
    history: np.ndarray,
    title: str,
    lidar: Lidar,
    environment: Optional[Environment],
    dt: float,
    *,
    interval_s: float = 1.0,
    fit_to: str = 'environment',
    show_info: bool = True,
    progress_cb: Optional[callable] = None,
    quiet: bool = True,
) -> None:
    """Salva immagini delle scansioni LiDAR a intervalli regolari lungo una singola traiettoria.

    Visualizza gli ostacoli/bounds dell'ambiente come sfondo e, sopra, il robot alla posa corrente
    e le linee dei raggi che colpiscono (hit). Niente raggi dei miss.
    Se show_info=True, aggiunge un riquadro con: tempo della scansione e posa (x,y,α).
    """
    if history is None or len(history) == 0:
        return

    step_idx = max(1, int(round(float(interval_s) / max(1e-9, float(dt)))))
    N = len(history)
    total = len(range(0, N, step_idx))
    case_folder = f"scans/{_slugify(title)}"

    def _set_axes_limits_scan(ax, env: Optional[Environment], pts: Optional[np.ndarray]):
        # Se c'è un environment con bounds, usa quelli per includere tutti gli ostacoli
        if env is not None and getattr(env, 'bounds', None) is not None:
            try:
                bx, by = env.bounds.exterior.xy  # type: ignore[attr-defined]
                x_min, x_max = float(np.min(bx)), float(np.max(bx))
                y_min, y_max = float(np.min(by)), float(np.max(by))
            except Exception:
                x_min = y_min = -1.0; x_max = y_max = 1.0
        else:
            # Fallback: usa i soli punti
            if pts is not None and len(pts) > 0:
                x_min = float(np.min(pts[:, 0])); x_max = float(np.max(pts[:, 0]))
                y_min = float(np.min(pts[:, 1])); y_max = float(np.max(pts[:, 1]))
                if x_max - x_min < 1e-6:
                    x_min -= 0.5; x_max += 0.5
                if y_max - y_min < 1e-6:
                    y_min -= 0.5; y_max += 0.5
            else:
                x_min, x_max, y_min, y_max = -1.0, 1.0, -1.0, 1.0
        pad = 0.04 * max(x_max - x_min, y_max - y_min, 1.0)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect('equal', 'box')

    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / 'img' / case_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx_k, k in enumerate(range(0, N, step_idx), start=1):
        pose = history[k]
        try:
            scan_pts, ranges = lidar.scan(pose, environment, return_ranges=True) if environment is not None else (None, None)
        except Exception:
            scan_pts, ranges = None, None
        if scan_pts is not None and ranges is not None:
            mask_hits = np.asarray(ranges) < float(lidar.r_max) - 1e-12
            hit_points = np.asarray(scan_pts)[mask_hits]
        else:
            hit_points = None

        fig2, ax2 = plt.subplots(figsize=(7, 7))
        if environment is not None:
            with suppress(Exception):
                environment.plot(ax=ax2)
        # Limiti assi basati sui bounds dell'ambiente se disponibile, altrimenti sui punti
        _set_axes_limits_scan(ax2, environment, hit_points)

        # Disegna le linee dei raggi SOLO verso i punti di impatto (hit)
        if hit_points is not None and len(hit_points) > 0:
            _draw_lidar_rays(ax2, pose, hit_points, ray_color='tab:red', hit_marker_color='tab:red', alpha=0.40)

        # Colori del robot per prima/ultima scansione salvata
        last_k = ((N - 1) // step_idx) * step_idx
        is_first = (k == 0)
        is_last = (k == last_k)
        body_col = 'green' if is_first else ('red' if is_last else 'tab:blue')
        center_col = 'green' if is_first else ('red' if is_last else 'orange')

        # Disegna il robot alla posa corrente, con scala coerente all'estensione degli assi
        try:
            x0, x1 = ax2.get_xlim(); y0, y1 = ax2.get_ylim()
            ref = max(float(x1 - x0), float(y1 - y0), 1.0)
            robot_radius = max(0.02, 0.012 * ref)
            dir_len = 2.5 * robot_radius
        except Exception:
            robot_radius = 0.08
            dir_len = 2.5 * robot_radius
        draw_robot(ax2, pose, robot_radius=robot_radius, dir_len=dir_len, color=body_col, arrow_color='orange', center_color=center_col)

        # Rimuovi ogni elemento di contorno/assi e legende per eliminare spazi bianchi
        ax2.set_axis_off()
        with suppress(Exception):
            leg = ax2.get_legend()
            if leg is not None:
                leg.remove()
        with suppress(Exception):
            ax2.margins(0)
        with suppress(Exception):
            fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Overlay informativo opzionale (solo tempo e posa) ancorato agli assi per non espandere il bbox
        if show_info:
            t = float(k) * float(dt)
            x, y, th = map(float, pose)
            th_deg = (np.degrees(th) + 360.0) % 360.0
            info_text = (
                f"t={t:.2f} s\n"
                f"x={x:.2f} m, y={y:.2f} m, α={th_deg:.0f}°"
            )
            ax2.text(
                0.98, 0.98, info_text,
                transform=ax2.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7')
            )

        # Salvataggio
        t = float(k) * float(dt)
        filename_base = f"scan_t{t:.2f}s"
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        out_path_pts = out_dir / f"{_slugify(title)}_{filename_base}_points_{stamp}.png"
        fig2.savefig(out_path_pts, dpi=120, bbox_inches='tight', pad_inches=0.01)
        if callable(progress_cb):
            with suppress(Exception):
                progress_cb(idx_k, total)
        plt.close(fig2)


def save_lidar_polar_images(
    history: np.ndarray,
    title: str,
    lidar: Lidar,
    environment: Optional[Environment],
    dt: float,
    *,
    interval_s: float = 1.0,
    include_misses: bool = True,
    progress_cb: Optional[callable] = None,
    quiet: bool = True,
) -> None:
    """Salva grafici r(θ) delle scansioni LiDAR lungo una traiettoria.

    - Asse x: θ (angolo relativo del raggio rispetto al frame del LiDAR), ora in gradi 0..360
    - Asse y: r (distanza misurata), in metri
    - Mostra i colpi reali (hit) e, opzionalmente, anche i miss (raggi a r_max) con colore differente.
    """
    if history is None or len(history) == 0:
        return
    step_idx = max(1, int(round(float(interval_s) / max(1e-9, float(dt)))))
    N = len(history)
    total = len(range(0, N, step_idx))

    # Precalcolo degli angoli relativi dei raggi (come in Lidar.scan), convertiti in gradi 0..360
    half = 0.5 * float(lidar.angle_span)
    rel_angles = np.linspace(-half, half, num=lidar.n_rays, endpoint=True)
    rel_angles_deg = (np.degrees(rel_angles) + 360.0) % 360.0

    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / 'img' / f"scans_polar/{_slugify(title)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx_k, k in enumerate(range(0, N, step_idx), start=1):
        pose = history[k]
        try:
            _pts, ranges = lidar.scan(pose, environment, return_ranges=True)
        except Exception:
            continue
        ranges = np.asarray(ranges)
        mask_hit = ranges < float(lidar.r_max) - 1e-12
        mask_miss = ~mask_hit
        th_hit = rel_angles_deg[mask_hit]
        rr_hit = ranges[mask_hit]
        th_miss = rel_angles_deg[mask_miss]
        rr_miss = ranges[mask_miss]

        fig, ax = plt.subplots(figsize=(7, 4))
        # Punti di hit
        if th_hit.size > 0:
            ax.scatter(th_hit, rr_hit, s=10, c='tab:blue', alpha=0.95, label='hit')
        # Punti di miss (a r_max)
        if include_misses and th_miss.size > 0:
            ax.scatter(th_miss, rr_miss, s=8, c='tab:gray', alpha=0.6, label='miss (r_max)')
        ax.set_xlabel("θ [°]")
        ax.set_ylabel("r [m]")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"r(θ) – {title} – t={float(k)*float(dt):.2f} s")
        # Limiti y: 0..r_max con piccolo margine
        y_max = float(lidar.r_max)
        ax.set_ylim(-0.02 * y_max, 1.02 * y_max)
        # Limiti x in gradi: 0..360
        ax.set_xlim(0.0 - 1e-3, 360.0 + 1e-3)
        try:
            ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
        except Exception:
            pass
        # Legenda se almeno una serie è presente
        if (th_hit.size > 0) or (include_misses and th_miss.size > 0):
            ax.legend(loc='upper right', framealpha=0.85, fontsize=8)
        out_path = out_dir / f"{_slugify(title)}_polar_t{float(k)*float(dt):.2f}s_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        if callable(progress_cb):
            with suppress(Exception):
                progress_cb(idx_k, total)
        plt.close(fig)

