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


def _default_save_path(title: str) -> Path:
    """Costruisce il percorso di salvataggio in img/ con titolo normalizzato + timestamp."""
    # project_root = cartella padre di 'src'
    project_root = Path(__file__).resolve().parents[1]
    img_dir = project_root / 'img'
    img_dir.mkdir(parents=True, exist_ok=True)  # Assicura l'esistenza della cartella
    # Normalizzazione del titolo per un nome file pulito
    base = title.lower().strip() or 'traiettoria'
    base = re.sub(r'\s+', '_', base)
    base = re.sub(r'[^a-z0-9_\-]', '', base)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return img_dir / f"{base}_{stamp}.png"


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
        f"x={x_k:.2f} m,  y={y_k:.2f} m,  ϑ={th_k:.2f} rad"
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


def plot_trajectory(history, show_orient_every=20, title="Traiettoria del robot", save_path=None, *, environment: Optional[Environment] = None, fit_to: str = 'trajectory', error_message: Optional[str] = None):
    """Plotta una singola traiettoria e (opzionalmente) salva l'immagine PNG.

    Nota: l'overlay d'errore non viene mostrato nelle immagini statiche; il messaggio appare solo nel viewer al momento della collisione.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    step = max(1, int(show_orient_every))
    _plot_static_trajectory_on_axes(ax, history, step=step, title=title, include_title=True, include_axis_labels=True, environment=environment, fit_to=fit_to)
    out_path = Path(save_path) if save_path else _default_save_path(title)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Figura salvata in: {out_path}")
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
):
    """Viewer interattivo per più traiettorie con pulsanti e Play/Pausa.

    - error_messages: messaggi opzionali da mostrare SOLO quando si raggiunge la collisione.
    - stop_indices: indice (per-traiettoria) a cui fermare il player (collisione); None => nessun blocco.
    - stop_fractions: frazione temporale tra stop_indices-1 e stop_indices dove avviene l'impatto (0..1].
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

    # Figura principale (spazio extra sotto per i pulsanti)
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.18)

    # Stato del viewer
    state = {"idx": 0, "show_info": bool(show_info), "playing": False, "frame": 0, "revealed": set()}
    info_artist: Optional[Text] = None
    moving_artists: List[Artist] = []
    err_artist: Optional[Text] = None

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

    def _clear_moving():
        """Rimuove l'istanza mobile (tutti i suoi artisti)."""
        nonlocal moving_artists
        _clear_artists(moving_artists)

    def _draw_moving_at(k: int):
        """Disegna il robot mobile alla posa k, sostituendo quello precedente."""
        nonlocal moving_artists
        _clear_moving()
        hist = histories[state["idx"]]
        k = int(max(0, min(k, len(hist) - 1)))
        r_robot, d_arrow = _robot_scale_from_history(hist)
        # draw_robot ritorna gli artisti creati, li conserviamo per rimozione al frame successivo
        moving_artists = draw_robot(ax, hist[k], robot_radius=r_robot, dir_len=d_arrow, color='tab:blue', arrow_color='orange', center_color='orange')

    def _reveal_static_as_needed(k: int):
        """Disegna i robot statici per tutti gli indici <= k, se non già disegnati (rivelazione progressiva)."""
        hist = histories[state["idx"]]
        n = len(hist)
        if n <= 0:
            return
        step_i = _resolve_show_every(state["idx"]) if isinstance(show_orient_every, (list, tuple, np.ndarray)) else max(1, int(show_orient_every))
        # Indici su cui vogliamo simboli (come nelle immagini salvate)
        idxs = list(range(0, n, step_i))
        if (n - 1) not in idxs:
            idxs.append(n - 1)
        r_robot, d_arrow = _robot_scale_from_history(hist)
        for i in idxs:
            # Se l'indice è oltre k o già rivelato, salta
            if i > int(k) or i in state["revealed"]:
                continue
            # Colori coerenti con inizio/fine/intermedio
            if i == 0:
                body_col, arr_col, ctr_col = 'green', 'orange', 'green'
            elif i == n - 1:
                body_col, arr_col, ctr_col = 'red', 'orange', 'red'
            else:
                body_col, arr_col, ctr_col = 'tab:blue', 'orange', 'orange'
            draw_robot(ax, hist[i], robot_radius=r_robot, dir_len=d_arrow, color=body_col, arrow_color=arr_col, center_color=ctr_col)
            state["revealed"].add(i)

    # Box legenda (facoltativo) in alto a sinistra fuori dall'axes
    if show_legend:
        legend_text = (
            "Legenda:\n"
            "t: tempo [s]\n"
            "v: velocità lineare [m/s]\n"
            "ω: velocità angolare [rad/s]\n"
            "x, y: posizione [m]\n"
            "θ: orientamento [rad]"
        )
        fig.text(0.02, 0.96, legend_text, ha='left', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'))

    def draw_current():
        nonlocal info_artist, err_artist
        state["playing"] = False
        with suppress(Exception):
            timer.stop()
        _clear_moving()
        ax.clear()
        hist = histories[state["idx"]]
        title = titles[state["idx"]]
        n = len(hist)
        step = _resolve_show_every(state["idx"])
        env_cur = _resolve_env(state["idx"])
        # Pulisci eventuale errore precedente
        err_artist = _update_error_artist(fig, err_artist, None)

        _plot_static_trajectory_on_axes(ax, hist, step=step, title=title, include_title=True, include_axis_labels=True, draw_glyphs=False, environment=env_cur, fit_to=fit_to)

        state["revealed"] = set()
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
        _draw_moving_at(0)
        _reveal_static_as_needed(0)
        _set_timer_interval_for_current()
        fig.canvas.draw_idle()

        if save_each:
            fig_save, ax_save = plt.subplots(figsize=(7, 7))
            _plot_static_trajectory_on_axes(ax_save, hist, step=step, title=title, include_title=True, include_axis_labels=True, draw_glyphs=True, environment=env_cur, fit_to=fit_to)
            out_path = _default_save_path(title)
            fig_save.savefig(out_path, dpi=120, bbox_inches='tight')
            print(f"Figura salvata in: {out_path}")
            plt.close(fig_save)

    def _stop_if_collision_reached(next_k: int) -> bool:
        """Se c'è una collisione definita e la raggiungiamo, ferma il player, mostra il messaggio e ritorna True."""
        nonlocal err_artist, moving_artists
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
                _clear_moving()
                r_robot, d_arrow = _robot_scale_from_history(hist)
                moving_artists = draw_robot(ax, pose, robot_radius=r_robot, dir_len=d_arrow, color='tab:blue', arrow_color='orange', center_color='orange')
                # Rivela statici fino al frame precedente
                _reveal_static_as_needed(kcol - 1)
                state["frame"] = kcol  # stato logico fermo a kcol
            else:
                # Collisione alla posa iniziale
                _clear_moving()
                r_robot, d_arrow = _robot_scale_from_history(hist)
                moving_artists = draw_robot(ax, hist[0], robot_radius=r_robot, dir_len=d_arrow, color='tab:blue', arrow_color='orange', center_color='orange')
                _reveal_static_as_needed(0)
                state["frame"] = 0
            # Mostra messaggio d'errore per questa traiettoria
            msg = None if error_messages is None else error_messages[idxc]
            err_artist = _update_error_artist(fig, err_artist, msg)
            # Metti in pausa e aggiorna pulsante
            state["playing"] = False
            with suppress(Exception):
                timer.stop()
            with suppress(Exception):
                btn_play.label.set_text('▶ Play')
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
            with suppress(Exception):
                btn_play.label.set_text('▶ Play')
            return
        # Avanza frame e aggiorna
        state["frame"] = k_next
        _draw_moving_at(k_next)
        _reveal_static_as_needed(k_next)
        if state["show_info"]:
            with suppress(Exception):
                dt_cur = float(dts_resolved[idxc])
                cmds = commands_list[idxc] if commands_list is not None else None
                info_text = _build_info_text(hist, k_pose=int(k_next), dt=dt_cur, commands=cmds, use_cmd_of_prev=True, show_next_pose=False)
                info_artist = _update_info_artist(fig, info_artist, info_text)
        fig.canvas.draw_idle()

    timer.add_callback(_on_timer)

    # Pulsanti con icone Unicode (compatibili su Windows)
    ax_prev = fig.add_axes((0.16, 0.05, 0.22, 0.08))
    btn_prev = Button(ax_prev, '◀◀ Precedente')

    ax_play = fig.add_axes((0.40, 0.05, 0.20, 0.08))
    btn_play = Button(ax_play, '▶ Play')

    ax_next = fig.add_axes((0.64, 0.05, 0.24, 0.08))
    btn_next = Button(ax_next, 'Successivo ▶▶')

    def _navigate(delta: int):
        """Cambia traiettoria (delta=-1 precedente, +1 successiva) e ridisegna."""
        state["idx"] = (state["idx"] + int(delta)) % len(histories)
        state["playing"] = False
        with suppress(Exception):
            timer.stop()
        with suppress(Exception):
            btn_play.label.set_text('▶ Play')
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
            with suppress(Exception):
                btn_play.label.set_text('▮▮ Pausa')  # simbolo pausa compatibile con Windows
        else:
            state["playing"] = False
            with suppress(Exception):
                timer.stop()
            with suppress(Exception):
                btn_play.label.set_text('▶ Play')

    # Collega i pulsanti
    btn_prev.on_clicked(lambda _event: _navigate(-1))
    btn_play.on_clicked(on_play)
    btn_next.on_clicked(lambda _event: _navigate(+1))


    # Disegna subito la prima traiettoria
    draw_current()
    plt.show()


def save_trajectories_images(histories, titles, show_orient_every=20, *, environment: Optional[Union[Environment, Sequence[Optional[Environment]]]] = None, fit_to: str = 'trajectory', error_messages: Optional[Sequence[Optional[str]]] = None):
    """Salva PNG per ciascuna traiettoria, con simboli del robot completi.

    Nota: non mostra messaggi di errore sulle immagini statiche.
    """
    assert len(histories) == len(titles) and len(histories) > 0, "Liste vuote o di diversa lunghezza"
    if isinstance(show_orient_every, (list, tuple, np.ndarray)):
        assert len(show_orient_every) == len(histories), "show_orient_every deve avere stessa lunghezza di delle traiettorie"

    def _resolve_show_every(idx: int) -> int:
        """Ritorna lo step da usare per la traiettoria idx (singolo valore o per-traiettoria)."""
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

    for i, (hist, title_str) in enumerate(zip(histories, titles)):
        # Figura temporanea solo per il salvataggio
        fig, ax = plt.subplots(figsize=(7, 7))
        step = _resolve_show_every(i)
        env_cur = _resolve_env(i)
        _plot_static_trajectory_on_axes(ax, hist, step=step, title=None, include_title=False, include_axis_labels=False, draw_glyphs=True, environment=env_cur, fit_to=fit_to)
        out_path = _default_save_path(title_str)
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        print(f"Figura salvata in: {out_path}")
        plt.close(fig)
