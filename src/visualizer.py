# Classe che ha il compito di plottare le traiettorie, disegnare il robot in alcuni istanti e salvare figure

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import numpy as np
from pathlib import Path  # Per costruire percorsi portabili
import re
from datetime import datetime
from matplotlib.widgets import Button  # Pulsanti UI per navigare tra i grafici
from matplotlib import transforms as mtransforms
from typing import Optional
from matplotlib.text import Text
from contextlib import suppress


# Helper per rimuovere in sicurezza un artista matplotlib (gestisce None ed eccezioni)
def _safe_remove_artist(artist: Optional[object]):
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
    verso l'esterno rispetto al vertice del corpo (in coordinate locali)."""
    wheel_radius = 0.22 * robot_radius
    offset_out = 0.15 * robot_radius
    return wheel_radius, offset_out


def draw_robot(ax, state, robot_radius=0.1, color='tab:blue', dir_len=None, arrow_color='orange', center_color='orange',
               draw_wheels: bool = True, wheel_facecolor='white', wheel_edgecolor='k'):
    """Disegna il robot come un rettangolo orientato (lato corto = fronte),
    con pallino centrale e freccia arancioni e quattro rotelle posizionate
    in corrispondenza della fine dei lati lunghi (cioè davanti e dietro, allineate
    con le due fiancate), non sui vertici.
    - ax: oggetto Axes su cui disegnare
    - state: array/tupla [x, y, theta]
    - robot_radius: parametro di scala
    - color: colore del corpo del robot (default blu)
    - dir_len: lunghezza freccia; default 3×raggio
    - arrow_color / center_color: colori per freccia e centro
    - draw_wheels: se True disegna quattro rotelle come cerchi ai vertici
    """
    x, y, th = state

    # Dimensioni rettangolo: lato lungo allineato con theta (direzione di marcia), fronte = lato corto
    width, length = _rect_dims_from_radius(robot_radius)

    # Rettangolo in frame locale con asse lungo allineato con x-locale
    rect = Rectangle((-length/2.0, -width/2.0), length, width, linewidth=1.0, facecolor=color, alpha=0.3, edgecolor='k', zorder=3)
    trans = mtransforms.Affine2D().rotate(th).translate(x, y) + ax.transData
    rect.set_transform(trans)
    ax.add_patch(rect)

    # Rotelle: quattro cerchi allineati con la fine dei lati lunghi (sporgenza solo lungo x)
    if draw_wheels:
        w_r, w_off = _wheel_params(robot_radius)
        # Fattore di posizionamento lungo l'asse lungo: 1.0 sarebbe sugli spigoli;
        # usiamo <1 per portarli vicino alla fine dei lati lunghi ma non agli spigoli.
        wheel_long_frac = 0.8  # 80% della semi-lunghezza
        x_off = wheel_long_frac * (length / 2.0)
        corners = [
            ( +x_off, +width/2.0 + w_off),  # lato lungo superiore, vicino all'estremità destra
            ( -x_off, +width/2.0 + w_off),  # lato lungo superiore, vicino all'estremità sinistra
            ( +x_off, -width/2.0 - w_off),  # lato lungo inferiore, vicino all'estremità destra
            ( -x_off, -width/2.0 - w_off),  # lato lungo inferiore, vicino all'estremità sinistra
        ]
        for cx, cy in corners:
            wheel = Circle((cx, cy), w_r, facecolor=wheel_facecolor, edgecolor=wheel_edgecolor, linewidth=1.0, zorder=4)
            wheel.set_transform(trans)
            ax.add_patch(wheel)

    # Pallino centrale (arancione)
    center_r = 0.25 * robot_radius
    center = Circle((0.0, 0.0), center_r, fill=True, color=center_color, ec='none', zorder=4)
    center.set_transform(trans)
    ax.add_patch(center)

    # Freccia di orientamento (arancione), punta sul lato corto frontale (verso +x del frame locale dopo rotazione)
    if dir_len is None:
        dir_len = 3.0 * robot_radius
    dx = dir_len * np.cos(th)
    dy = dir_len * np.sin(th)
    ax.arrow(
        x,
        y,
        dx,
        dy,
        head_width=0.3 * robot_radius,
        head_length=0.4 * robot_radius,
        fc=arrow_color,
        ec=arrow_color,
        length_includes_head=True,
        zorder=4,
    )


def _default_save_path(title: str) -> Path:
    """Crea un percorso di default nella cartella Tesi/img a partire dal titolo.
    - La cartella img viene creata se non esiste.
    - Il nome file viene derivato dal titolo + timestamp per evitare sovrascritture.
    """
    # project_root = cartella padre di 'src'
    project_root = Path(__file__).resolve().parents[1]
    img_dir = project_root / 'img'
    img_dir.mkdir(parents=True, exist_ok=True)  # Crea la cartella se non esiste
    base = title.lower().strip() or 'traiettoria'
    base = re.sub(r'\s+', '_', base)
    base = re.sub(r'[^a-z0-9_\-]', '', base)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return img_dir / f"{base}_{stamp}.png"


def _robot_scale_from_history(history):
    """Calcola una scala adatta per disegnare il robot in funzione dell'estensione della traiettoria.
    Ritorna (robot_radius, dir_len).
    """
    x_range = float(np.ptp(history[:, 0]))  # max-min su x
    y_range = float(np.ptp(history[:, 1]))  # max-min su y
    # Scala di riferimento: estensione massima tra x e y, con un minimo per evitare raggio nullo
    ref = max(x_range, y_range, 1.0)
    # Raggio ~1.2% dell'estensione; minimo assoluto ridotto per figure compatte
    robot_radius = max(0.02, 0.012 * ref)
    # Freccia ~2.5× raggio
    dir_len = 2.5 * robot_radius
    return robot_radius, dir_len


def _compute_axes_limits_with_glyphs(history, step, r_robot, d_arrow):
    """Calcola limiti x/y includendo corpo del robot (rettangolo), rotelle e punte delle frecce.
    Usa l'angolo theta salvato nello stato, include l'estensione massima indipendente dall'orientamento
    pari al raggio della circonferenza circoscritta al corpo + offset rotelle."""
    xs = history[:, 0]
    ys = history[:, 1]
    # Estensione base della traiettoria
    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    # Margine legato al corpo (diagonale del rettangolo) + rotelle poste esternamente
    width, length = _rect_dims_from_radius(r_robot)
    w_r, w_off = _wheel_params(r_robot)
    body_half_diag = 0.5 * float(np.hypot(length, width))
    wheels_extra = float(w_off + w_r)
    extent_radius = body_half_diag + wheels_extra

    x_min -= extent_radius
    x_max += extent_radius
    y_min -= extent_radius
    y_max += extent_radius

    # Estensione per punte delle frecce (valutata a intervalli) + sempre ultima posa
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

    # Piccolo margine finale
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
):
    n = len(hist)
    step = max(1, int(step))
    # Linea della traiettoria (nera) come sfondo
    ax.plot(hist[:, 0], hist[:, 1], '-', linewidth=1.5, color='k', zorder=0)
    # Scala robot e freccia
    r_robot, d_arrow = _robot_scale_from_history(hist)
    # Pose sparse
    for i in range(0, n, step):
        if i == 0:
            body_col, arr_col, ctr_col = 'green', 'orange', 'green'
        elif i == n - 1:
            body_col, arr_col, ctr_col = 'red', 'orange', 'red'
        else:
            body_col, arr_col, ctr_col = 'tab:blue', 'orange', 'orange'
        draw_robot(ax, hist[i], robot_radius=r_robot, dir_len=d_arrow, color=body_col, arrow_color=arr_col, center_color=ctr_col)
    # Assicura l'ultima posa
    if n > 0 and ((n - 1) % step != 0 or n == 1):
        draw_robot(ax, hist[-1], robot_radius=r_robot, dir_len=d_arrow, color='red', arrow_color='orange', center_color='red')

    # Limiti che includono frecce/cerchi
    x0, x1, y0, y1 = _compute_axes_limits_with_glyphs(hist, step, r_robot, d_arrow)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Aspetto e opzioni
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
    """Crea la stringa info per il pannello.
    - hist: array (N x 3) [x, y, th]
    - k_pose: indice della posa corrente da mostrare
    - dt: passo temporale
    - commands: opzionale (M x 2) [v, w]
    - use_cmd_of_prev: se True usa comando k_pose-1, altrimenti k_pose (clip in range)
    - show_next_pose: se True mostra la posa k_pose+1 (clip a N-1), altrimenti k_pose
    """
    N = int(len(hist) if hist is not None else 0)
    dt = float(max(dt, 1e-9))
    k_pose = int(max(0, min(k_pose, max(N - 1, 0))))

    # v, w: da comandi (se disponibili) oppure stimati da history
    v_k: float
    w_k: float
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
            dth = (dth + np.pi) % (2 * np.pi) - np.pi
            w_k = dth / dt
        else:
            v_k = 0.0
            w_k = 0.0

    # Tempo e posa da visualizzare
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
        f"k={k_pose}  t={t_k:.2f} s\n"
        f"v={v_k:.2f} m/s,  ω={w_k:.2f} rad/s\n"
        f"x={x_k:.2f} m,  y={y_k:.2f} m,  ϑ={th_k:.2f} rad"
    )
    return info_text


def _update_info_artist(fig, info_artist: Optional[Text], info_text: str) -> Text:
    """Rimuove il box info precedente (se presente) e crea un nuovo fig.text standard.
    Ritorna il nuovo artista Text creato."""
    if info_artist is not None:
        _safe_remove_artist(info_artist)
    return fig.text(
        0.98,
        0.96,
        info_text,
        ha='right',
        va='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'),
    )


def plot_trajectory(history, show_orient_every=20, title="Traiettoria del robot", save_path=None):
    """Plotta la traiettoria del robot e disegna il robot a intervalli regolari.
    Se save_path è fornito salva la figura su file; altrimenti salva automaticamente in Tesi/img.
    - history: array di shape (N, 3) con colonne [x, y, theta] per ciascun istante
    - show_orient_every: ogni quante pose disegnare il robot (freccia di orientamento)
    - title: titolo della figura
    - save_path: percorso del file dove salvare l'immagine (opzionale)
    """
    # Crea una figura quadrata per non distorcere la forma della traiettoria
    fig, ax = plt.subplots(figsize=(7, 7))  # fig è l'oggetto figura, ax gli assi su cui si disegna

    # Usa helper centralizzato per disegnare
    step = max(1, int(show_orient_every))
    _plot_static_trajectory_on_axes(ax, history, step=step, title=title, include_title=True, include_axis_labels=True)

    # Determina il percorso di salvataggio
    out_path = Path(save_path) if save_path else _default_save_path(title)
    # Salva la figura su file
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Figura salvata in: {out_path}")  # Messaggio informativo sul percorso di salvataggio

    # Mostra la finestra grafica (nel backend non interattivo 'Agg' non apre una finestra)
    plt.show()


def show_trajectories_carousel(
    histories,
    titles,
    show_orient_every=20,
    save_each=False,
    commands_list=None,
    dts=None,
    show_info=False,
    show_legend=True,
):
    """Mostra più traiettorie in un'unica finestra con pulsanti per scorrere.
    - histories: lista di array (N_i x 3) con [x, y, theta]
    - titles: lista di stringhe titoli, stessa lunghezza di histories
    - show_orient_every: intero unico oppure lista/tupla/ndarray di interi (uno per traiettoria)
    - save_each: se True, salva un PNG in Tesi/img per ogni traiettoria mostrata
    - commands_list: lista di array (N_i x 2) con [v, omega] per step; opzionale (per info panel)
    - dts: float unico o lista di float (uno per traiettoria) per convertire k -> tempo; opzionale
    - show_info: se True, il pannello info è attivo (in alto a destra, fuori dal grafico)
    - show_legend: se True, mostra una legenda in alto a sinistra con la spiegazione dei simboli
    """
    assert len(histories) == len(titles) and len(histories) > 0, "Liste vuote o di diversa lunghezza"
    # Se è una sequenza, deve avere la stessa lunghezza di histories
    if isinstance(show_orient_every, (list, tuple, np.ndarray)):
        assert len(show_orient_every) == len(histories), "show_orient_every deve avere stessa lunghezza di delle traiettorie"
    if commands_list is not None:
        assert len(commands_list) == len(histories), "commands_list deve avere stessa lunghezza di histories"
    # Normalizza dts a lista
    if dts is None:
        dts_resolved = [1.0] * len(histories)  # fallback unitario se non fornito (solo per t indicativo)
    elif isinstance(dts, (list, tuple, np.ndarray)):
        assert len(dts) == len(histories), "dts deve avere stessa lunghezza di histories"
        dts_resolved = [float(x) for x in dts]
    else:
        dts_resolved = [float(dts)] * len(histories)

    def _resolve_show_every(idx: int) -> int:
        if isinstance(show_orient_every, (list, tuple, np.ndarray)):
            return max(1, int(show_orient_every[idx]))
        return max(1, int(show_orient_every))

    # Figura e assi principali
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.18)  # Lascia spazio ai pulsanti

    # Stato interno dell'indice corrente + player
    state = {"idx": 0, "show_info": bool(show_info), "playing": False, "frame": 0}
    info_artist: Optional[Text] = None  # handle del box info (fig.text)
    moving_artists = []  # lista di patch dell'istanza mobile da rimuovere/aggiornare

    # Gestione icone vettoriali nei pulsanti
    icon_prev_artists = []
    icon_play_artists = []
    icon_next_artists = []

    def _clear_artists(lst):
        if not lst:
            return
        for art in lst:
            with suppress(Exception):
                art.remove()
        lst.clear()

    def _draw_prev_icon(ax_btn):
        nonlocal icon_prev_artists
        _clear_artists(icon_prev_artists)
        # Doppia freccia a sinistra (due triangoli) nella metà sinistra del bottone
        tri1 = Polygon([[0.34, 0.30], [0.34, 0.70], [0.22, 0.50]], closed=True, transform=ax_btn.transAxes,
                       facecolor='black', edgecolor='black', linewidth=1.0)
        tri2 = Polygon([[0.46, 0.30], [0.46, 0.70], [0.34, 0.50]], closed=True, transform=ax_btn.transAxes,
                       facecolor='black', edgecolor='black', linewidth=1.0)
        icon_prev_artists.append(ax_btn.add_patch(tri1))
        icon_prev_artists.append(ax_btn.add_patch(tri2))

    def _draw_next_icon(ax_btn):
        nonlocal icon_next_artists
        _clear_artists(icon_next_artists)
        # Doppia freccia a destra (due triangoli) nella metà destra del bottone
        tri1 = Polygon([[0.60, 0.30], [0.60, 0.70], [0.72, 0.50]], closed=True, transform=ax_btn.transAxes,
                       facecolor='black', edgecolor='black', linewidth=1.0)
        tri2 = Polygon([[0.72, 0.30], [0.72, 0.70], [0.84, 0.50]], closed=True, transform=ax_btn.transAxes,
                       facecolor='black', edgecolor='black', linewidth=1.0)
        icon_next_artists.append(ax_btn.add_patch(tri1))
        icon_next_artists.append(ax_btn.add_patch(tri2))

    def _set_play_icon(ax_btn, playing: bool):
        nonlocal icon_play_artists
        _clear_artists(icon_play_artists)
        if playing:
            # Pausa: due barre verticali nella metà sinistra
            r1 = Rectangle((0.36, 0.30), 0.08, 0.40, transform=ax_btn.transAxes,
                           facecolor='black', edgecolor='black', linewidth=1.0)
            r2 = Rectangle((0.50, 0.30), 0.08, 0.40, transform=ax_btn.transAxes,
                           facecolor='black', edgecolor='black', linewidth=1.0)
            icon_play_artists.append(ax_btn.add_patch(r1))
            icon_play_artists.append(ax_btn.add_patch(r2))
        else:
            # Play: triangolo punta a destra nella metà sinistra
            tri = Polygon([[0.34, 0.30], [0.34, 0.70], [0.58, 0.50]], closed=True, transform=ax_btn.transAxes,
                          facecolor='black', edgecolor='black', linewidth=1.0)
            icon_play_artists.append(ax_btn.add_patch(tri))

    # Timer per il player
    timer = fig.canvas.new_timer(interval=int(dts_resolved[0] * 1000))

    def _set_timer_interval_for_current():
        # Aggiorna intervallo in ms in base al dt della traiettoria corrente
        cur_dt = max(1e-6, float(dts_resolved[state["idx"]]))
        interval_ms = int(round(cur_dt * 1000))
        # Tenta entrambi i metodi senza sollevare: mantiene il comportamento (prima attributo, poi fallback)
        with suppress(Exception):
            timer.interval = interval_ms
        with suppress(Exception):
            if hasattr(timer, 'set_interval'):
                timer.set_interval(interval_ms)

    def _clear_moving():
        nonlocal moving_artists
        _clear_artists(moving_artists)

    def _draw_moving_at(k: int):
        """Disegna il robot mobile alla posa k, rimuovendo quello precedente."""
        nonlocal moving_artists
        _clear_moving()
        hist = histories[state["idx"]]
        k = int(max(0, min(k, len(hist) - 1)))
        # Cattura i patch creati da draw_robot per poterli rimuovere al prossimo frame
        before = len(ax.patches)
        r_robot, d_arrow = _robot_scale_from_history(hist)
        draw_robot(ax, hist[k], robot_radius=r_robot, dir_len=d_arrow, color='tab:blue', arrow_color='orange', center_color='orange')
        after = len(ax.patches)
        if after > before:
            moving_artists = ax.patches[before:after]

    # Box legenda statico (alto-sinistra, fuori dal grafico)
    if show_legend:
        legend_text = (
            "Legenda:\n"
            "k: indice campione\n"
            "t: tempo [s]\n"
            "v: velocità lineare [m/s]\n"
            "ω: velocità angolare [rad/s]\n"
            "x, y: posizione [m]\n"
            "θ: orientamento [rad]"
        )
        fig.text(
            0.02,
            0.96,
            legend_text,
            ha='left',
            va='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'),
        )

    def draw_current():
        """Disegna la traiettoria corrente (pulisce l'axes e ridisegna)."""
        nonlocal info_artist
        # Stoppa il player mentre si ridisegna/si cambia traiettoria
        state["playing"] = False
        with suppress(Exception):
            timer.stop()
        _clear_moving()

        ax.clear()  # Pulisce l'axes
        hist = histories[state["idx"]]
        title = titles[state["idx"]]
        n = len(hist)
        step = _resolve_show_every(state["idx"])  # passo specifico per traiettoria

        # Disegno statico tramite helper centralizzato
        _plot_static_trajectory_on_axes(ax, hist, step=step, title=title, include_title=True, include_axis_labels=True)

        # Pannello informazioni opzionale (controllato dal parametro)
        if state["show_info"]:
            idx = state["idx"]
            dt_cur = dts_resolved[idx]
            # Seleziona un indice "frame" rappresentativo: ultimo disegnato dalla griglia
            last_draw_idx = ((n - 1) // step) * step  # in [0, n-1]
            cmds = commands_list[idx] if commands_list is not None else None
            info_text = _build_info_text(
                hist,
                k_pose=int(last_draw_idx),
                dt=float(dt_cur),
                commands=cmds,
                use_cmd_of_prev=False,
                show_next_pose=True,
            )
            info_artist = _update_info_artist(fig, info_artist, info_text)
        else:
            # Rimuove il box se presente
            _safe_remove_artist(info_artist)
            info_artist = None

        # Inizializza frame corrente e robot mobile all'inizio
        state["frame"] = 0
        _draw_moving_at(0)
        _set_timer_interval_for_current()
        fig.canvas.draw_idle()  # Aggiorna il rendering
        if save_each:
            out_path = _default_save_path(title)
            fig.savefig(out_path, dpi=120, bbox_inches='tight')
            print(f"Figura salvata in: {out_path}")

    # Callback timer: avanza un frame secondo dt
    def _on_timer():
        nonlocal info_artist
        idx = state["idx"]
        hist = histories[idx]
        n = len(hist)
        k = state["frame"] + 1
        if k >= n:
            # Stoppa a fine traiettoria
            state["playing"] = False
            with suppress(Exception):
                timer.stop()
            # Aggiorna icona e testo del pulsante play (stato fermo)
            with suppress(Exception):
                _set_play_icon(ax_play, playing=False)
                btn_play.label.set_text('Play')
            return
        state["frame"] = k
        _draw_moving_at(k)
        # Aggiorna pannello info in tempo reale se attivo
        if state["show_info"]:
            with suppress(Exception):
                dt_cur = float(dts_resolved[idx])
                cmds = commands_list[idx] if commands_list is not None else None
                info_text = _build_info_text(
                    hist,
                    k_pose=int(k),
                    dt=dt_cur,
                    commands=cmds,
                    use_cmd_of_prev=True,
                    show_next_pose=False,
                )
                info_artist = _update_info_artist(fig, info_artist, info_text)
        fig.canvas.draw_idle()

    timer.add_callback(_on_timer)

    # Pulsanti: PRECEDENTE, PLAY/PAUSA, SUCCESSIVO
    ax_prev = fig.add_axes((0.16, 0.05, 0.22, 0.08))
    btn_prev = Button(ax_prev, 'Precedente')  # label + icona

    ax_play = fig.add_axes((0.40, 0.05, 0.20, 0.08))
    btn_play = Button(ax_play, 'Play')  # label + icona

    ax_next = fig.add_axes((0.64, 0.05, 0.24, 0.08))
    btn_next = Button(ax_next, 'Successivo')  # label + icona

    # Funzione di layout UNIFORME: stessa dimensione font per tutte le etichette
    def _layout_all_button_labels_uniform():
        with suppress(Exception):
            fig.canvas.draw()
        # Recupero renderer compatibile
        renderer = None
        get_r = getattr(fig.canvas, 'get_renderer', None)
        with suppress(Exception):
            renderer = get_r() if callable(get_r) else getattr(fig.canvas, 'renderer', None)
        if renderer is None:
            return
        # Costanti di layout per icone e margini
        right_pad = 0.04
        left_pad = 0.04
        # Gap minimi per pulsante (riduciamo quello di Play per avvicinare il testo all'icona)
        min_gap_prev = 0.02
        min_gap_play = 0.010
        min_gap_next = 0.02
        prev_icon_right_x = 0.50
        play_icon_right_x = 0.58
        next_icon_left_x = 0.60
        # Ripristina testo pieno e allineamenti/ancore iniziali
        btn_labels = [(btn_prev.label, 'prev'), (btn_play.label, 'play'), (btn_next.label, 'next')]
        for lab, kind in btn_labels:
            cur_txt = lab.get_text()
            full_txt = getattr(lab, '_full_text', None)
            if full_txt is None or (cur_txt != full_txt and not cur_txt.endswith('…')):
                setattr(lab, '_full_text', cur_txt)
                full_txt = cur_txt
            lab.set_text(full_txt or cur_txt)
            if kind in ('prev', 'play'):
                lab.set_horizontalalignment('right')
                lab.set_verticalalignment('center')
                lab.set_position((1.0 - right_pad, 0.50))
            else:  # next
                lab.set_horizontalalignment('left')
                lab.set_verticalalignment('center')
                lab.set_position((left_pad, 0.50))
        with suppress(Exception):
            fig.canvas.draw()

        def _avail_for_kind(kind_name: str) -> float:
            if kind_name == 'prev':
                anchor_x = 1.0 - right_pad
                return max(0.0, anchor_x - (prev_icon_right_x + min_gap_prev))
            elif kind_name == 'play':
                anchor_x = 1.0 - right_pad
                return max(0.0, anchor_x - (play_icon_right_x + min_gap_play))
            else:  # next
                anchor_x = left_pad
                return max(0.0, (next_icon_left_x - min_gap_next) - anchor_x)

        # Ricerca della massima dimensione font uniforme che entra per tutti
        max_fs = 12.0
        min_fs = 7.0
        fs_step = 0.5
        def fits_all(test_fs: float) -> bool:
            for lab0, kind0 in btn_labels:
                lab0.set_fontsize(test_fs)
                with suppress(Exception):
                    fig.canvas.draw()
                ax_bbox_i = lab0.axes.get_window_extent(renderer=renderer)
                txt_bbox_i = lab0.get_window_extent(renderer=renderer)
                text_w_axes_i = txt_bbox_i.width / max(ax_bbox_i.width, 1.0)
                avail_i = _avail_for_kind(kind0)
                if text_w_axes_i > avail_i + 1e-6:
                    return False
            return True
        # Trova il font size più grande che soddisfa tutti
        chosen = None
        fs = max_fs
        while fs >= min_fs:
            if fits_all(fs):
                chosen = fs
                break
            fs -= fs_step
        if chosen is None:
            chosen = min_fs
        # Applica font uniforme
        for lab, _ in btn_labels:
            lab.set_fontsize(chosen)
        with suppress(Exception):
            fig.canvas.draw()
        # Troncamento con ellissi se ancora serve, mantenendo font uniforme
        for lab2, kind2 in btn_labels:
            ax_bbox = lab2.axes.get_window_extent(renderer=renderer)
            txt_bbox = lab2.get_window_extent(renderer=renderer)
            text_w_axes = txt_bbox.width / max(ax_bbox.width, 1.0)
            avail = _avail_for_kind(kind2)
            if text_w_axes > avail + 1e-6:
                base = getattr(lab2, '_full_text', lab2.get_text()) or ''
                if not base:
                    continue
                trunc = base
                # Tronca finché non rientra o restano 3 char
                while text_w_axes > avail + 1e-6 and len(trunc) > 3:
                    trunc = trunc[:-1]
                    lab2.set_text(trunc.rstrip() + '…')
                    with suppress(Exception):
                        fig.canvas.draw()
                    txt_bbox = lab2.get_window_extent(renderer=renderer)
                    text_w_axes = txt_bbox.width / max(ax_bbox.width, 1.0)
        # Fine layout uniforme

    # Disegna icone iniziali
    _draw_prev_icon(ax_prev)
    _set_play_icon(ax_play, playing=False)
    _draw_next_icon(ax_next)

    # Layout iniziale uniforme
    with suppress(Exception):
        _layout_all_button_labels_uniform()

    # Rilayout su resize della finestra
    def _on_resize(_event):
        _layout_all_button_labels_uniform()
    with suppress(Exception):
        fig.canvas.mpl_connect('resize_event', _on_resize)

    # Navigazione generica tra traiettorie (delta = -1 per precedente, +1 per successiva)
    def _navigate(delta: int):
        state["idx"] = (state["idx"] + int(delta)) % len(histories)
        state["playing"] = False
        with suppress(Exception):
            timer.stop()
        _set_play_icon(ax_play, playing=False)
        with suppress(Exception):
            btn_play.label.set_text('Play')
        draw_current()
        with suppress(Exception):
            _layout_all_button_labels_uniform()

    def on_play(_event):
        # Toggle Play/Pausa
        if not state["playing"]:
            state["playing"] = True
            _set_timer_interval_for_current()
            with suppress(Exception):
                timer.start()
            _set_play_icon(ax_play, playing=True)
            with suppress(Exception):
                btn_play.label.set_text('Pausa')
        else:
            state["playing"] = False
            with suppress(Exception):
                timer.stop()
            _set_play_icon(ax_play, playing=False)
            with suppress(Exception):
                btn_play.label.set_text('Play')
        with suppress(Exception):
            _layout_all_button_labels_uniform()

    # Collega i pulsanti direttamente alla navigazione
    btn_prev.on_clicked(lambda _event: _navigate(-1))
    btn_play.on_clicked(on_play)
    btn_next.on_clicked(lambda _event: _navigate(+1))

    # Scorciatoie da tastiera: sinistra/destra per navigare, spazio per play/pausa, 'q' per chiudere
    def on_key(event):
        key = getattr(event, 'key', None)
        if key in ('left', 'a'):
            _navigate(-1)
        elif key in ('right', 'd'):
            _navigate(+1)
        elif key in (' ', 'space'):
            on_play(event)
        elif key in ('q', 'escape'):
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Disegna la prima traiettoria all'apertura
    draw_current()

    # Mostra finestra con pulsanti
    plt.show()


def save_trajectories_images(histories, titles, show_orient_every=20):
    """Salva tutte le traiettorie passate in input nella cartella Tesi/img in un colpo solo.
    - histories: lista di array (N_i x 3) con [x, y, theta]
    - titles: lista di stringhe, stessa lunghezza di histories
    - show_orient_every: intero unico oppure lista/tupla/ndarray di interi (uno per traiettoria)

    Le immagini vengono salvate con nomi derivati dal titolo + timestamp.
    Non viene aperta alcuna finestra per questi salvataggi (le figure sono create e chiuse subito)."""
    assert len(histories) == len(titles) and len(histories) > 0, "Liste vuote o di diversa lunghezza"
    if isinstance(show_orient_every, (list, tuple, np.ndarray)):
        assert len(show_orient_every) == len(histories), "show_orient_every deve avere stessa lunghezza di delle traiettorie"

    def _resolve_show_every(idx: int) -> int:
        if isinstance(show_orient_every, (list, tuple, np.ndarray)):
            return max(1, int(show_orient_every[idx]))
        return max(1, int(show_orient_every))

    for i, (hist, title_str) in enumerate(zip(histories, titles)):
        # Crea figura temporanea per il solo salvataggio
        fig, ax = plt.subplots(figsize=(7, 7))
        step = _resolve_show_every(i)
        # Disegno statico tramite helper centralizzato (senza etichette/titolo per immagini pulite)
        _plot_static_trajectory_on_axes(ax, hist, step=step, title=None, include_title=False, include_axis_labels=False)
        out_path = _default_save_path(title_str)  # Percorso nella cartella img
        fig.savefig(out_path, dpi=120, bbox_inches='tight')  # Salvataggio su file
        print(f"Figura salvata in: {out_path}")
        plt.close(fig)  # Chiude la figura per non aprire finestre multiple
