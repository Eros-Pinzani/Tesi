# Classe che ha il compito di plottare le traiettorie, disegnare il robot in alcuni istanti e salvare figure

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from pathlib import Path  # Per costruire percorsi portabili
import re
from datetime import datetime
from matplotlib.widgets import Button  # Pulsanti UI per navigare tra i grafici
from matplotlib import transforms as mtransforms


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

    # Rettangolo in frame locale con asse lungo lungo x-locale
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
    # Sanitizzo il titolo per usarlo come nome file
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

    n = len(history)
    # Linea della traiettoria (nera) come sfondo
    ax.plot(history[:, 0], history[:, 1], '-', linewidth=1.5, color='k', zorder=0)
    step = max(1, int(show_orient_every))
    # (RIMOSSO) marker neri: lasciamo solo il pallino arancione del robot

    # Calcola dimensione robot e freccia in modo adattivo rispetto alla traiettoria
    r_robot, d_arrow = _robot_scale_from_history(history)

    # Disegna il robot usando direttamente [x, y, theta] dallo stato
    for i in range(0, n, step):
        # Colori: primo verde, ultimo rosso, altri blu con freccia arancione
        if i == 0:
            body_col, arr_col, ctr_col = 'green', 'orange', 'green'
        elif i == n - 1:
            body_col, arr_col, ctr_col = 'red', 'orange', 'red'
        else:
            body_col, arr_col, ctr_col = 'tab:blue', 'orange', 'orange'
        draw_robot(ax, history[i], robot_radius=r_robot, dir_len=d_arrow, color=body_col, arrow_color=arr_col, center_color=ctr_col)

    # Assicura che l'ultima posa sia sempre disegnata (anche se non cade sulla griglia)
    if n > 0 and ((n - 1) % step != 0 or n == 1):
        draw_robot(ax, history[-1], robot_radius=r_robot, dir_len=d_arrow, color='red', arrow_color='orange', center_color='red')

    # Limiti che includono anche le frecce e i cerchi
    x0, x1, y0, y1 = _compute_axes_limits_with_glyphs(history, step, r_robot, d_arrow)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Imposta proporzioni uguali sugli assi per non deformare la geometria
    ax.set_aspect('equal', 'box')

    # Etichette degli assi con unità di misura
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # Griglia disattivata e titolo
    ax.grid(False)
    ax.set_title(title)

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
    info_artist = None  # handle del box info (fig.text)
    moving_artists = []  # lista di patch dell'istanza mobile da rimuovere/aggiornare

    # Timer per il player
    timer = fig.canvas.new_timer(interval=int(dts_resolved[0] * 1000))

    def _set_timer_interval_for_current():
        # Aggiorna intervallo in ms in base al dt della traiettoria corrente
        cur_dt = max(1e-6, float(dts_resolved[state["idx"]]))
        try:
            timer.interval = int(round(cur_dt * 1000))
        except Exception:
            # Alcuni backend usano set_interval
            if hasattr(timer, 'set_interval'):
                timer.set_interval(int(round(cur_dt * 1000)))

    def _clear_moving():
        nonlocal moving_artists
        if moving_artists:
            for art in moving_artists:
                try:
                    art.remove()
                except Exception:
                    pass
            moving_artists = []

    def _draw_moving_at(k: int):
        """Disegna il robot mobile alla posa k, rimuovendo quello precedente."""
        nonlocal moving_artists
        _clear_moving()
        hist = histories[state["idx"]]
        k = int(max(0, min(k, len(hist) - 1)))
        # Cattura i patch creati da draw_robot per poterli rimuovere al prossimo frame
        before = len(ax.patches)
        draw_robot(ax, hist[k], robot_radius=_robot_scale_from_history(hist)[0], dir_len=_robot_scale_from_history(hist)[1], color='tab:blue', arrow_color='orange', center_color='orange')
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
        try:
            timer.stop()
        except Exception:
            pass
        _clear_moving()

        ax.clear()  # Pulisce l'axes
        hist = histories[state["idx"]]
        title = titles[state["idx"]]
        n = len(hist)
        # Linea nera (senza marker) come sfondo
        ax.plot(hist[:, 0], hist[:, 1], '-', linewidth=1.5, color='k', zorder=0)
        step = _resolve_show_every(state["idx"])  # passo specifico per traiettoria
        # Dimensioni adattive per il robot sulla traiettoria corrente
        r_robot, d_arrow = _robot_scale_from_history(hist)
        for i in range(0, n, step):  # Robot a intervalli regolari
            if i == 0:
                body_col, arr_col, ctr_col = 'green', 'orange', 'green'
            elif i == n - 1:
                body_col, arr_col, ctr_col = 'red', 'orange', 'red'
            else:
                body_col, arr_col, ctr_col = 'tab:blue', 'orange', 'orange'
            draw_robot(ax, hist[i], robot_radius=r_robot, dir_len=d_arrow, color=body_col, arrow_color=arr_col, center_color=ctr_col)
        # Assicura ultima posa sempre disegnata
        if n > 0 and ((n - 1) % step != 0 or n == 1):
            draw_robot(ax, hist[-1], robot_radius=r_robot, dir_len=d_arrow, color='red', arrow_color='orange', center_color='red')
        # Limiti che includono anche frecce e cerchi
        x0, x1, y0, y1 = _compute_axes_limits_with_glyphs(hist, step, r_robot, d_arrow)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(False)
        ax.set_title(title)

        # Pannello informazioni opzionale (controllato dal parametro)
        if state["show_info"]:
            idx = state["idx"]
            dt_cur = dts_resolved[idx]
            # Seleziona un indice "frame" rappresentativo: ultimo disegnato dalla griglia
            last_draw_idx = ((n - 1) // step) * step  # in [0, n-1]
            # Map in spazio comandi (che tipicamente ha lunghezza n-1)
            if commands_list is not None and commands_list[idx] is not None:
                cmd = commands_list[idx]
                k_cmd_max = len(cmd) - 1
                k = max(0, min(last_draw_idx, k_cmd_max))
                v_k, w_k = float(cmd[k][0]), float(cmd[k][1])
            else:
                # Stima da history (se possibile)
                k = max(1, min(last_draw_idx, n - 1))
                dx = float(hist[k][0] - hist[k - 1][0])
                dy = float(hist[k][1] - hist[k - 1][1])
                dth = float(hist[k][2] - hist[k - 1][2])
                v_k = (dx**2 + dy**2) ** 0.5 / max(dt_cur, 1e-9)
                # Normalizzo dth su [-pi,pi] per stima omega
                dth = (dth + np.pi) % (2 * np.pi) - np.pi
                w_k = dth / max(dt_cur, 1e-9)
            t_k = k * dt_cur
            # Stato da mostrare: pose successiva se disponibile (effetto del comando k)
            pose_idx = min(k + 1, n - 1)
            x_k, y_k, th_k = hist[pose_idx]
            info_text = (
                f"k={k}  t={t_k:.2f} s\n"
                f"v={v_k:.2f} m/s,  ω={w_k:.2f} rad/s\n"
                f"x={x_k:.2f} m,  y={y_k:.2f} m,  θ={th_k:.2f} rad"
            )
            # Ricrea sempre il box info (evita chiamate a set_text/set_visible)
            if info_artist is not None:
                try:
                    info_artist.remove()
                except Exception:
                    pass
            info_artist = fig.text(
                0.98,
                0.96,
                info_text,
                ha='right',
                va='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'),
            )
        else:
            # Rimuove il box se presente
            if info_artist is not None:
                try:
                    info_artist.remove()
                except Exception:
                    pass
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
            try:
                timer.stop()
            except Exception:
                pass
            # Aggiorna label del pulsante play
            try:
                btn_play.label.set_text('Play ⯈')
            except Exception:
                pass
            return
        state["frame"] = k
        _draw_moving_at(k)
        # Aggiorna pannello info in tempo reale se attivo
        if state["show_info"]:
            try:
                dt_cur = float(dts_resolved[idx])
                if commands_list is not None and commands_list[idx] is not None:
                    cmd = commands_list[idx]
                    k_cmd = max(0, min(k - 1, len(cmd) - 1))
                    v_k, w_k = float(cmd[k_cmd][0]), float(cmd[k_cmd][1])
                else:
                    # Stima da history tra k-1 e k
                    dx = float(hist[k][0] - hist[k - 1][0])
                    dy = float(hist[k][1] - hist[k - 1][1])
                    dth = float(hist[k][2] - hist[k - 1][2])
                    v_k = (dx**2 + dy**2) ** 0.5 / max(dt_cur, 1e-9)
                    dth = (dth + np.pi) % (2 * np.pi) - np.pi
                    w_k = dth / max(dt_cur, 1e-9)
                t_k = k * dt_cur
                x_k, y_k, th_k = hist[k]
                info_text = (
                    f"k={k}  t={t_k:.2f} s\n"
                    f"v={v_k:.2f} m/s,  ω={w_k:.2f} rad/s\n"
                    f"x={x_k:.2f} m,  y={y_k:.2f} m,  θ={th_k:.2f} rad"
                )
                # Ricrea sempre il box info (evita chiamate a set_text/set_visible)
                if info_artist is not None:
                    try:
                        info_artist.remove()
                    except Exception:
                        pass
                info_artist = fig.text(
                    0.98,
                    0.96,
                    info_text,
                    ha='right',
                    va='top',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.7'),
                )
            except Exception:
                pass
        fig.canvas.draw_idle()

    timer.add_callback(_on_timer)

    # Pulsanti: PRECEDENTE, PLAY/PAUSA, SUCCESSIVO
    ax_prev = fig.add_axes((0.16, 0.05, 0.20, 0.08))
    btn_prev = Button(ax_prev, '⟨ Precedente')

    ax_play = fig.add_axes((0.40, 0.05, 0.20, 0.08))
    btn_play = Button(ax_play, 'Play ⯈')

    ax_next = fig.add_axes((0.64, 0.05, 0.20, 0.08))
    btn_next = Button(ax_next, 'Successivo ⟩')

    def on_prev(event):
        # Cambia traiettoria indietro, resetta player
        state["idx"] = (state["idx"] - 1) % len(histories)
        state["playing"] = False
        try:
            timer.stop()
        except Exception:
            pass
        btn_play.label.set_text('Play ⯈')
        draw_current()

    def on_next(event):
        # Cambia traiettoria avanti, resetta player
        state["idx"] = (state["idx"] + 1) % len(histories)
        state["playing"] = False
        try:
            timer.stop()
        except Exception:
            pass
        btn_play.label.set_text('Play ⯈')
        draw_current()

    def on_play(event):
        # Toggle Play/Pausa
        if not state["playing"]:
            state["playing"] = True
            _set_timer_interval_for_current()
            try:
                timer.start()
            except Exception:
                pass
            btn_play.label.set_text('Pausa ⏸')
        else:
            state["playing"] = False
            try:
                timer.stop()
            except Exception:
                pass
            btn_play.label.set_text('Play ⯈')

    btn_prev.on_clicked(on_prev)
    btn_play.on_clicked(on_play)
    btn_next.on_clicked(on_next)

    # Scorciatoie da tastiera: sinistra/destra per navigare, spazio per play/pausa, 'q' per chiudere
    def on_key(event):
        key = getattr(event, 'key', None)
        if key in ('left', 'a'):
            on_prev(event)
        elif key in ('right', 'd'):
            on_next(event)
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

    for idx, (hist, title) in enumerate(zip(histories, titles)):
        # Crea figura temporanea per il solo salvataggio
        fig, ax = plt.subplots(figsize=(7, 7))
        n = len(hist)
        # Linea nera (senza marker) come sfondo
        ax.plot(hist[:, 0], hist[:, 1], '-', linewidth=1.5, color='k', zorder=0)
        step = _resolve_show_every(idx)
        # (RIMOSSO) marker neri
        # Dimensioni adattive per il robot anche nei salvataggi batch
        r_robot, d_arrow = _robot_scale_from_history(hist)
        for i in range(0, n, step):  # Pose sparse per orientamento
            if i == 0:
                body_col, arr_col, ctr_col = 'green', 'orange', 'green'
            elif i == n - 1:
                body_col, arr_col, ctr_col = 'red', 'orange', 'red'
            else:
                body_col, arr_col, ctr_col = 'tab:blue', 'orange', 'orange'
            draw_robot(ax, hist[i], robot_radius=r_robot, dir_len=d_arrow, color=body_col, arrow_color=arr_col, center_color=ctr_col)
        # Assicura ultima posa sempre disegnata
        if n > 0 and ((n - 1) % step != 0 or n == 1):
            draw_robot(ax, hist[-1], robot_radius=r_robot, dir_len=d_arrow, color='red', arrow_color='orange', center_color='red')
        # Limiti che includono anche frecce e cerchi
        x0, x1, y0, y1 = _compute_axes_limits_with_glyphs(hist, step, r_robot, d_arrow)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect('equal', 'box')
        ax.grid(False)
        out_path = _default_save_path(title)  # Percorso nella cartella img
        fig.savefig(out_path, dpi=120, bbox_inches='tight')  # Salvataggio su file
        print(f"Figura salvata in: {out_path}")
        plt.close(fig)  # Chiude la figura per non aprire finestre multiple
