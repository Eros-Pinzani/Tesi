# Classe che ha il compito di plottare le traiettorie, disegnare il robot in alcuni istanti e salvare figure

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from pathlib import Path  # Per costruire percorsi portabili
import re
from datetime import datetime
from matplotlib.widgets import Button  # Pulsanti UI per navigare tra i grafici


def draw_robot(ax, state, robot_radius=0.1, color='tab:blue', dir_len=None):
    """Disegna il robot come un cerchio con una freccia che indica l'orientamento.
    - ax: oggetto Axes su cui disegnare
    - state: array/tupla [x, y, theta] (posizione e orientamento del robot)
    - robot_radius: raggio del cerchio che rappresenta il robot (in metri nelle stesse unità dell'asse)
    - color: colore del robot (nome o codice matplotlib)
    - dir_len: lunghezza della freccia direzionale in unità dati (se None, 3×robot_radius)
    """
    x, y, th = state  # Decomposizione dello stato nei tre componenti

    # Corpo del robot: cerchio pieno, leggermente trasparente, contorno nero
    circ = Circle((x, y), robot_radius, fill=True, alpha=0.3, color=color, ec='k')
    ax.add_patch(circ)  # Aggiunge il cerchio all'axes

    # Vettore direzione: piccolo offset in direzione theta per mostrare l'orientamento
    if dir_len is None:
        dir_len = 3.0 * robot_radius  # Default: freccia ~3× raggio robot
    dx = dir_len * np.cos(th)  # componente x della direzione
    dy = dir_len * np.sin(th)  # componente y della direzione

    # Freccia che indica l'orientamento; head_width/head_length in proporzione a dir_len
    ax.arrow(
        x,
        y,
        dx,
        dy,
        head_width=0.3 * robot_radius,
        head_length=0.4 * robot_radius,
        fc=color,
        ec=color,
        length_includes_head=True,
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
    # Raggio ~3% dell'estensione; limite minimo per non scomparire su traiettorie molto piccole
    robot_radius = max(0.05, 0.03 * ref)
    # Freccia ~2.5× raggio
    dir_len = 2.5 * robot_radius
    return robot_radius, dir_len


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

    # Traccia la polilinea della traiettoria (x, y). markevery riduce i marker per traiettorie lunghe
    n = len(history)
    ax.plot(history[:, 0], history[:, 1], '-o', markevery=max(1, n // 20))

    # Calcola dimensione robot e freccia in modo adattivo rispetto alla traiettoria
    r_robot, d_arrow = _robot_scale_from_history(history)

    # Disegna il robot (cerchio + freccia) ogni show_orient_every campioni
    for i in range(0, n, max(1, int(show_orient_every))):
        draw_robot(ax, history[i], robot_radius=r_robot, dir_len=d_arrow)

    # Imposta limiti assi con margine per includere il robot anche su traiettorie piatte
    x_min, x_max = float(np.min(history[:, 0])), float(np.max(history[:, 0]))
    y_min, y_max = float(np.min(history[:, 1])), float(np.max(history[:, 1]))
    pad = max(1.2 * r_robot, 0.02 * max(x_max - x_min, y_max - y_min, 1.0))
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)

    # Imposta proporzioni uguali sugli assi per non deformare la geometria
    ax.set_aspect('equal', 'box')

    # Etichette degli assi con unità di misura
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # Griglia e titolo per leggibilità
    ax.grid(True)
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
        assert len(show_orient_every) == len(histories), "show_orient_every deve avere stessa lunghezza delle traiettorie"
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

    # Stato interno dell'indice corrente
    state = {"idx": 0, "show_info": bool(show_info)}
    info_artist = None  # handle del box info (fig.text), gestito come variabile di chiusura

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
        ax.clear()  # Pulisce l'axes
        hist = histories[state["idx"]]
        title = titles[state["idx"]]
        n = len(hist)
        ax.plot(hist[:, 0], hist[:, 1], '-o', markevery=max(1, n // 20))  # Traiettoria (linea + marker radi)
        # Dimensioni adattive per il robot sulla traiettoria corrente
        r_robot, d_arrow = _robot_scale_from_history(hist)
        step = _resolve_show_every(state["idx"])  # passo specifico per traiettoria
        for i in range(0, n, step):  # Robot a intervalli regolari
            draw_robot(ax, hist[i], robot_radius=r_robot, dir_len=d_arrow)
        # Limiti con margine per includere il robot
        x_min, x_max = float(np.min(hist[:, 0])), float(np.max(hist[:, 0]))
        y_min, y_max = float(np.min(hist[:, 1])), float(np.max(hist[:, 1]))
        pad = max(1.2 * r_robot, 0.02 * max(x_max - x_min, y_max - y_min, 1.0))
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)
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
            if info_artist is None:
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
                info_artist.set_text(info_text)
                info_artist.set_visible(True)
        else:
            # Nasconde il box se presente
            if info_artist is not None:
                info_artist.set_visible(False)
        fig.canvas.draw_idle()  # Aggiorna il rendering
        if save_each:
            out_path = _default_save_path(title)
            fig.savefig(out_path, dpi=120, bbox_inches='tight')
            print(f"Figura salvata in: {out_path}")

    # Pulsante PRECEDENTE
    ax_prev = fig.add_axes((0.20, 0.05, 0.20, 0.08))  # [left, bottom, width, height] in coord figure
    btn_prev = Button(ax_prev, '⟨ Precedente')

    # Pulsante SUCCESSIVO
    ax_next = fig.add_axes((0.60, 0.05, 0.20, 0.08))
    btn_next = Button(ax_next, 'Successivo ⟩')

    def on_prev(event):
        state["idx"] = (state["idx"] - 1) % len(histories)  # Indice circolare
        draw_current()

    def on_next(event):
        state["idx"] = (state["idx"] + 1) % len(histories)
        draw_current()

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    # Scorciatoie da tastiera: sinistra/destra per navigare, 'q' per chiudere
    def on_key(event):
        if event.key in ('left', 'a'):
            on_prev(event)
        elif event.key in ('right', 'd'):
            on_next(event)
        elif event.key in ('q', 'escape'):
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
        assert len(show_orient_every) == len(histories), "show_orient_every deve avere stessa lunghezza delle traiettorie"

    def _resolve_show_every(idx: int) -> int:
        if isinstance(show_orient_every, (list, tuple, np.ndarray)):
            return max(1, int(show_orient_every[idx]))
        return max(1, int(show_orient_every))

    for idx, (hist, title) in enumerate(zip(histories, titles)):
        # Crea figura temporanea per il solo salvataggio
        fig, ax = plt.subplots(figsize=(7, 7))
        n = len(hist)
        ax.plot(hist[:, 0], hist[:, 1], '-o', markevery=max(1, n // 20))  # Traiettoria (linea + alcuni marker)
        # Dimensioni adattive per il robot anche nei salvataggi batch
        r_robot, d_arrow = _robot_scale_from_history(hist)
        step = _resolve_show_every(idx)
        for i in range(0, n, step):  # Pose sparse per orientamento
            draw_robot(ax, hist[i], robot_radius=r_robot, dir_len=d_arrow)
        # Limiti con margine per includere il robot
        x_min, x_max = float(np.min(hist[:, 0])), float(np.max(hist[:, 0]))
        y_min, y_max = float(np.min(hist[:, 1])), float(np.max(hist[:, 1]))
        pad = max(1.2 * r_robot, 0.02 * max(x_max - x_min, y_max - y_min, 1.0))
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect('equal', 'box')
        out_path = _default_save_path(title)  # Percorso nella cartella img
        fig.savefig(out_path, dpi=120, bbox_inches='tight')  # Salvataggio su file
        print(f"Figura salvata in: {out_path}")
        plt.close(fig)  # Chiude la figura per non aprire finestre multiple
