# Classe che ha il compito di plottare le traiettorie, disegnare il robot in alcuni istanti e salvare figure

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from pathlib import Path  # Per costruire percorsi portabili
import re
from datetime import datetime
from matplotlib.widgets import Button  # Pulsanti UI per navigare tra i grafici


def draw_robot(ax, state, robot_radius=0.1, color='tab:blue'):
    """Disegna il robot come un cerchio con una freccia che indica l'orientamento.
    - ax: oggetto Axes su cui disegnare
    - state: array/tupla [x, y, theta] (posizione e orientamento del robot)
    - robot_radius: raggio del cerchio che rappresenta il robot (in metri nelle stesse unità dell'asse)
    - color: colore del robot (nome o codice matplotlib)
    """
    x, y, th = state  # Decomposizione dello stato nei tre componenti

    # Corpo del robot: cerchio pieno, leggermente trasparente, contorno nero
    circ = Circle((x, y), robot_radius, fill=True, alpha=0.3, color=color, ec='k')
    ax.add_patch(circ)  # Aggiunge il cerchio all'axes

    # Vettore direzione: piccolo offset in direzione theta per mostrare l'orientamento
    dir_len = 0.3  # lunghezza della freccia in unità dell'asse (scalare a piacere)
    dx = dir_len * np.cos(th)  # componente x della direzione
    dy = dir_len * np.sin(th)  # componente y della direzione

    # Freccia che indica l'orientamento; head_width/head_length controllano la punta
    ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.008, fc=color, ec=color)


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

    # Disegna il robot (cerchio + freccia) ogni show_orient_every campioni
    for i in range(0, n, max(1, int(show_orient_every))):
        draw_robot(ax, history[i])

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


def show_trajectories_carousel(histories, titles, show_orient_every=20, save_each=False):
    """Mostra più traiettorie in un'unica finestra con pulsanti per scorrere.
    - histories: lista di array (N_i x 3) con [x, y, theta]
    - titles: lista di stringhe titoli, stessa lunghezza di histories
    - show_orient_every: ogni quante pose disegnare il robot
    - save_each: se True, salva un PNG in Tesi/img per ogni traiettoria mostrata
    """
    assert len(histories) == len(titles) and len(histories) > 0, "Liste vuote o di diversa lunghezza"

    # Figura e assi principali
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.18)  # Lascia spazio ai pulsanti

    # Stato interno dell'indice corrente
    state = {"idx": 0}

    def draw_current():
        """Disegna la traiettoria corrente (pulisce l'axes e ridisegna)."""
        ax.clear()  # Pulisce l'axes
        hist = histories[state["idx"]]
        title = titles[state["idx"]]
        n = len(hist)
        ax.plot(hist[:, 0], hist[:, 1], '-o', markevery=max(1, n // 20))  # Traiettoria (linea + marker radi)
        for i in range(0, n, max(1, int(show_orient_every))):  # Robot a intervalli regolari
            draw_robot(ax, hist[i])
        ax.set_aspect('equal', 'box')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)
        ax.set_title(title)
        fig.canvas.draw_idle()  # Aggiorna il rendering
        if save_each:
            out_path = _default_save_path(title)
            fig.savefig(out_path, dpi=120, bbox_inches='tight')
            print(f"Figura salvata in: {out_path}")

    # Pulsante PRECEDENTE
    ax_prev = fig.add_axes([0.20, 0.05, 0.20, 0.08])  # [left, bottom, width, height] in coord figure
    btn_prev = Button(ax_prev, '⟨ Precedente')

    # Pulsante SUCCESSIVO
    ax_next = fig.add_axes([0.60, 0.05, 0.20, 0.08])
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
    - show_orient_every: ogni quante pose disegnare il robot nella figura salvata

    Le immagini vengono salvate con nomi derivati dal titolo + timestamp.
    Non viene aperta alcuna finestra per questi salvataggi (le figure sono create e chiuse subito)."""
    assert len(histories) == len(titles) and len(histories) > 0, "Liste vuote o di diversa lunghezza"
    for hist, title in zip(histories, titles):
        # Crea figura temporanea per il solo salvataggio
        fig, ax = plt.subplots(figsize=(7, 7))
        n = len(hist)
        ax.plot(hist[:, 0], hist[:, 1], '-o', markevery=max(1, n // 20))  # Traiettoria (linea + alcuni marker)
        for i in range(0, n, max(1, int(show_orient_every))):  # Pose sparse per orientamento
            draw_robot(ax, hist[i])
        ax.set_aspect('equal', 'box')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)
        ax.set_title(title)
        out_path = _default_save_path(title)  # Percorso nella cartella img
        fig.savefig(out_path, dpi=120, bbox_inches='tight')  # Salvataggio su file
        print(f"Figura salvata in: {out_path}")
        plt.close(fig)  # Chiude la figura per non aprire finestre multiple
