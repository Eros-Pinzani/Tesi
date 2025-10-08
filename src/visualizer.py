# Classe che ha il compito di plottare le traiettorie, disegnare il robot in alcuni istanti e salvare figure

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


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


def plot_trajectory(history, show_orient_every=20, title="Traiettoria del robot"):
    """Plotta la traiettoria del robot e disegna il robot a intervalli regolari.
    - history: array di shape (N, 3) con colonne [x, y, theta] per ciascun istante
    - show_orient_every: ogni quante pose disegnare il robot (freccia di orientamento)
    - title: titolo della figura
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

    # Mostra la finestra grafica (nel backend non interattivo 'Agg' non apre una finestra)
    plt.show()
