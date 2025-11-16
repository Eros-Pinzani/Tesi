"""
Classe che orchestra la simulazione del robot mobile.
Gestisce l'avanzamento temporale, l'applicazione dei comandi e la registrazione della storia.
"""

import numpy as np
from robot import Robot


class Simulator:
    def __init__(self, robot=None):
        """
        Inizializza il simulatore con un robot.

        Args:
            robot: Istanza di Robot da simulare (ne crea una nuova se None)
        """
        self.robot = robot or Robot()
        self.history = None  # Storia degli stati [x, y, theta] del robot
        self.commands = None  # Sequenza di comandi [v, omega] applicati

    def run_from_sequence(self, vs, omegas, dt):
        """
        Esegue la simulazione applicando una sequenza di comandi di velocità.

        Args:
            vs: Array di velocità lineari (N elementi)
            omegas: Array di velocità angolari (N elementi)
            dt: Intervallo di tempo tra i comandi

        Returns:
            history: Array (N+1, 3) con gli stati del robot, includendo lo stato iniziale
        """
        n = len(vs)  # Numero di passi temporali da simulare

        # Alloca gli array per salvare storia e comandi
        self.history = np.zeros((n+1, 3))  # N+1 stati (include stato iniziale)
        self.commands = np.zeros((n, 2))  # N comandi applicati

        # Salva lo stato iniziale del robot
        self.history[0] = self.robot.state()

        # Esegue la simulazione passo per passo
        for k in range(n):
            # Imposta il comando corrente
            self.robot.set_command(vs[k], omegas[k])

            # Avanza la dinamica del robot di un passo temporale dt
            self.robot.step(dt)

            # Registra il nuovo stato dopo l'avanzamento
            self.history[k+1] = self.robot.state()

            # Salva il comando applicato
            self.commands[k] = [vs[k], omegas[k]]

        return self.history

    def reset_robot(self, x=0.0, y=0.0, theta=0.0):
        """
        Reimposta il robot a una nuova posa iniziale.

        Args:
            x: Nuova coordinata x
            y: Nuova coordinata y
            theta: Nuovo orientamento in radianti
        """
        # Crea un nuovo robot con la posa specificata
        self.robot = Robot(x=x, y=y, theta=theta)

        # Azzera storia e comandi per la nuova simulazione
        self.history = None
        self.commands = None
