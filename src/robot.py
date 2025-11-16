"""
Classe che rappresenta lo stato e la dinamica di un robot mobile differenziale.
Gestisce la posa (x, y, theta) e applica l'integrazione numerica discreta con schema di Eulero.
"""

import numpy as np

class Robot:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        """
        Inizializza il robot con una posa iniziale.

        Args:
            x: Coordinata x in metri
            y: Coordinata y in metri
            theta: Orientamento in radianti
        """
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
        self.v = 0.0  # Velocità lineare in m/s
        self.omega = 0.0  # Velocità angolare in rad/s

    def state(self):
        """Restituisce la posa corrente come array [x, y, theta]"""
        return np.array([self.x, self.y, self.theta])

    def set_command(self, v, omega):
        """
        Imposta i comandi di velocità per il prossimo step.

        Args:
            v: Velocità lineare in m/s
            omega: Velocità angolare in rad/s
        """
        self.v = float(v)
        self.omega = float(omega)

    def step(self, dt):
        """
        Aggiorna lo stato del robot applicando il metodo di Eulero esplicito:

        x_{k+1} = x_k + v_k * cos(theta_k) * dt
        y_{k+1} = y_k + v_k * sin(theta_k) * dt
        theta_{k+1} = theta_k + omega_k * dt

        Args:
            dt: Intervallo di tempo in secondi
        """
        # Calcola la nuova posizione in base alla velocità e orientamento correnti
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.y * np.sin(self.theta) * dt

        # Aggiorna l'orientamento
        self.theta += self.omega * dt

        # Normalizza l'angolo nell'intervallo [-π, π]
        self._normalize_angle()

    def _normalize_angle(self):
        """Normalizza l'angolo theta nell'intervallo [-π, π] per evitare overflow"""
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi