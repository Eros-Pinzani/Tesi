# Classe che fornisce traiettorie prefissate (circle, eight, line) e casuali per il robot

import numpy as np

class TrajectoryGenerator:
    def straight(self, v, T, dt):
        """Genera una traiettoria lineare con velocità v per un tempo T con passo dt
        Ritorna due array: (velocità lineari, velocità angolari)
        - v: velocità lineare costante
        - omega: sempre 0 (moto rettilineo)
        """
        n = int(np.ceil(T/dt))  # Numero di step discreti (arrotondato per eccesso)
        return np.full(n, v), np.zeros(n)  # v costante per tutti gli step, omega=0 ⇒ linea retta

    def straight_var_speed(self, v_min, v_max, T, dt, phase=0.0):
        """Moto rettilineo con velocità lineare variabile (profilo sinusoidale tra v_min e v_max).
        - v(t) = v_mid + v_amp * sin(2π t / T + phase)
        - omega(t) = 0 ⇒ traiettoria in linea retta
        Note: supponiamo v_min ≤ v_max e velocità non negative (tipico per uniciclo)."""
        n = int(np.ceil(T/dt))  # Numero di step discreti
        t = np.linspace(0, T, n)  # Asse temporale uniformemente campionato
        v_mid = 0.5 * (v_max + v_min)  # Valore medio del profilo di velocità
        v_amp = 0.5 * (v_max - v_min)  # Ampiezza dell'oscillazione
        vs = v_mid + v_amp * np.sin(2 * np.pi * t / T + phase)  # Profilo v(t) sinusoidale
        omegas = np.zeros(n)  # Nessuna rotazione: moto rettilineo
        return vs, omegas

    def circle(self, v, radius, T, dt):
        """Genera una traiettoria circolare con velocità v e raggio radius per un tempo T con passo dt
        - omega = v / R: velocità angolare costante (rad/s) per descrivere un cerchio di raggio "radius"
        """
        omega = v/float(radius)  # Relazione cinematica cerchio: omega = v / R
        n = int(np.ceil(T/dt))  # Numero di campioni temporali
        return np.full(n, v), np.full(n, omega)  # v e omega costanti ⇒ traiettoria circolare

    def circle_var_speed(self, v_min, v_max, radius, T, dt, phase=0.0):
        """Traiettoria circolare a raggio costante con velocità lineare variabile (sinusoidale).
        - v(t) sinusoidale tra v_min e v_max
        - omega(t) = v(t) / R, così il raggio rimane costante mentre varia la velocità (e la velocità angolare)"""
        n = int(np.ceil(T/dt))  # Numero di campioni
        t = np.linspace(0, T, n)  # Asse temporale
        v_mid = 0.5 * (v_max + v_min)  # Media del profilo
        v_amp = 0.5 * (v_max - v_min)  # Ampiezza del profilo
        vs = v_mid + v_amp * np.sin(2 * np.pi * t / T + phase)  # v(t) sinusoidale
        omegas = vs / float(radius)  # Mantiene il raggio costante imponendo omega(t) coerente
        return vs, omegas

    def eight(self, v, radius, T, dt):
        """Traiettoria "otto" molto semplice: prima metà curvatura positiva, seconda metà curvatura negativa.
        Non è una lemniscata esatta ma un concatenamento di due archi con stessa |omega|."""
        n = int(np.ceil(T / dt))              # Numero di step discreti totali in cui dividiamo la durata T
        mid = n // 2                          # Indice di separazione tra prima e seconda metà della traiettoria
        vs = np.full(n, v)                    # Velocità lineare costante v in tutti gli step
        omegas = np.zeros(n)                  # Pre-allocazione array velocità angolare
        omegas[:mid] = v / float(radius)      # Prima metà: curvatura costante (giro "a sinistra" se theta cresce)
        omegas[mid:] = -v / float(radius)     # Seconda metà: curvatura opposta (giro "a destra"), crea il cambio di lobo
        return vs, omegas                     # Ritorna i profili (v_k, omega_k) per ogni passo

    def random_walk(self, v_mean, omega_std, T, dt, seed=None):
        """Genera una traiettoria randomica con velocità media v_mean e deviazione standard omega_std per un tempo T con passo dt
        - v_mean: velocità lineare costante (media del moto)
        - omega ~ N(0, omega_std^2): rumore gaussiano per esplorazione angolare
        - seed: rende il risultato riproducibile se specificato
        """
        rng = np.random.default_rng(seed)  # Generatore di numeri casuali (riproducibile tramite seed)
        n = int(np.ceil(T/dt))  # Numero di step
        vs = np.full(n, v_mean)  # Velocità lineare costante per tutta la durata
        omegas = rng.normal(0.0, omega_std, size=n)  # Campiona omega i.i.d. da N(0, sigma^2)
        return vs, omegas  # Ritorna le sequenze (v(t), omega(t))
