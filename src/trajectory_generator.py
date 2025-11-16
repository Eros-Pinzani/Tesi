"""
Generatore di traiettorie per robot mobile differenziale.
Fornisce traiettorie prefissate (linea retta, cerchio, figura a otto) e casuali.
"""

import numpy as np

class TrajectoryGenerator:
    @staticmethod
    def straight(v, T, dt):
        """
        Genera una traiettoria lineare a velocità costante.

        Args:
            v: Velocità lineare costante (m/s)
            T: Durata totale (secondi)
            dt: Intervallo di campionamento (secondi)

        Returns:
            vs: Array di velocità lineari costanti
            omegas: Array di velocità angolari nulle (moto rettilineo)
        """
        n = int(np.ceil(T/dt))
        return np.full(n, v), np.zeros(n)

    @staticmethod
    def straight_var_speed(v_min, v_max, T, dt, phase=0.0):
        """
        Moto rettilineo con profilo di velocità sinusoidale.

        La velocità varia secondo: v(t) = v_medio + v_ampiezza * sin(2πt/T + phase)

        Args:
            v_min: Velocità minima (m/s)
            v_max: Velocità massima (m/s)
            T: Durata totale (secondi)
            dt: Intervallo di campionamento (secondi)
            phase: Fase iniziale del seno (radianti)

        Returns:
            vs: Array di velocità lineari variabili
            omegas: Array di velocità angolari nulle
        """
        n = int(np.ceil(T/dt))
        t = np.linspace(0, T, n)

        # Calcola media e ampiezza del profilo sinusoidale
        v_mid = 0.5 * (v_max + v_min)
        v_amp = 0.5 * (v_max - v_min)

        # Genera il profilo di velocità
        vs = v_mid + v_amp * np.sin(2 * np.pi * t / T + phase)
        omegas = np.zeros(n)

        return vs, omegas

    @staticmethod
    def circle(v, radius, T, dt):
        """
        Genera una traiettoria circolare a velocità costante.

        Usa la relazione cinematica: omega = v / R

        Args:
            v: Velocità lineare costante (m/s)
            radius: Raggio del cerchio (metri)
            T: Durata totale (secondi)
            dt: Intervallo di campionamento (secondi)

        Returns:
            vs: Array di velocità lineari costanti
            omegas: Array di velocità angolari costanti
        """
        omega = v / float(radius)
        n = int(np.ceil(T/dt))
        return np.full(n, v), np.full(n, omega)

    @staticmethod
    def circle_var_speed(v_min, v_max, radius, T, dt, phase=0.0):
        """
        Traiettoria circolare a raggio costante con velocità variabile.

        La velocità lineare varia sinusoidalmente e omega si adatta per mantenere
        il raggio costante: omega(t) = v(t) / R

        Args:
            v_min: Velocità minima (m/s)
            v_max: Velocità massima (m/s)
            radius: Raggio del cerchio (metri)
            T: Durata totale (secondi)
            dt: Intervallo di campionamento (secondi)
            phase: Fase iniziale del seno (radianti)

        Returns:
            vs: Array di velocità lineari variabili
            omegas: Array di velocità angolari variabili (proporzionali a vs)
        """
        n = int(np.ceil(T/dt))
        t = np.linspace(0, T, n)

        # Profilo sinusoidale della velocità
        v_mid = 0.5 * (v_max + v_min)
        v_amp = 0.5 * (v_max - v_min)
        vs = v_mid + v_amp * np.sin(2 * np.pi * t / T + phase)

        # Adatta omega per mantenere il raggio costante
        omegas = vs / float(radius)

        return vs, omegas

    @staticmethod
    def eight(v, radius, T, dt):
        """
        Genera una traiettoria a forma di otto (∞) con transizione smooth.

        Primo lobo: rotazione oraria (omega positivo)
        Secondo lobo: rotazione antioraria (omega negativo)
        Transizione: interpolazione smooth con funzione coseno

        Args:
            v: Velocità lineare costante (m/s)
            radius: Raggio dei lobi (metri)
            T: Durata totale (secondi)
            dt: Intervallo di campionamento (secondi)

        Returns:
            vs: Array di velocità lineari costanti
            omegas: Array di velocità angolari con transizione smooth
        """
        n = int(np.ceil(T / dt))
        mid = n // 2  # Punto di transizione tra i due lobi

        vs = np.full(n, v)
        omegas = np.zeros(n)

        # Zona di transizione: 15% del tempo totale per garantire smoothness
        transition_width = max(3, int(0.15 * n))
        transition_start = mid - transition_width // 2
        transition_end = mid + transition_width // 2

        omega_val = v / float(radius)

        # Primo lobo: curvatura positiva
        omegas[:transition_start] = omega_val

        # Zona di transizione con interpolazione coseno (derivata continua)
        for i in range(transition_start, min(transition_end, n)):
            alpha = (i - transition_start) / float(transition_end - transition_start)
            # Interpolazione smooth: 0.5 * (1 - cos(π*alpha)) garantisce C1-continuity
            smooth_alpha = 0.5 * (1 - np.cos(np.pi * alpha))
            omegas[i] = omega_val * (1 - 2 * smooth_alpha)

        # Secondo lobo: curvatura negativa
        omegas[transition_end:] = -omega_val

        return vs, omegas

    @staticmethod
    def random_walk(v_mean, omega_std, T, dt, seed=None):
        """
        Genera una traiettoria casuale con velocità angolare stocastica.

        Velocità lineare costante, velocità angolare estratta da distribuzione gaussiana.
        Utile per simulare esplorazione o moto Browniano.

        Args:
            v_mean: Velocità lineare media (costante) (m/s)
            omega_std: Deviazione standard delle velocità angolari (rad/s)
            T: Durata totale (secondi)
            dt: Intervallo di campionamento (secondi)
            seed: Seed per riproducibilità (opzionale)

        Returns:
            vs: Array di velocità lineari costanti
            omegas: Array di velocità angolari stocastiche ~ N(0, omega_std²)
        """
        rng = np.random.default_rng(seed)
        n = int(np.ceil(T/dt))

        vs = np.full(n, v_mean)
        omegas = rng.normal(0.0, omega_std, size=n)

        return vs, omegas
