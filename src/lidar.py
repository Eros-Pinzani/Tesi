# Classe che ha il compito di simulare un sensore LiDAR, realizzato con Shapely e ha il compito di fare scansioni

import numpy as np
from shapely.geometry import LineString

class Lidar:
    def __init__(self, n_rays=360, angle_span=2*np.pi, r_max=6.0, angle_offset=0.0, add_noise=False, noise_std=0.01):
        """
        :param n_rays: numero di raggi per scansione
        :param angle_span: ampiezza angolare totale(rad)
        :param r_max: portata massima dei raggi
        :param angle_offset: orientamento iniziale relativo al robot (rad)
        :param add_noise: aggiunge rumore gaussiano alle distanze misurate
        :param noise_std: deviazione standard del rumore (metri)
        """
        self.n_rays = int(n_rays)
        self.angle_span = float(angle_span)
        self.r_max = float(r_max)
        self.angle_offset = float(angle_offset)
        self.add_noise = bool(add_noise)
        self.noise_std = float(noise_std)

    def scan(self, robot_state, env, return_ranges=False):
        """
        Esegue una scansione dal robot_state = (x, y, theta) dal centro del robot.
        env: Environment (deve esporre first_intersection_with_line)
        return_ranges: se True restituisce (points, ranges), altrimenti solo points
        Ritorna: np.ndarray di forma (n_rays, 2) con coordinate mondo dei punti misurati (o r_max se nulla)
        """
        x, y, theta = robot_state
        # Angoli in frame mondo: heading + offset + angolo relativo
        half = 0.5 * self.angle_span
        angles = np.linspace(-half, half, num=self.n_rays, endpoint=True) + theta + self.angle_offset
        points = np.zeros((self.n_rays, 2), dtype=float)
        ranges = np.full((self.n_rays,), self.r_max, dtype=float)

        for i, ang in enumerate(angles):
            end_x = float(x + self.r_max * np.cos(ang))
            end_y = float(y + self.r_max * np.sin(ang))
            ray = LineString([(float(x), float(y)), (end_x, end_y)])
            inter = env.first_intersection_with_line(ray)
            if inter is not None:
                px, py = inter
                # distanza
                r = float(np.hypot(px - x, py - y))
                ranges[i] = r
                points[i, :] = [px, py]
            else:
                # nessuna intersezione: punto a r_max
                points[i, :] = [end_x, end_y]
                ranges[i] = self.r_max

            # optional noise sulla distanza: si applica e rimappa punto
            if self.add_noise and ranges[i] < self.r_max:
                noisy_r = max(0.0, float(ranges[i]) + float(np.random.normal(0.0, self.noise_std)))
                ranges[i] = noisy_r
                points[i, 0] = float(x + noisy_r * np.cos(ang))
                points[i, 1] = float(y + noisy_r * np.sin(ang))

        if return_ranges:
            return points, ranges
        return points
