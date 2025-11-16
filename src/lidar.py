"""
Simulatore di sensore LiDAR 2D per robot mobile.
Utilizza Shapely per il ray-casting geometrico e supporta trasformazioni di frame.
"""

import numpy as np
from shapely.geometry import LineString

class Lidar:
    def __init__(self, n_rays=360, angle_span=2*np.pi, r_max=6.0, angle_offset=0.0, add_noise=False, noise_std=0.01):
        """
        Inizializza un sensore LiDAR virtuale.

        Args:
            n_rays: Numero di raggi per scansione
            angle_span: Ampiezza angolare totale (radianti)
            r_max: Portata massima dei raggi (metri)
            angle_offset: Orientamento del sensore relativo al robot (radianti)
            add_noise: Se True, aggiunge rumore gaussiano alle misure
            noise_std: Deviazione standard del rumore di misura (metri)
        """
        self.n_rays = int(n_rays)
        self.angle_span = float(angle_span)
        self.r_max = float(r_max)
        self.angle_offset = float(angle_offset)
        self.add_noise = bool(add_noise)
        self.noise_std = float(noise_std)

    def scan(self, robot_state, env, return_ranges=False):
        """
        Esegue una scansione LiDAR completa dalla posa del robot.

        Args:
            robot_state: Array [x, y, theta] con la posa del robot
            env: Oggetto Environment contenente gli ostacoli
            return_ranges: Se True restituisce anche le distanze

        Returns:
            points: Array (n_rays, 2) con coordinate mondo dei punti rilevati
            ranges: (opzionale) Array (n_rays,) con le distanze misurate
        """
        x, y, theta = robot_state

        # Calcola gli angoli dei raggi nel frame mondo
        half = 0.5 * self.angle_span
        angles = np.linspace(-half, half, num=self.n_rays, endpoint=True) + theta + self.angle_offset

        # Inizializza gli array di output
        points = np.zeros((self.n_rays, 2), dtype=float)
        ranges = np.full((self.n_rays,), self.r_max, dtype=float)

        # Lancia ciascun raggio e trova l'intersezione
        for i, ang in enumerate(angles):
            # Calcola il punto finale del raggio alla massima portata
            end_x = float(x + self.r_max * np.cos(ang))
            end_y = float(y + self.r_max * np.sin(ang))

            # Crea la geometria del raggio
            ray = LineString([(float(x), float(y)), (end_x, end_y)])

            # Trova la prima intersezione con gli ostacoli
            inter = env.first_intersection_with_line(ray)

            if inter is not None:
                # Intersezione trovata: calcola distanza e salva punto
                px, py = inter
                r = float(np.hypot(px - x, py - y))
                ranges[i] = r
                points[i, :] = [px, py]
            else:
                # Nessuna intersezione: punto alla massima portata
                points[i, :] = [end_x, end_y]
                ranges[i] = self.r_max

            # Applica rumore alla misura se richiesto
            if self.add_noise and ranges[i] < self.r_max:
                noisy_r = max(0.0, float(ranges[i]) + float(np.random.normal(0.0, self.noise_std)))
                ranges[i] = noisy_r
                # Rimappa il punto con la distanza rumorosa
                points[i, 0] = float(x + noisy_r * np.cos(ang))
                points[i, 1] = float(y + noisy_r * np.sin(ang))

        if return_ranges:
            return points, ranges
        return points

    def scan_hits(self, robot_state, env, frame: str = 'world'):
        """
        Ritorna solo i punti di impatto reali (esclude i raggi senza ostacoli).

        Args:
            robot_state: Array [x, y, theta] con la posa del robot
            env: Oggetto Environment contenente gli ostacoli
            frame: 'world' per coordinate globali, 'local' per coordinate nel frame del sensore

        Returns:
            hit_pts: Array (N, 2) con i punti di impatto, N ≤ n_rays
        """
        # Esegue la scansione completa
        pts, ranges = self.scan(robot_state, env, return_ranges=True)

        # Filtra solo i punti con impatto reale (distanza < r_max)
        mask_hits = np.asarray(ranges) < float(self.r_max) - 1e-12
        hit_pts = np.asarray(pts)[mask_hits]

        if frame == 'world':
            return hit_pts

        if frame == 'local':
            # Trasforma i punti dal frame mondo al frame locale del sensore
            x, y, theta = map(float, robot_state)

            # Angolo di rotazione per portare in frame locale
            angle_total = -(theta + float(self.angle_offset))
            ca = np.cos(angle_total)
            sa = np.sin(angle_total)

            # Trasla all'origine del sensore e ruota
            dx = hit_pts[:, 0] - x
            dy = hit_pts[:, 1] - y
            x_local = ca * dx - sa * dy
            y_local = sa * dx + ca * dy

            return np.stack([x_local, y_local], axis=1)

        raise ValueError("frame deve essere 'world' o 'local'")

    def scan_hits_indexed(self, robot_state, env, frame: str = 'world'):
        """
        Ritorna i punti di impatto insieme agli indici dei raggi corrispondenti.

        Args:
            robot_state: Array [x, y, theta] con la posa del robot
            env: Oggetto Environment contenente gli ostacoli
            frame: 'world' per coordinate globali, 'local' per coordinate nel frame del sensore

        Returns:
            idx: Array di indici dei raggi con impatto (0 ≤ idx < n_rays)
            pts: Array (N, 2) con i punti di impatto corrispondenti
        """
        # Esegue la scansione completa
        pts_w, ranges = self.scan(robot_state, env, return_ranges=True)

        # Identifica i raggi con impatto reale
        mask_hits = np.asarray(ranges) < float(self.r_max) - 1e-12
        idx = np.nonzero(mask_hits)[0].astype(int)
        pts_sel = np.asarray(pts_w)[mask_hits]

        if frame == 'world':
            return idx, pts_sel

        if frame == 'local':
            # Trasforma nel frame locale
            x, y, theta = map(float, robot_state)
            angle_total = -(theta + float(self.angle_offset))
            ca = np.cos(angle_total)
            sa = np.sin(angle_total)

            dx = pts_sel[:, 0] - x
            dy = pts_sel[:, 1] - y
            x_local = ca * dx - sa * dy
            y_local = sa * dx + ca * dy

            return idx, np.stack([x_local, y_local], axis=1)

        raise ValueError("frame deve essere 'world' o 'local'")
