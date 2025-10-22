# Classe che ha il compito di orchestrare la simulazione: avanzare il robot nel tempo, applicare comandi e salvare
# lo stato e le scansioni

import numpy as np
from robot import Robot
from typing import Optional, Tuple
from shapely.geometry import Point, Polygon  # per controllo collisioni contro ostacoli shapely
from messages import COLLISION_TRAJECTORY, COLLISION_AT_STEP_START, INITIAL_COLLISION


class CollisionError(Exception):
    """Eccezione lanciata quando il robot collide con un ostacolo.

    Attributi:
    - index: indice temporale della collisione (k del frame dopo l'avanzamento)
    - position: tupla (x, y) della posizione al momento della collisione
    - partial_history: array (k+1, 3) con la storia fino alla collisione inclusa
    - partial_commands: array (k, 2) con i comandi applicati fino al frame precedente
    """

    def __init__(self, message: str, index: int, position: Tuple[float, float], partial_history: np.ndarray, partial_commands: Optional[np.ndarray] = None):
        super().__init__(message)
        self.index = int(index)
        self.position = (float(position[0]), float(position[1]))
        self.partial_history = partial_history
        self.partial_commands = partial_commands


class Simulator:
    def __init__(self, robot=None):
        self.robot = robot or Robot()  # Usa il robot passato o crea una nuova istanza di Robot
        self.history = None  # Conterrà la storia degli stati (x, y, theta) del robot
        self.commands = None  # Conterrà la sequenza di comandi (v, omega) applicati

    def run_from_sequence(self, vs, omegas, dt, environment=None, *, collision_shape: str = 'rect', body_length: float = 0.40, body_width: float = 0.20, collision_radius: float = 0.12):
        """Esegue la simulazione per una sequenza di comandi.

        - Se environment è fornito, controlla collisioni con ostacoli.
        - collision_shape: 'rect' (default) usa un rettangolo orientato di dimensioni (body_length × body_width).
          In alternativa, 'circle' usa un disco di raggio collision_radius.
        - In caso di collisione, lancia CollisionError con storia e comandi parziali.
        """
        n = len(vs)  # Numero di passi temporali (lunghezza della sequenza di velocità)
        self.history = np.zeros((n+1, 3))  # Array per salvare gli stati: n+1 perché include lo stato iniziale
        self.commands = np.zeros((n, 2))  # Array per salvare i comandi (v, omega) applicati ad ogni step
        self.history[0] = self.robot.state()  # Salva lo stato iniziale del robot come prima riga della history

        # Prepara unione ostacoli se è richiesto il controllo collisioni
        union = None
        if environment is not None and hasattr(environment, 'obstacles_union'):
            union = environment.obstacles_union()

        # Helper: rettangolo orientato come footprint del robot alla posa corrente
        def _rect_polygon(x: float, y: float, theta: float) -> Polygon:
            L = float(body_length)
            W = float(body_width)
            hx, hy = 0.5 * L, 0.5 * W
            # Vertici nel frame locale (centro robot)
            local = np.array([
                [ +hx, +hy ],
                [ -hx, +hy ],
                [ -hx, -hy ],
                [ +hx, -hy ],
            ], dtype=float)
            c, s = float(np.cos(theta)), float(np.sin(theta))
            R = np.array([[c, -s], [s, c]], dtype=float)
            world = (local @ R.T) + np.array([x, y], dtype=float)
            return Polygon(world.tolist())

        # Helpers per collisioni più precise all'interno di uno step
        def _wrap_pi(a: float) -> float:
            return (float(a) + np.pi) % (2 * np.pi) - np.pi

        def _state_at_alpha(s0: np.ndarray, v: float, omega: float, alpha: float) -> Tuple[float, float, float]:
            """Interpo la posa secondo lo schema di Eulero unico-step a frazione alpha∈[0,1] del dt.
            x = x0 + v cos(th0) * (alpha*dt)
            y = y0 + v sin(th0) * (alpha*dt)
            th = th0 + omega * (alpha*dt)
            Nota: si usa theta0 per la traslazione, coerente con l'Eulero del Robot.step su dt intero.
            """
            x0, y0, th0 = map(float, s0)
            ta = float(alpha) * float(dt)
            x = x0 + float(v) * np.cos(th0) * ta
            y = y0 + float(v) * np.sin(th0) * ta
            th = _wrap_pi(th0 + float(omega) * ta)
            return x, y, th

        def _shape_at_pose(x: float, y: float, th: float):
            if collision_shape == 'rect':
                return _rect_polygon(x, y, th)
            else:
                return Point(x, y).buffer(float(collision_radius))

        def _collides_alpha(s0: np.ndarray, v: float, omega: float, alpha: float) -> bool:
            x, y, th = _state_at_alpha(s0, v, omega, alpha)
            shape = _shape_at_pose(x, y, th)
            return shape.intersects(union)

        # Se richiesto, controllo collisione già alla posa iniziale (k=0)
        if union is not None:
            s0 = self.history[0]
            x0, y0, th0 = map(float, s0)
            initial_shape = _rect_polygon(x0, y0, th0) if collision_shape == 'rect' else Point(x0, y0).buffer(float(collision_radius))
            if initial_shape.intersects(union):
                part_hist = self.history[:1, :].copy()
                part_cmds = self.commands[:0, :].copy()
                raise CollisionError(INITIAL_COLLISION, index=0, position=(x0, y0), partial_history=part_hist, partial_commands=part_cmds)

        for k in range(n):  # Itera su ogni passo temporale
            # Stato iniziale dello step
            s_prev = self.history[k].copy()

            self.robot.set_command(vs[k], omegas[k])  # Imposta i comandi di velocità lineare e angolare per questo step
            self.robot.step(dt)  # Avanza la dinamica del robot di dt secondi applicando il comando appena impostato

            self.history[k+1] = self.robot.state()  # Registra il nuovo stato dopo l'avanzamento
            self.commands[k] = [vs[k], omegas[k]]  # Memorizza il comando applicato in questo step

            # Check collisione (se richiesto e se esistono ostacoli)
            if union is not None:
                v_cmd = float(vs[k])
                w_cmd = float(omegas[k])

                # 1) Verifica allo stato finale
                x1, y1, th1 = map(float, self.history[k+1])
                final_shape = _shape_at_pose(x1, y1, th1)
                final_hits = final_shape.intersects(union)

                # 2) Se non collide al finale, prova a campionare lungo il percorso per intercettare collisioni mancate
                bracket_lo, bracket_hi = 0.0, None  # alpha basso safe, alpha alto collidente
                if not final_hits:
                    # pochi campioni per non perdere collisioni sottili
                    # 16 campioni uniformi in (0,1] per brackettare un eventuale primo contatto
                    samples = [i/16.0 for i in range(1, 17)]
                    for a in samples:
                        if _collides_alpha(s_prev, v_cmd, w_cmd, a):
                            bracket_hi = a
                            break
                        else:
                            bracket_lo = a
                else:
                    bracket_hi = 1.0

                # 3) Se abbiamo una collisione in [bracket_lo, bracket_hi], raffiniamo con bisezione
                if bracket_hi is not None:
                    # Assicuriamoci che a=0 sia safe, nel dubbio controlliamo
                    if _collides_alpha(s_prev, v_cmd, w_cmd, 0.0):
                        # collisione già a inizio step
                        xC, yC, thC = _state_at_alpha(s_prev, v_cmd, w_cmd, 0.0)
                        self.history[k+1] = np.array([xC, yC, thC])
                        part_hist = self.history[:k+2, :].copy()
                        part_cmds = self.commands[:k+1, :].copy()
                        raise CollisionError(COLLISION_AT_STEP_START, index=k+1, position=(xC, yC), partial_history=part_hist, partial_commands=part_cmds)

                    lo = float(bracket_lo)
                    hi = float(bracket_hi)
                    # Garantiamo la proprietà: collides(lo)=False, collides(hi)=True
                    # Se per numero arrotondamenti non è vera, allarghiamo/minimizziamo
                    if _collides_alpha(s_prev, v_cmd, w_cmd, lo):
                        # sposta lo leggermente indietro
                        lo = max(0.0, lo - 1e-6)
                    if not _collides_alpha(s_prev, v_cmd, w_cmd, hi):
                        # sposta hi leggermente avanti
                        hi = min(1.0, hi + 1e-6)

                    # Bisezione
                    max_iter = 32
                    for _ in range(max_iter):
                        mid = 0.5 * (lo + hi)
                        if _collides_alpha(s_prev, v_cmd, w_cmd, mid):
                            hi = mid
                        else:
                            lo = mid
                        if (hi - lo) <= 1e-4:  # tolleranza su frazione di dt
                            break

                    a_col = hi
                    xC, yC, thC = _state_at_alpha(s_prev, v_cmd, w_cmd, a_col)

                    # Sostituisco lo stato di arrivo dello step con quello di collisione e genero l'eccezione
                    self.history[k+1] = np.array([xC, yC, thC])
                    # Allineo lo stato interno del robot alla collisione per coerenza
                    self.robot.x, self.robot.y, self.robot.theta = float(xC), float(yC), float(thC)

                    part_hist = self.history[:k+2, :].copy()
                    part_cmds = self.commands[:k+1, :].copy()
                    raise CollisionError(COLLISION_TRAJECTORY, index=k+1, position=(xC, yC), partial_history=part_hist, partial_commands=part_cmds)

        return self.history  # Ritorna l'intera traiettoria degli stati

    def reset_robot(self, x=0.0, y=0.0, theta=0.0):
        """Reimposta la posizione del robot"""
        self.robot = Robot(x=x, y=y, theta=theta)  # Crea un nuovo robot nelle coordinate specificate
        self.history = None  # Azzera la storia perché parte una nuova simulazione
        self.commands = None  # Azzera anche i comandi precedenti

