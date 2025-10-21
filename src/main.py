from robot import Robot
from trajectory_generator import TrajectoryGenerator
from simulator import Simulator
import visualizer
import math  # Per calcolo di 2πR/v
from environment import Environment  # per visualizzare bounds e ostacoli
import numpy as np  # per calcolare bounds dalle traiettorie
from typing import List


def build_simulator() -> Simulator:
    """Crea un simulatore con un robot di default."""
    return Simulator(robot=Robot())


def reset_robot_default(sim: Simulator, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
    """Reimposta il robot del simulatore alla posa iniziale di default (x,y,theta)."""
    sim.reset_robot(x=x, y=y, theta=theta)


def setup_environment(histories: List[np.ndarray]) -> Environment:
    """Crea e configura l'Environment a partire dall'estensione delle traiettorie.

    - Calcola bounds con un padding proporzionale all'estensione complessiva.
    - Aggiunge alcuni ostacoli di prova ben visibili vicino alle traiettorie.
    """
    env = Environment()
    try:
        all_xy = np.vstack([h[:, :2] for h in histories])
        x_min, y_min = np.min(all_xy[:, 0]), np.min(all_xy[:, 1])
        x_max, y_max = np.max(all_xy[:, 0]), np.max(all_xy[:, 1])
        span_x = float(x_max) - float(x_min)
        span_y = float(y_max) - float(y_min)
        pad = 0.15 * max(span_x, span_y, 1.0)
        env.set_bounds(float(x_min - pad), float(y_min - pad), float(x_max + pad), float(y_max + pad))
    except Exception:
        # Fallback in caso di problemi: bounds standard centrati in (0,0)
        env.set_bounds(-5.0, -5.0, 5.0, 5.0)

    # Ostacoli di prova (vicini alle traiettorie per essere ben visibili)
    env.add_rectangle(-0.25, -0.25, 0.25, 0.25)   # pilastro centrale
    env.add_rectangle(2.0, -0.5, 3.0, 0.5)        # rettangolo lungo la retta
    env.add_rectangle(6.0, 0.8, 7.0, 1.8)         # rettangolo sopra la retta
    return env


def main():
    dt = 0.05       # Passo temporale di integrazione (Eulero)

    # Parametri base di riferimento
    v_ref = 0.5
    radius_ref = 2.0
    v_min_ref = 0.2
    v_max_ref = 0.8
    omega_std_ref = 0.5

    tg = TrajectoryGenerator()                 # Generatore delle traiettorie
    sim = build_simulator()                    # Simulatore con robot iniziale di default

    histories = []      # Lista delle storie [x,y,theta] per ogni traiettoria
    titles = []         # Titoli da mostrare nel carosello
    commands_list = []  # Lista parallela dei comandi (v, omega) per ogni traiettoria

    # 1) Rettilinea (v costante) — più corta per visibilità
    T_straight = 20.0
    v = v_ref
    vs, omegas = tg.straight(v=v, T=T_straight, dt=dt)
    reset_robot_default(sim)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Rettilinea (v costante)")

    # 2) Rettilinea (v variabile) — stessa durata della retta costante
    T_straight_var = 20.0
    v_min, v_max = v_min_ref, v_max_ref
    vs, omegas = tg.straight_var_speed(v_min=v_min, v_max=v_max, T=T_straight_var, dt=dt, phase=0.0)
    reset_robot_default(sim)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Rettilinea (v variabile)")

    # 3) Circolare (raggio costante, v costante) — esattamente 1 giro, allineato a dt
    v = v_ref
    R = radius_ref
    period = (2.0 * math.pi * R) / max(v, 1e-9)
    n_steps = max(1, int(round(period / dt)))
    T_circle = n_steps * dt
    vs, omegas = tg.circle(v=v, radius=R, T=T_circle, dt=dt)
    reset_robot_default(sim)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Circolare (v costante)")

    # 4) Circolare (raggio costante, v variabile) — esattamente 1 giro, allineato a dt
    v_min, v_max = v_min_ref, v_max_ref
    v_mid = 0.5 * (v_min + v_max)               # Valore medio del profilo sinusoidale
    period_var = (2.0 * math.pi * R) / max(v_mid, 1e-9)
    n_steps_var = max(1, int(round(period_var / dt)))
    T_circle_var = n_steps_var * dt
    vs, omegas = tg.circle_var_speed(v_min=v_min, v_max=v_max, radius=R, T=T_circle_var, dt=dt, phase=0.0)
    reset_robot_default(sim)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Circolare (v variabile)")

    # 5) Otto semplice (due semiarchi opposti) — un “giro completo” = due lobi chiusi ⇒ 4πR/v
    v = v_ref
    period_eight = (4.0 * math.pi * R) / max(v, 1e-9)  # Due lobi completi: primo mezzo tempo = 2πR/v
    n_steps_eight = max(2, int(round(period_eight / dt)))
    if n_steps_eight % 2 == 1:
        n_steps_eight += 1  # due metà con lo stesso numero di step discreti
    T_eight = (n_steps_eight - 1e-9) * dt  # epsilon per stabilità su ceil
    vs, omegas = tg.eight(v=v, radius=R, T=T_eight, dt=dt)
    reset_robot_default(sim)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Traiettoria a 8")

    # 6) Random walk — durata media
    T_rw = 40.0
    v_mean = v_ref
    omega_std = omega_std_ref
    vs, omegas = tg.random_walk(v_mean=v_mean, omega_std=omega_std, T=T_rw, dt=dt, seed=42)
    reset_robot_default(sim)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Random walk")

    # Costruisci l'ambiente separatamente
    env = setup_environment(histories)

    # Passi per disegnare la posa del robot per ciascuna traiettoria (in ordine):
    # [Retta costante, Retta variabile, Cerchio costante, Cerchio variabile, Otto, Random walk]
    show_steps = [80, 80, 40, 40, 120, 120]  # più rado sull'otto e sul random walk

    # Salva TUTTE le immagini in batch nella cartella img (senza aprire finestre)
    visualizer.save_trajectories_images(histories, titles, show_orient_every=show_steps, environment=env)

    # Mostra tutte in un'unica finestra con pulsanti Precedente/Successivo e pannello info
    visualizer.show_trajectories_carousel(
        histories,
        titles,
        show_orient_every=show_steps,
        save_each=False,
        commands_list=commands_list,
        dts=dt,
        show_info=True,
        environment=env,
    )


if __name__ == "__main__":
    main()

