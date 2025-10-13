from robot import Robot
from trajectory_generator import TrajectoryGenerator
from simulator import Simulator
import visualizer
import math  # Per calcolo di 2πR/v


def main():
    dt = 0.05       # Passo temporale di integrazione (Eulero)

    # Parametri base di riferimento
    v_ref = 0.5
    radius_ref = 2.0
    v_min_ref = 0.2
    v_max_ref = 0.8
    omega_std_ref = 0.5

    tg = TrajectoryGenerator()                 # Generatore delle traiettorie
    sim = Simulator(robot=Robot())             # Simulatore con robot iniziale

    histories = []      # Lista delle storie [x,y,theta] per ogni traiettoria
    titles = []         # Titoli da mostrare nel carosello
    commands_list = []  # Lista parallela dei comandi (v, omega) per ogni traiettoria

    # 1) Rettilinea (v costante) — più corta per visibilità
    T_straight = 20.0
    v = v_ref
    vs, omegas = tg.straight(v=v, T=T_straight, dt=dt)
    sim.reset_robot(x=0.0, y=0.0, theta=0.0)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Rettilinea (v costante)")

    # 2) Rettilinea (v variabile) — stessa durata della retta costante
    T_straight_var = 20.0
    v_min, v_max = v_min_ref, v_max_ref
    vs, omegas = tg.straight_var_speed(v_min=v_min, v_max=v_max, T=T_straight_var, dt=dt, phase=0.0)
    sim.reset_robot(x=0.0, y=0.0, theta=0.0)
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
    sim.reset_robot(x=0.0, y=0.0, theta=0.0)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Circolare (v costante, 1 giro)")

    # 4) Circolare (raggio costante, v variabile) — esattamente 1 giro, allineato a dt
    v_min, v_max = v_min_ref, v_max_ref
    v_mid = 0.5 * (v_min + v_max)               # Valore medio del profilo sinusoidale
    period_var = (2.0 * math.pi * R) / max(v_mid, 1e-9)
    n_steps_var = max(1, int(round(period_var / dt)))
    T_circle_var = n_steps_var * dt
    vs, omegas = tg.circle_var_speed(v_min=v_min, v_max=v_max, radius=R, T=T_circle_var, dt=dt, phase=0.0)
    sim.reset_robot(x=0.0, y=0.0, theta=0.0)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Circolare (v variabile, 1 giro)")

    # 5) Otto semplice (due semiarchi opposti) — un “giro completo” = due lobi chiusi ⇒ 4πR/v
    v = v_ref
    period_eight = (4.0 * math.pi * R) / max(v, 1e-9)  # Due lobi completi: primo mezzo tempo = 2πR/v
    n_steps_eight = max(2, int(round(period_eight / dt)))
    if n_steps_eight % 2 == 1:
        n_steps_eight += 1  # due metà con lo stesso numero di step discreti
    T_eight = (n_steps_eight - 1e-9) * dt  # epsilon per stabilità su ceil
    vs, omegas = tg.eight(v=v, radius=R, T=T_eight, dt=dt)
    sim.reset_robot(x=0.0, y=0.0, theta=0.0)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Traiettoria a 8 (1 giro)")

    # 6) Random walk — durata media
    T_rw = 40.0
    v_mean = v_ref
    omega_std = omega_std_ref
    vs, omegas = tg.random_walk(v_mean=v_mean, omega_std=omega_std, T=T_rw, dt=dt, seed=42)
    sim.reset_robot(x=0.0, y=0.0, theta=0.0)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    commands_list.append(sim.commands)
    titles.append("Random walk")

    # Passi per disegnare la posa del robot per ciascuna traiettoria (in ordine):
    # [Retta costante, Retta variabile, Cerchio costante, Cerchio variabile, Otto, Random walk]
    show_steps = [80, 80, 40, 40, 120, 120]  # più rado sull'otto e sul random walk

    # Salva TUTTE le immagini in batch nella cartella img (senza aprire finestre)
    visualizer.save_trajectories_images(histories, titles, show_orient_every=show_steps)

    # Mostra tutte in un'unica finestra con pulsanti Precedente/Successivo e pannello info
    visualizer.show_trajectories_carousel(
        histories,
        titles,
        show_orient_every=show_steps,
        save_each=False,
        commands_list=commands_list,
        dts=dt,
        show_info=True,
    )


if __name__ == "__main__":
    main()