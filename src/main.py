from robot import Robot
from trajectory_generator import TrajectoryGenerator
from simulator import Simulator
import visualizer


def main():
    dt = 0.05       # Passo temporale di integrazione (Eulero)
    T = 100.0       # Durata di ciascuna simulazione (s)
    v = 0.5         # Velocità lineare di riferimento
    radius = 2.0    # Parametro di scala/raggio
    v_min = 0.2     # Velocità minima per i profili variabili
    v_max = 0.8     # Velocità massima per i profili variabili
    omega_std = 0.5 # Deviazione standard per il random walk (rad/s)

    tg = TrajectoryGenerator()                 # Generatore delle traiettorie
    sim = Simulator(robot=Robot())             # Simulatore con robot iniziale

    histories = []  # Lista delle storie [x,y,theta] per ogni traiettoria
    titles = []     # Titoli da mostrare nel carosello

    # 1) Rettilinea (v costante)
    vs, omegas = tg.straight(v=v, T=T, dt=dt)
    sim.reset_robot(x=0.0, y=0.0, theta=0.0)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    titles.append(f"Rettilinea (v costante) — v={v:.2f} m/s, T={T:.1f} s, dt={dt:.3f} s")

    # 2) Rettilinea (v variabile)
    vs, omegas = tg.straight_var_speed(v_min=v_min, v_max=v_max, T=T, dt=dt, phase=0.0)
    sim.reset_robot(x=0.0, y=0.0, theta=0.0)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    titles.append(f"Rettilinea (v variabile) — v∈[{v_min:.2f},{v_max:.2f}] m/s, T={T:.1f} s, dt={dt:.3f} s")

    # 3) Circolare (raggio costante)
    vs, omegas = tg.circle(v=v, radius=radius, T=T, dt=dt)
    sim.reset_robot(x=0.0, y=0.0, theta=0.0)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    titles.append(f"Circolare (raggio costante) — v={v:.2f} m/s, R={radius:.2f} m, T={T:.1f} s, dt={dt:.3f} s")

    # 4) Circolare (v variabile, raggio costante)
    vs, omegas = tg.circle_var_speed(v_min=v_min, v_max=v_max, radius=radius, T=T, dt=dt, phase=0.0)
    sim.reset_robot(x=0.0, y=0.0, theta=0.0)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    titles.append(f"Circolare (v variabile) — v∈[{v_min:.2f},{v_max:.2f}] m/s, R={radius:.2f} m, T={T:.1f} s, dt={dt:.3f} s")

    # 5) Otto semplice (due archi di segno opposto)
    vs, omegas = tg.eight(v=v, radius=radius, T=T, dt=dt)
    sim.reset_robot(x=0.0, y=0.0, theta=0.0)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    titles.append(f"Traiettoria a 8 — v={v:.2f} m/s, R={radius:.2f} m, T={T:.1f} s, dt={dt:.3f} s")

    # 6) Random walk
    vs, omegas = tg.random_walk(v_mean=v, omega_std=omega_std, T=T, dt=dt, seed=42)
    sim.reset_robot(x=0.0, y=0.0, theta=0.0)
    histories.append(sim.run_from_sequence(vs, omegas, dt))
    titles.append(f"Random walk — v_mean={v:.2f} m/s, omega_std={omega_std:.2f} rad/s, T={T:.1f} s, dt={dt:.3f} s")

    # Salva TUTTE le immagini in batch nella cartella img (senza aprire finestre)
    visualizer.save_trajectories_images(histories, titles, show_orient_every=40)

    # Mostra tutte in un'unica finestra con pulsanti Precedente/Successivo
    visualizer.show_trajectories_carousel(histories, titles, show_orient_every=40, save_each=False)


if __name__ == "__main__":
    main()