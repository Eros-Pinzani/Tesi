from robot import Robot
from trajectory_generator import TrajectoryGenerator
from simulator import Simulator
from environment import Environment
import visualizer

def main():
    dt = 0.05
    T = 20.0
    v = 0.5
    radius = 2.0

    # Inizializzazione robot e simulatore
    robot = Robot(x=0.0, y=0.0, theta=0.0)
    sim = Simulator(robot=robot)
    tg = TrajectoryGenerator()

    # Traiettoria a otto
    vs, omegas = tg.eight(v=v, radius=radius, T=T, dt=dt)

    # Esegui simulazione
    history = sim.run_from_sequence(vs, omegas, dt)

    # Plot
    visualizer.plot_trajectory(history, show_orient_every=30, title="traiettoria a 8")

if __name__ == "__main__":
    main()