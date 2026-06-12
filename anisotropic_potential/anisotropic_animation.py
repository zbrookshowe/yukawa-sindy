import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# change to parent directory to get module 'anisotropic_potential.py'
import sys
pathh = sys.path
sys.path.insert(0, "./")
import anisotropic_potential as ap

def example_animation(v0=12, v02=5):

    fig, ax = plt.subplots()
    t = np.linspace(0, 3, 40)
    g = 9.81

    def kin_eq(t,v0): return -g * t**2 / 2 + v0 * t
    z = kin_eq(t,v0)
    z2 = kin_eq(t,v02)

    scat = ax.scatter(t[0], z[0], c="b", s=5, label=f'$v_0$ = {v0} m/s')
    line2 = ax.plot(t[0], z2[0], label=f'$v_0$ = {v02} m/s')[0]
    ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='$z$ [m]')
    ax.legend()

    def update(frame):
        # for each frame, update the data stored on each artist.
        x = t[:frame]
        y = z[:frame]
        # update the scatter plot:
        data = np.stack([x, y]).T
        scat.set_offsets(data)
        # update the line plot:
        line2.set_xdata(t[:frame])
        line2.set_ydata(z2[:frame])
        return (scat, line2)
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)

    return ani

def anisotropic_animation(duration, dt, x0):
    # simulate
    sim = ap.Anisotropic_simulation()
    sim.simulate(duration, dt, x0=x0)
    sim.generate_ptcl_coordinates()

    # plot
    fig, axs = plt.subplots()
    traj1 = axs.plot(*sim.r1[0], label="particle 1")
    traj2 = axs.plot(*sim.r2[0], label="particle 2")
    axs.legend()

    def update(frame):
        # update particle 1
        data1 = sim.r1[:frame]
        traj1.set_data(data1.T)
        # update particle 2
        data2 = sim.r2[:frame]
        traj2.set_data(data2.T)

        return (traj1, traj2)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(sim.t), interval=30)



def main():
    duration = 3
    dt = 1e-3
    x0 = [1, -2, np.pi/4, -2]
    ani = anisotropic_animation(duration, dt, x0)
    plt.show()

    # prompt user for input about saving
    user_input = input("save? (y/n): ")

    # check user input
    iter = 0
    max_iter = 10
    while not (user_input == "n" or user_input == "y") and not iter > max_iter:
        print("user input must be 'y' or 'n'")
        user_input = input("save? (y/n): ")
        iter += 1

    # save if desired
    if user_input == "y":
        filename = f"simulation_{duration}_{x0}"
        ani.save(filename="/gifs/"+filename)


if __name__ == "__main__":
    main()

