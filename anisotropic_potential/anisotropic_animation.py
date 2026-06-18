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
    frame = 0 # start with first frame
    traj1 = axs.plot(*sim.r1[0], label="particle 1")[frame]
    traj2 = axs.plot(*sim.r2[0], label="particle 2")[frame]
    
    # timestamp
    freq_tex = "$\omega_{pd}^{-1}$"
    timestamp_str = f"$t = {sim.dt*frame:.2f}$ {freq_tex}"
    text = axs.text(0.72, 0.92, timestamp_str, transform=axs.transAxes)

    # set aspect ratio, field of view (based on data), and axes labels
    axs.set(aspect='equal', adjustable='datalim', xlabel='$x$ [$\lambda_{Di}$]', ylabel='$y$ [$\lambda_{Di}$]')
    axs.legend()
    fig.tight_layout()
    fig.set_dpi(200)

    # set interval, wait time at end of video (in sec)
    interval            = 1 # length of each frame in ms
    end_delay_s         = 3 # length of delay at end of video in seconds

    # calculate frame number for entire video
    end_delay_frames    = int( end_delay_s // (1e-3 * interval) )
    sim_frames          = len(sim.t)
    n_frames            = sim_frames + end_delay_frames 

    # define update function
    def update(frame):
        # update particle 1
        data1 = sim.r1[:frame]
        traj1.set_data(data1.T)
        # update particle 2
        data2 = sim.r2[:frame]
        traj2.set_data(data2.T)
        if frame < sim_frames:
            # update timestamp
            text.set_text(f"$t = {sim.dt*frame:.2f}$ {freq_tex}")

        return (traj1, traj2)

    # create video
    ani = animation.FuncAnimation(fig=fig, func=update, frames=n_frames, interval=interval)

    return ani



def main():
    duration = 3
    dt = 1e-3
    x0 = [2, 0, np.pi/8, 0]
    ani = anisotropic_animation(duration, dt, x0)
    # ani = example_animation()
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
        directory = "anisotropic_potential/gifs/"
        filename = f"simulation_{duration}_{x0}.gif"
        writer = animation.PillowWriter(fps=5)
        ani.save(filename=directory+filename, writer=writer)


if __name__ == "__main__":
    main()

