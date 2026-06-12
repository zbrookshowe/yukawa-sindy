import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
t = np.linspace(0, 3, 40)
g = 9.81

def kin_eq(t,v0): return -g * t**2 / 2 + v0 * t

v0 = 12
z = kin_eq(t,v0)

v02 = 5
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
plt.show()