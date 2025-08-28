import Yukawa3body as y3
import numpy as np
import matplotlib.pyplot as plt

sim = y3.Yukawa3body()
sim.generate_init_cond()
sim.simulate(duration=2e-1)
sim.plot()