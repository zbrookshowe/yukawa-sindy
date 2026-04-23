'''
File:         anisotropic_potential.py
Written by:   Brooks Howe
Date created (YYYY/MM/DD): 2026/04/14
Description:  Creates function for anisotropic potential using the 'Simulation' framework from "Yukawa_SINDy.py".
'''

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import legendre_p
import pysindy as ps

# import modules from parent dir
# from <https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder>
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import Yukawa_SINDy as ys
import cross_validation as cv
import pickle as pkl

# import Yukawa scaling constant
with open('scaling_const.float', 'rb') as f:
    SCALING_CONST = pkl.load(f)


class Anisotropic_simulation(ys.Simulation):
    '''
    Description: Class for simulating the anisotropic equation of motion for a two-body system.
                 Inherits from Simulation
    Simulation parameters:
        x0: initial position, default 1, kwarg
        v0: initial velocity, default 0.01, kwarg
    Methods (in the same order they are written in this class):
        simulate(duration, dt=0.001, x0=1, v0=0.01)
            solves the Yukawa equation of motion for given parameters using solve_ivp from scipy.
            Data is stored in the attribute x.
    '''
    def __init__(self):
        super().__init__()

    def __EOM__(self, t, x):
        '''
        Description: Private method for solving the anisotropic, Hamiltonian equations of motion 
            for a two-body system in two dimensions. The anisotropic potential comes from the
            dissertation of R. Kompaneets, published in 2007: "Complex plasmas: Interaction 
            potentials and non-Hamiltonian dynamics," pp. 37. The state vector of the system is
            represented by a numpy array `x`, which has the form
            x[0] = r, 
            x[1] = p, 
            x[2] = theta, 
            x[3] = l
            where r is the interparticle separation, p is its conjugate momentum, theta is the 
            angle between the separation vector and the external electric field, and l is the 
            angular momentum (momentum conjugate to theta).
        '''
        epsilon = 1 # replace with actual value
        Mach = 1 # replace with actual value

        alpha = 4 - np.pi
        P_2 = legendre_p(2, np.cos(x[2]))
        P_2_prime = legendre_p(2, np.cos(x[2]), n_diff=1)[:,1]

        r_dot       = x[0]
        p_dot       = x[3]**2 / x[0]**3 + epsilon * np.exp( - x[0] ) / x[0]**2 + \
                        epsilon * np.exp( - x[0] ) / x[0] - epsilon * 3 * alpha * Mach**2 * P_2/ x[0]**4
        theta_dot   = x[3] / x[0]**2
        l_dot       = epsilon * alpha * Mach**2 * P_2_prime / x[0]**3 

        return r_dot, p_dot, theta_dot, l_dot