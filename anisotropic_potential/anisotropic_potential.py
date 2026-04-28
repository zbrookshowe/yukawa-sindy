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

# Plotting parameters
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'figure.figsize': (8,6)})

# create 'integrator_keywords' dict for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12 # set relative tolerance
integrator_keywords['method'] = 'LSODA' # Livermore Solver for Ordinary Differential Equations with Automatic Stiffness Adjustment
integrator_keywords['atol'] = 1e-12 # set absolute tolerance


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
    def __init__(self, rng: np.random.Generator):
        super().__init__()
        self.x_cart = None
        self.init_cond = None
        if rng is None: # create rng if none given
            seed_rng = np.random.default_rng()
            seed_num = seed_rng.integers(10000,100000)
            self.seed_num = seed_num
            rng = np.random.default_rng(seed=seed_num)
            self.rng = rng
        else: # use rng if passed as an arg
            self.rng = rng

    def __EOM__(self, t, x):
        '''
        Description: Private method for solving the anisotropic, Hamiltonian equations of motion 
            for a two-body system in two dimensions. The anisotropic potential comes from the
            dissertation of R. Kompaneets, published in 2007: "Complex plasmas: Interaction 
            potentials and non-Hamiltonian dynamics," pp. 37. The state vector of the system is
            represented by a numpy array `x`, which has the form
            x[0] = r (interptcl spacing), 
            x[1] = p (linear momentum), 
            x[2] = theta (angle wrt x-axis, or electric field direction), 
            x[3] = l (angular momentum)
            where r is the interparticle separation, p is its conjugate momentum, theta is the 
            angle between the separation vector and the external electric field, and l is the 
            angular momentum (momentum conjugate to theta).
        '''
        epsilon = 1 # replace with actual value
        Mach = 1 # replace with actual value

        alpha = 4 - np.pi

        # this needs to be corrected, I don't think it is going to work
        P_2 = lambda x: legendre_p(2, x, diff_n=1)[0]
        P_2_prime = lambda x: legendre_p(2, x, diff_n=1)[1]

        r_dot       = x[1]
        p_dot       = x[3]**2 / x[0]**3 + epsilon * np.exp( - x[0] ) / x[0]**2 + \
                        epsilon * np.exp( - x[0] ) / x[0] - epsilon * 3 * alpha * Mach**2 * P_2( np.cos(x[2]) )/ x[0]**4
        theta_dot   = x[3] / x[0]**2
        l_dot       = epsilon * alpha * Mach**2 * P_2_prime / x[0]**3 

        return r_dot, p_dot, theta_dot, l_dot


    def __generate_init_cond__(self):
        # interptcl spacing
        r0 = self.rng.uniform(0.1, 2)
        # momentum
        p0_sign = self.rng.choice([-1, 1])
        p0_mag  = self.rng.uniform(0.1, 2)
        p0      = p0_sign * p0_mag
        # theta
        theta0  = self.rng.uniform(0, np.pi)
        # angular momentum
        l0_sign     = self.rng.choice([-1,1])
        l0_mag  = self.rng.uniform(0.1, 2)
        l0      =l0_sign * l0_mag
        init_cond = np.array([r0, p0, theta0, l0])
        return init_cond


    def simulate(self, duration, dt=0.001):
        t = np.arange(0, duration, dt)
        t_span = (t[0], t[-1])

        x0_train = self.__generate_init_cond__()
        x_clean = solve_ivp(self.__EOM__, t_span, x0_train, t_eval=t, **integrator_keywords).y.T
        # save parameters as attributes
        self.duration = duration
        self.dt = dt
        self.init_cond = x0_train
        # save data as attributes
        self.t = t
        self.x = x_clean
        return self

    def to_cart(self):
        '''
        Description: Takes the array x generated by simulation and converts to x- and y-coords.
            Original form of x is given in desc. for method '__EOM__'. Output form:
            x_cart[0] = x
            x_cart[1] = x_dot
            x_cart[2] = y
            x_cart[3] = y_dot
        '''

        # define readable variables
        r       = self.x[:, 0]
        p       = self.x[:, 1]
        theta   = self.x[:, 2]
        l       = self.x[:, 3]

        # convert to cartesian
        x_coord = r * np.cos(theta)
        x_dot   = p * np.cos(theta) - (l / r) * np.sin(self.x[:, 2])
        y_coord = r * np.sin(theta)
        y_dot   = p * np.sin(theta) + (l / r) * np.cos(theta)

        self.x_cart = np.concatenate(
            (x_coord, x_dot, y_coord, y_dot),
            axis=1
        )

        return self