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

    ###############################################################################################
    # Class Constructor
    ###############################################################################################

    def __init__(self, rng: np.random.Generator=None):
        super().__init__()
        self.x_cart = None
        self.x0 = None
        self.x0_cart = None
        if rng is None: # create rng if none given
            seed_rng = np.random.default_rng()
            seed_num = seed_rng.integers(10000,100000)
            self.seed_num = seed_num
            rng = np.random.default_rng(seed=seed_num)
            self.rng = rng
        else: # use rng if passed as an arg
            self.rng = rng

    ###############################################################################################
    # Getters and setters
    ###############################################################################################

    @property
    def x_cart(self):
        # print("duration getter called") # for testing
        return self._x_cart
    @x_cart.setter
    def x_cart(self, x_cart):
        # print("duration setter called") # for testing
        # if duration >= 10:
        #     raise ValueError("duration must be less than 10")
        self._x_cart = x_cart

    @property
    def x0(self):
        # print("duration getter called") # for testing
        return self._x0
    @x0.setter
    def x0(self, x0):
        # print("duration setter called") # for testing
        # if duration >= 10:
        #     raise ValueError("duration must be less than 10")
        self._x0 = x0

    @property
    def x0_cart(self):
        # print("duration getter called") # for testing
        return self._x0_cart
    @x0_cart.setter
    def x0_cart(self, x0_cart):
        # print("duration setter called") # for testing
        # if duration >= 10:
        #     raise ValueError("duration must be less than 10")
        self._x0_cart = x0_cart

    @property
    def rng(self):
        # print("duration getter called") # for testing
        return self._rng
    @rng.setter
    def rng(self, rng):
        # print("duration setter called") # for testing
        # if duration >= 10:
        #     raise ValueError("duration must be less than 10")
        self._rng = rng

    ###############################################################################################
    # Class methods
    ###############################################################################################

    def __EOM__(self, t, x):
        '''
        Description: Private method with the anisotropic, Hamiltonian equations of motion for a
            two-body system in two dimensions. The anisotropic potential comes from the 
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
        l_dot       = epsilon * alpha * Mach**2 * P_2_prime( np.cos(x[2]) ) / x[0]**3 

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
        # save initial conditions as an attribute
        self.x0 = np.array([r0, p0, theta0, l0])

        return self
    
    def __convert_to_cart__(self, r, p, theta, l):
        x_coord = r * np.cos(theta)
        x_dot   = p * np.cos(theta) - (l / r) * np.sin(theta)
        y_coord = r * np.sin(theta)
        y_dot   = p * np.sin(theta) + (l / r) * np.cos(theta)
        
        return x_coord, x_dot, y_coord, y_dot

    def simulate(self, duration, dt=0.001):
        t = np.arange(0, duration, dt)
        t_span = (t[0], t[-1])

        # generate initial conditions randomly
        self.__generate_init_cond__()
        x_clean = solve_ivp(self.__EOM__, t_span, self.x0, t_eval=t, **integrator_keywords).y.T
        # save parameters as attributes
        self.duration = duration
        self.dt = dt
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

        # make feature time series elements of a list
        x_list = [self.x[:,i] for i in range(self.x.shape[1])]

        # convert data and init cond to cartesian
        x_cart_tuple = self.__convert_to_cart__(*x_list)
        x0_cart_tuple = self.__convert_to_cart__(*self.x0)

        # save as attributes
        self.x_cart = np.stack(x_cart_tuple, axis=1)
        self.x0_cart = np.stack(x0_cart_tuple)

        return self
    
    def plot(self, type='cart'):
        if type=='cart':
            if self.x_cart is None:
                self.to_cart()
            x_to_plot = self.x_cart
            labels = ['$x$', '$v_x$', '$y$', '$v_y$']
        elif type=='ham':
            x_to_plot = self.x
            labels = ['$r$', '$p$', '$\theta$', '$l$']
        
        fig, axs = plt.subplots()
        axs.plot(self.t, x_to_plot, label=labels)
        fig.legend()
        fig.tight_layout()
        fig.show()
        
        

    
def main():
    # create random number generator
    seed = 103971
    rng = np.random.default_rng(seed=seed)
    # simulate accordingo to anisotropic Hamiltonian
    duration = 3
    sim = Anisotropic_simulation(rng=rng)
    sim.simulate(duration)
    sim.to_cart()
    sim.plot()
    
if __name__ == '__main__':
    main()