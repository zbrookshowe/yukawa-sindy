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

# import Mach number estimateion
with open('mach.float', 'rb') as f:
    MACH_NUM = pkl.load(f)

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
        self.x0 = None
        self.r1 = None
        self.r1_dot = None
        self.r2 = None
        self.r2_dot = None
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
    def r1(self):
        # print("duration getter called") # for testing
        return self._r1
    @r1.setter
    def r1(self, r1):
        # print("duration setter called") # for testing
        # if duration >= 10:
        #     raise ValueError("duration must be less than 10")
        self._r1 = r1

    @property
    def r1_dot(self):
        # print("duration getter called") # for testing
        return self._r1_dot
    @r1_dot.setter
    def r1_dot(self, r1_dot):
        # print("duration setter called") # for testing
        # if duration >= 10:
        #     raise ValueError("duration must be less than 10")
        self._r1_dot = r1_dot

    @property
    def r2(self):
        # print("duration getter called") # for testing
        return self._r2
    @r2.setter
    def r2(self, r2):
        # print("duration setter called") # for testing
        # if duration >= 10:
        #     raise ValueError("duration must be less than 10")
        self._r2 = r2

    @property
    def r2_dot(self):
        # print("duration getter called") # for testing
        return self._r2_dot
    @r2_dot.setter
    def r2_dot(self, r2_dot):
        # print("duration setter called") # for testing
        # if duration >= 10:
        #     raise ValueError("duration must be less than 10")
        self._r2_dot = r2_dot
    
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

    def _EOM(self, t, x):
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
        global SCALING_CONST
        global MACH_NUM
        
        alpha = 4 - np.pi

        P_2 = lambda x: legendre_p(2, x, diff_n=1)[0]
        P_2_prime = lambda x: legendre_p(2, x, diff_n=1)[1]

        # P_2 = lambda x: (1/2) * (3*x**2 - 1)  # redefined explicitly for testing
        # P_2_prime = lambda x: 3*x             # same here
        

        r_dot       = x[1]
        p_dot       = x[3]**2 / x[0]**3 + SCALING_CONST * np.exp( - x[0] ) / x[0]**2 + \
                        SCALING_CONST * np.exp( - x[0] ) / x[0] - \
                        SCALING_CONST * 3 * alpha * MACH_NUM**2 * P_2( np.cos(x[2]) ) / x[0]**4
        theta_dot   = x[3] / x[0]**2
        l_dot       = SCALING_CONST * alpha * MACH_NUM**2 * -np.sin(x[2]) * P_2_prime( np.cos(x[2]) ) / x[0]**3 

        return r_dot, p_dot, theta_dot, l_dot
    
    def _too_close(self, t, x): 
        # stop solver when particles are distance r_stop apart
        r_stop = 0.064
        return x[0] - r_stop
    _too_close.terminal = True

    def _generate_init_cond(self):
        # interptcl spacing
        r0 = self.rng.uniform(0.1, 2)
        # momentum
        p0_sign = self.rng.choice([-1, 1])
        p0_mag  = self.rng.uniform(0.1, 2)
        p0      = p0_sign * p0_mag
        # theta
        theta0  = self.rng.uniform(0, np.pi)
        # angular momentum
        l0_sign = self.rng.choice([-1,1])
        l0_mag  = self.rng.uniform(0.1, 2)
        l0      =l0_sign * l0_mag
        # save initial conditions as an attribute
        self.x0 = np.array([r0, p0, theta0, l0])

        return self
    
    def generate_ptcl_coordinates(self):

        # convert from (r, p, theta, l) to cartesian
        r, p, theta, l = self.x.T
        x_sep       = r * np.cos(theta)
        x_sep_dot   = p * np.cos(theta) - (l / r) * np.sin(theta)
        y_sep       = r * np.sin(theta)
        y_sep_dot   = p * np.sin(theta) + (l / r) * np.cos(theta)

        # define position vector and derivative
        r       = np.stack([x_sep, y_sep], axis=1)
        r_dot   = np.stack([x_sep_dot, y_sep_dot], axis=1)

        # convert to individual particle positions and trajectories assuming the center of mass
        # is located at the origin. save as attrs
        self.r1      = 0.5 * r
        self.r1_dot  = 0.5 * r_dot
        self.r2      = -0.5 * r
        self.r2_dot  = -0.5 * r_dot

        return self

    def simulate(self, duration, dt=0.001, x0=None):
        '''
        Description: Simulates according to Equations of Motion above and saves the data in the 
            attr 'x'. 'x' has the form as described in method '_EOM', i.e.
            x[0] = r (interptcl spacing), 
            x[1] = p (linear momentum), 
            x[2] = theta (angle wrt x-axis, or electric field direction), 
            x[3] = l (angular momentum).
            
            Also converts the result to cartesian x- and y-coordinates and saves in attribute 
            'x_cart'. Note that this is the separation vector r1 - r2. Output form:
            x_cart[0] = x
            x_cart[1] = x_dot
            x_cart[2] = y
            x_cart[3] = y_dot
        '''
        t_desired = np.arange(0, duration, dt)
        t_span = (t_desired[0], t_desired[-1])

        # generate initial conditions randomly if none entered
        if x0 is None:
            self._generate_init_cond()
        else:
            self.x0 = x0
        soln = solve_ivp(self._EOM, t_span, self.x0, t_eval=t_desired, events=self._too_close, **integrator_keywords)
        x_clean = soln.y.T
        t_actual = soln.t

        # save parameters as attributes
        self.duration = t_actual[-1]
        self.dt = dt

        # save data as attributes
        self.t = t_actual
        self.x = x_clean

        return self
    
    def plot(self):
        # generate individual particle positions
        self.generate_ptcl_coordinates()
        # plot trajectories and initial positions
        fig, axs = plt.subplots()
        # square field of view
        axs.set_aspect('equal', adjustable='datalim')
        # use blue and green colors
        colors = ['tab:blue', 'tab:green']
        
        # plot trajectories and starting positions
        axs.plot(*self.r1.T, label="particle 1", c=colors[0]) # transposed because matplotlib cycles through rows, not cols
        axs.plot(*self.r1[0], 'o', label="particle 1 start", c=colors[0])
        axs.plot(*self.r2.T, label="particle 2", c=colors[1])
        axs.plot(*self.r2[0], 'x', label="particle 2 start", c=colors[1])
        
        # initial velocity arrows
        stretch = 3e-1
        scaling = lambda x: stretch * np.sign(x) * np.log(np.abs(x) + 1)
        r1_0_dot_arrowlength = scaling(self.r1_dot[0])
        r2_0_dot_arrowlength = scaling(self.r2_dot[0])
        grayvalue = 0.4
        arrowprops=dict(arrowstyle="-|>",facecolor=f"{grayvalue}",edgecolor=f"{grayvalue}",linewidth=2, alpha=0.75)
        axs.annotate("", xytext = self.r1[0], xy = self.r1[0] + r1_0_dot_arrowlength, arrowprops=arrowprops)
        axs.annotate("", xytext = self.r2[0], xy = self.r2[0] + r2_0_dot_arrowlength, arrowprops=arrowprops)

        # labels
        axs.set_xlabel("$x$")
        axs.set_ylabel("$y$")
        axs.legend(loc="upper left")
        # duration label
        duration_str = f"duration = {self.duration:.3f}" + "$\omega_{pd}^{-1}$"
        axs.text(0.72, 0.92, duration_str, transform=axs.transAxes)

        fig.tight_layout()

        return fig, axs
        
        

    
def main():
    x0 = np.array([1, 0, np.pi, 0]) # format [r0, p0, theta0, l0]

    sim = Anisotropic_simulation()
    sim.simulate(0.4, x0=x0)
    sim.plot()
    plt.show()
        
if __name__ == '__main__':
    main()