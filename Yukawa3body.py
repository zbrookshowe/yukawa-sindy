'''
File:         Yukawa3body.py
Written by:   Brooks Howe
Last updated: 2025/06/17
Description:  Python program which has a class for simulating the 3-body Yukawa system of point 
    particles. Also includes fitting and plotting functionality
'''
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 6})
import pysindy as ps

import pickle as pkl
import dill
import time
import os

from scipy.integrate import solve_ivp
from numpy import random

integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12 # set relative tolerance
# use Livermore Solver for Ordinary Differential Equations with Automatic Stiffness Adjustment
integrator_keywords['method'] = 'LSODA' 
integrator_keywords['atol'] = 1e-12 # set absolute tolerance

import importlib
import Yukawa_SINDy
importlib.reload(Yukawa_SINDy) # imports Yukawa_SINDy separately everytime (?)

# create new Simulation object to simulate 3-body system
class Yukawa3body(Yukawa_SINDy.Simulation):
    '''
    Description: Simulation object for 3-body Yukawa systems
    Attributes:
        Simulation parameters
            init_cond: list of floats, initial conditions for 3 body simulation, default:
                [0, 1e-2, 0, 1e-2,   1, -1e-2, 0, 1e-2,   0, 1e-2, 1, -1e-2]
            labels: list of strings,labels for data, default:
                [(f"x{i}", f"vx{i}", f"y{i}", f"vy{i}") for i in range(3)].reshape((12,))
            potential_type: str, type of potential, default "repulsive", can also be "attractive"
        boolean flags:
            is_subtracted: bool, is data subtracted? default False
        data variables:
            x_unsubtracted: unsubtracted position and velocity data, default None
        other:
            rng: np.random.Generator, random number generator instance, default None. Creates a
                new random number generator with random seed num if None, saves seed num


    Methods (in the same order they are written in this class):
        __Yukawa_3body_EOM(self,t, x): private method, returns equations of motion for 3-body 
            system in first-order form.
        simulate(self, duration, dt=1e-4, potential_type:str="repulsive"):
            solves the system of equations for given parameters using solve_ivp from scipy.
            Data is stored in the attribute x from base class.
        generate_init_cond(self, std_dev=0.1, print=False):
            creates random initial conditions to use in the simulation, calls private method
            __calculate_vel_init_cond to calculate initial velocities. Prints initial conditions
            if print is True.
        __calculate_vel_init_cond(self, pos_init_cond):
            private method that calculates initial velocity vectors such that all point radially 
            inward towards the origin.
        plot(self, which:str='position'):
            plots position or velocity data
        subtract_data(self):
            Transforms data to subtracted space, saves original data in attribute x_unsubtracted
            and new, subtracted data in attribute x. See func desc. below for more details
        unsubtract_data(self):
            Restores original data to attribute x if already subtracted, raises error if data not
            subtracted.
        save_data(self, directoryname:str='data'):
            Saves data using python pickle. Naming scheme based on date and time, if files would
            have the same name, adds integer counter to filename to avoid overwrite.
    '''
    ###############################################################################################
    # Class Constructor
    ###############################################################################################

    def __init__(self, potential_type:str = "repulsive", rng:np.random.Generator = None):
        super().__init__() # inherit attributes from parent class
        self.init_cond = [0, 1e-2, 0, 1e-2,   1, -1e-2, 0, 1e-2,   0, 1e-2, 1, -1e-2] 
        # above default values are bad, need to pull new ones at some point
        self.labels = np.array([(f"x{i}", f"vx{i}", f"y{i}", f"vy{i}") for i in range(3)]
                               ).reshape((12,))
        self.is_subtracted = False
        self.x_unsubtracted = None

        # str var potential can be 'attractive' or 'repulsive'.
        if potential_type == 'attractive' or potential_type == 'repulsive':
            self.potential_type = potential_type
        else:
            raise ValueError("attribute 'potential_type' must be either str 'attractive' or 'repulsive'")
        # construct rng for class instance
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
    def init_cond(self):
        return self._init_cond
    @init_cond.setter
    def init_cond(self, init_cond):
        self._init_cond = init_cond

    @property
    def labels(self):
        return self._labels
    @labels.setter
    def labels(self, labels):
        self._labels = labels

    @property
    def x_unsubtracted(self):
        return self._x_unsubtracted
    @x_unsubtracted.setter
    def x_unsubtracted(self, x_unsubtracted):
        self._x_unsubtracted = x_unsubtracted

    @property
    def potential_type(self):
        return self._potential_type
    @potential_type.setter
    def potential_type(self, potential_type):
        self._potential_type = potential_type

    ###############################################################################################
    # Class Methods
    ###############################################################################################

    def __Yukawa_3body_EOM(self,t, x):
        '''
        Description: 
            this is the equations of motion for a 3-body system in 2 dimensions. The particle 
            indices in this case are 0, 1, and 2. The equations of motion are coded below in first
            order form, which means the 2nd order ODE is split into 2 first order ODEs. Therefore, 
            we have 12 first order ODEs. Half of these, though, are the trivial relation of 
            d/dt (xi) = (vxi), and the other half are the equations of motion of the form 
            d/dt (vxi) = f(x0,x1,x2). Let the position of the ith particle be given by (xi, yi) and
            the velocity by (vxi, vyi). The data is stored in the numpy array as follows:

            x[0] = x0, 
            x[1] = vx0, 
            x[2] = y0, 
            x[3] = vy0, 

            x[4] = x1, 
            x[5] = vx1, 
            x[6] = y1,  
            x[7] = vy1, 

            x[8] = x2, 
            x[9] = vx2, 
            x[10] = y2, 
            x[11] = vy2

            The equations are coded in the following way:

            (xi)' = vxi
            (vxi)' = sum_over_j_j_neq_i [ a(Δx_ij)exp(|r_vec_i - r_vec_j|) ( | r_vec_i - r_vec_j |^(-1) + | r_vec_i - r_vec_j |^(-3/2) ) ]
            (yi)' = vyi
            (vyi)' = sum_over_j_j_neq_i [ a(Δy_ij)exp(|r_vec_i - r_vec_j|) ( | r_vec_i - r_vec_j |^(-1) + | r_vec_i - r_vec_j |^(-3/2) ) ]

            for i = 0, 1, 2
        '''
        # str var potential can be 'attractive' or 'repulsive'.
        if self.potential_type == 'attractive':
            a = -1
        elif self.potential_type == 'repulsive':
            a = 1
        else:
            raise ValueError("attribute 'potential_type' must be either str 'attractive' or 'repulsive'")
        ## Equations of motion in first-order form
        # Particle 0 equations
        x0_dot  = x[1]
        vx0_dot = a*(x[ 0] - x[ 4]) * np.exp( -np.sqrt( (x[ 0] - x[ 4])**2 + (x[ 2] - x[ 6])**2 ) ) * ( ( (x[ 0] - x[ 4])**2 + (x[ 2] - x[ 6])**2 )**(-1) + ( (x[ 0] - x[ 4])**2 + (x[ 2] - x[ 6])**2 )**(-3/2) ) + \
                  a*(x[ 0] - x[ 8]) * np.exp( -np.sqrt( (x[ 0] - x[ 8])**2 + (x[ 2] - x[10])**2 ) ) * ( ( (x[ 0] - x[ 8])**2 + (x[ 2] - x[10])**2 )**(-1) + ( (x[ 0] - x[ 8])**2 + (x[ 2] - x[10])**2 )**(-3/2) )
        y0_dot  = x[3]
        vy0_dot = a*(x[ 2] - x[ 6]) * np.exp( -np.sqrt( (x[ 0] - x[ 4])**2 + (x[ 2] - x[ 6])**2 ) ) * ( ( (x[ 0] - x[ 4])**2 + (x[ 2] - x[ 6])**2 )**(-1) + ( (x[ 0] - x[ 4])**2 + (x[ 2] - x[ 6])**2 )**(-3/2) ) + \
                  a*(x[ 2] - x[10]) * np.exp( -np.sqrt( (x[ 0] - x[ 8])**2 + (x[ 2] - x[10])**2 ) ) * ( ( (x[ 0] - x[ 8])**2 + (x[ 2] - x[10])**2 )**(-1) + ( (x[ 0] - x[ 8])**2 + (x[ 2] - x[10])**2 )**(-3/2) )
        
        # Particle 1 equations
        x1_dot  = x[5]
        vx1_dot = a*(x[ 4] - x[ 8]) * np.exp( -np.sqrt( (x[ 4] - x[ 8])**2 + (x[ 6] - x[10])**2 ) ) * ( ( (x[ 4] - x[ 8])**2 + (x[ 6] - x[10])**2 )**(-1) + ( (x[ 4] - x[ 8])**2 + (x[ 6] - x[10])**2 )**(-3/2) ) + \
                  a*(x[ 4] - x[ 0]) * np.exp( -np.sqrt( (x[ 4] - x[ 0])**2 + (x[ 6] - x[ 2])**2 ) ) * ( ( (x[ 4] - x[ 0])**2 + (x[ 6] - x[ 2])**2 )**(-1) + ( (x[ 4] - x[ 0])**2 + (x[ 6] - x[ 2])**2 )**(-3/2) )
        y1_dot  = x[7]
        vy1_dot = a*(x[ 6] - x[10]) * np.exp( -np.sqrt( (x[ 4] - x[ 8])**2 + (x[ 6] - x[10])**2 ) ) * ( ( (x[ 4] - x[ 8])**2 + (x[ 6] - x[10])**2 )**(-1) + ( (x[ 4] - x[ 8])**2 + (x[ 6] - x[10])**2 )**(-3/2) ) + \
                  a*(x[ 6] - x[ 2]) * np.exp( -np.sqrt( (x[ 4] - x[ 0])**2 + (x[ 6] - x[ 2])**2 ) ) * ( ( (x[ 4] - x[ 0])**2 + (x[ 6] - x[ 2])**2 )**(-1) + ( (x[ 4] - x[ 0])**2 + (x[ 6] - x[ 2])**2 )**(-3/2) )
        
        # Particle 2 equations
        x2_dot  = x[9]
        vx2_dot = a*(x[ 8] - x[ 0]) * np.exp( -np.sqrt( (x[ 8] - x[ 0])**2 + (x[10] - x[ 2])**2 ) ) * ( ( (x[ 8] - x[ 0])**2 + (x[10] - x[ 2])**2 )**(-1) + ( (x[ 8] - x[ 0])**2 + (x[10] - x[ 2])**2 )**(-3/2) ) + \
                  a*(x[ 8] - x[ 4]) * np.exp( -np.sqrt( (x[ 8] - x[ 4])**2 + (x[10] - x[ 6])**2 ) ) * ( ( (x[ 8] - x[ 4])**2 + (x[10] - x[ 6])**2 )**(-1) + ( (x[ 8] - x[ 4])**2 + (x[10] - x[ 6])**2 )**(-3/2) )
        y2_dot  = x[11]
        vy2_dot = a*(x[10] - x[ 2]) * np.exp( -np.sqrt( (x[ 8] - x[ 0])**2 + (x[10] - x[ 2])**2 ) ) * ( ( (x[ 8] - x[ 0])**2 + (x[10] - x[ 2])**2 )**(-1) + ( (x[ 8] - x[ 0])**2 + (x[10] - x[ 2])**2 )**(-3/2) ) + \
                  a*(x[10] - x[ 6]) * np.exp( -np.sqrt( (x[ 8] - x[ 4])**2 + (x[10] - x[ 6])**2 ) ) * ( ( (x[ 8] - x[ 4])**2 + (x[10] - x[ 6])**2 )**(-1) + ( (x[ 8] - x[ 4])**2 + (x[10] - x[ 6])**2 )**(-3/2) )
        
        return [x0_dot, vx0_dot, y0_dot, vy0_dot, x1_dot, vx1_dot, y1_dot, vy1_dot, x2_dot, vx2_dot, y2_dot, vy2_dot]
        # return [## PARTICLE 1
        #         # x
        #         x[1],
        #         a*(x[ 0] - x[ 4]) * np.exp( -np.sqrt( (x[ 0] - x[ 4])**2 + (x[ 2] - x[ 6])**2 ) ) * ( ( (x[ 0] - x[ 4])**2 + (x[ 2] - x[ 6])**2 )**(-1) + ( (x[ 0] - x[ 4])**2 + (x[ 2] - x[ 6])**2 )**(-3/2) ) + # 12 interaction
        #         a*(x[ 0] - x[ 8]) * np.exp( -np.sqrt( (x[ 0] - x[ 8])**2 + (x[ 2] - x[10])**2 ) ) * ( ( (x[ 0] - x[ 8])**2 + (x[ 2] - x[10])**2 )**(-1) + ( (x[ 0] - x[ 8])**2 + (x[ 2] - x[10])**2 )**(-3/2) ), # 13 interaction
        #         # y
        #         x[3],
        #         a*(x[ 2] - x[ 6]) * np.exp( -np.sqrt( (x[ 0] - x[ 4])**2 + (x[ 2] - x[ 6])**2 ) ) * ( ( (x[ 0] - x[ 4])**2 + (x[ 2] - x[ 6])**2 )**(-1) + ( (x[ 0] - x[ 4])**2 + (x[ 2] - x[ 6])**2 )**(-3/2) ) + # 12 interaction
        #         a*(x[ 2] - x[10]) * np.exp( -np.sqrt( (x[ 0] - x[ 8])**2 + (x[ 2] - x[10])**2 ) ) * ( ( (x[ 0] - x[ 8])**2 + (x[ 2] - x[10])**2 )**(-1) + ( (x[ 0] - x[ 8])**2 + (x[ 2] - x[10])**2 )**(-3/2) ) , # 13 interaction
        #         ## PARTICLE 2
        #         # x
        #         x[5],
        #         a*(x[ 4] - x[ 8]) * np.exp( -np.sqrt( (x[ 4] - x[ 8])**2 + (x[ 6] - x[10])**2 ) ) * ( ( (x[ 4] - x[ 8])**2 + (x[ 6] - x[10])**2 )**(-1) + ( (x[ 4] - x[ 8])**2 + (x[ 6] - x[10])**2 )**(-3/2) ) + # 23 interaction
        #         a*(x[ 4] - x[ 0]) * np.exp( -np.sqrt( (x[ 4] - x[ 0])**2 + (x[ 6] - x[ 2])**2 ) ) * ( ( (x[ 4] - x[ 0])**2 + (x[ 6] - x[ 2])**2 )**(-1) + ( (x[ 4] - x[ 0])**2 + (x[ 6] - x[ 2])**2 )**(-3/2) ) , # 21 interaction
        #         # y
        #         x[7],
        #         a*(x[ 6] - x[10]) * np.exp( -np.sqrt( (x[ 4] - x[ 8])**2 + (x[ 6] - x[10])**2 ) ) * ( ( (x[ 4] - x[ 8])**2 + (x[ 6] - x[10])**2 )**(-1) + ( (x[ 4] - x[ 8])**2 + (x[ 6] - x[10])**2 )**(-3/2) ) + # 23 interaction
        #         a*(x[ 6] - x[ 2]) * np.exp( -np.sqrt( (x[ 4] - x[ 0])**2 + (x[ 6] - x[ 2])**2 ) ) * ( ( (x[ 4] - x[ 0])**2 + (x[ 6] - x[ 2])**2 )**(-1) + ( (x[ 4] - x[ 0])**2 + (x[ 6] - x[ 2])**2 )**(-3/2) ) , # 21 interaction
        #         ## PARTICLE 3
        #         # x
        #         x[9],
        #         a*(x[ 8] - x[ 0]) * np.exp( -np.sqrt( (x[ 8] - x[ 0])**2 + (x[10] - x[ 2])**2 ) ) * ( ( (x[ 8] - x[ 0])**2 + (x[10] - x[ 2])**2 )**(-1) + ( (x[ 8] - x[ 0])**2 + (x[10] - x[ 2])**2 )**(-3/2) ) + # 31 interaction
        #         a*(x[ 8] - x[ 4]) * np.exp( -np.sqrt( (x[ 8] - x[ 4])**2 + (x[10] - x[ 6])**2 ) ) * ( ( (x[ 8] - x[ 4])**2 + (x[10] - x[ 6])**2 )**(-1) + ( (x[ 8] - x[ 4])**2 + (x[10] - x[ 6])**2 )**(-3/2) ) , # 32 interaction
        #         # y
        #         x[11],
        #         a*(x[10] - x[ 2]) * np.exp( -np.sqrt( (x[ 8] - x[ 0])**2 + (x[10] - x[ 2])**2 ) ) * ( ( (x[ 8] - x[ 0])**2 + (x[10] - x[ 2])**2 )**(-1) + ( (x[ 8] - x[ 0])**2 + (x[10] - x[ 2])**2 )**(-3/2) ) + # 31 interaction
        #         a*(x[10] - x[ 6]) * np.exp( -np.sqrt( (x[ 8] - x[ 4])**2 + (x[10] - x[ 6])**2 ) ) * ( ( (x[ 8] - x[ 4])**2 + (x[10] - x[ 6])**2 )**(-1) + ( (x[ 8] - x[ 4])**2 + (x[10] - x[ 6])**2 )**(-3/2) )   # 32 interaction
        #         ]
    

    def simulate(self, duration, dt=1e-4, potential_type:str="repulsive"):
        '''
        Syntax: obj_instance.simulate(3)
        Description:
            Uses scipy.integrate.solve_ivp to simulate the system of equations given by private 
            method __Yukawa_3body_EOM.
        '''
        # if self.init_cond==[0, 1e-2, 0, 1e-2,   1, -1e-2, 0, 1e-2,   0, 1e-2, 1, -1e-2]:
        #     print("using default initial conditions")
        # above not working for some reason
        
        # create time grid
        t = np.arange(0,duration+dt,dt)
        t_span = (t[0], t[-1])

        # solve initial value problem using the Yukawa_3body_EOM function and return trajectories
        x_clean = solve_ivp(self.__Yukawa_3body_EOM, t_span, self.init_cond, t_eval=t, 
                            **integrator_keywords).y.T
        # save parameters as attributes
        
        self.duration = duration
        self.dt = dt
        # save data as attributes
        self.t=t
        self.x=x_clean


    def generate_init_cond(self, std_dev=0.1, print=False):
        '''
        Syntax: obj_instance.generate_init_cond()
        Description:
            Creates random initial conditions to use in the simulation. Note: if not run before
            method "simulate()", simulation will use default initial conditions. In the case of an 
            attractive potential, all the initial positions and velocities are drawn at random from
            a normal distribution centered at the origin with a standard deviation of std_dev. In 
            the case of a repulsive potential, the initial positions are drawn from a normal 
            distribution centered at the origin with a standard deviation of std_dev, and the 
            initial velocitiesare calculated from those positions such that they point radially 
            inward towards the origin.
        '''
        
        # Attractive case:
        # create random initial conditions from normal distribution centered at the origin with
        # standard deviation std_dev
        if self.potential_type == "attractive":
            init_cond = self.rng.normal(0.0, std_dev, (12,))
            # save as attribute
            self.init_cond = init_cond

        # Repulsive case:
        # create random initial positions from normal distribution as before, and calculate
        elif self.potential_type == "repulsive":
            pos_init_cond = self.rng.normal(0, std_dev, (6,))
            vel_init_cond = self.__calculate_vel_init_cond(pos_init_cond)
            init_cond_list = []
            # combine position and velocity initial conditions into one numpy array
            for i in range(len(pos_init_cond)):
                init_cond_list.append(pos_init_cond[i])
                init_cond_list.append(vel_init_cond[i])
            init_cond = np.array(init_cond_list)
            self.init_cond = init_cond
            
        if print:
            print("initial conditions are now set to:")
            [print(self.labels[i] + " = " + str(init_cond[i])) for i in range(len(init_cond))]
            print()


    def __calculate_vel_init_cond(self,pos_init_cond:np.ndarray):
        '''
        Description: Private method that calculates initial velocity vectors such that all point
            radially inward towards the origin.
        '''
        # updated code:
        # calculate angle from the positive x-axis in radians on the int [-pi,pi]
        angles = np.arctan2(pos_init_cond[1::2],pos_init_cond[0::2])
        # calculate initial velocity using angle
        vel_init_cond = []
        for i in range(angles.shape[0]):
            speed = 10 # self.rng.random() # should change to random once I get it working
            # append negative of calculated x- and y-components to list for init vels
            vel_init_cond.append(-speed*np.cos(angles[i]))
            vel_init_cond.append(-speed*np.sin(angles[i]))
        return vel_init_cond


    def plot(self, which:str='position'):
        '''
        Syntax: obj_instance.plot()
        Description:
            Plots the trajectories of all particles in the simulation along with dots where the
            initial positions are.
        '''
        if which == 'position':
            loop_start = 0
        elif which == 'velocity':
            loop_start = 1
        else:
            raise ValueError("str var which must be either 'position' or 'velocity'")
        # visualize trajectories
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        colors = ['C0','C1','C2']

        # plot x- and y-axes
        # ax.axhline(0, color='black', linewidth=.5)
        # ax.axvline(0, color='black', linewidth=.5)

        # plot trajectories
        for i in range( loop_start, self.x.shape[1], 4 ):
            label = f"particle {i//4}"
            # plot particle trajectories
            ax.plot(self.x[:,i], self.x[:,i+2], colors[i//4], label=label)
            # plot dots for ptcl init position
            ax.plot(self.init_cond[i],self.init_cond[i+2], colors[i//4] + 'o', label=label + " start")
        ax.legend()
        fig.tight_layout()
        plt.show()


    def subtract_data(self):
        '''
        Syntax: obj_instance.subtract_data()
        Description: transforms data to be in the subtracted space of positions and velocities such
        that:
        [[x0],              [[x0-x1],
         [vx0],              [vx0-vx1],
         [y0],               [y0-y1],
         [vy0],     ==>      [vy0-vy1],
         [x1],               [x1-x2],
         [vx1],              [vx1-vx2],
         ...,                ...,
         [vy2]]              [vy2-vy0]]
        Checks if data has already been subtracted or if no simulation has been run, raises error
        in either of these cases.
        '''
        # check if data is already subtracted
        if self.is_subtracted:
            raise Exception("data has already been transformed to be subtracted")
        if self.x is None:
            raise Exception("No simulation performed. Use .simulate() first.")
        # save unsubtracted data to attribute 'x_unsubtracted' before transforming
        self.x_unsubtracted = self.x

        # generate repeated list of indices
        idxs = np.tile(np.arange(0,12),2)
        # transform labels and data to be subtracted as explained above
        x_subtracted_labels = np.hstack([self.labels[i] + "-" 
                                         + self.labels[j] for i,j in zip(idxs[:12],idxs[4:])])
        x_subtracted = np.vstack([self.x[:,i]-self.x[:,j] for i,j in zip(idxs[:12],idxs[4:])]).T
        # save data as attributes
        self.x = x_subtracted
        self.labels = x_subtracted_labels
        # set attribute 'is_subtracted' to True
        self.is_subtracted = True
        return self
    
    def unsubtract_data(self):
        # restore original data if already subtracted, raise error if data not subtracted
        if self.is_subtracted:
            self.x = self.x_unsubtracted
            self.x_unsubtracted = None
            self.is_subtracted = False
        else:
            raise Exception("data has not been transformed to be subtracted")
        return self
    

    def save_data(self, directoryname:str='data'):
        '''
        Syntax: obj_instance.save_data()
        Description: create filename using date and time and save Simulation object to '.obj' file.
            Saves to directory specified in kwarg 'directoryname', creates directory if it doesn't
            exist, saves to 'data' by default. Uses a file naming scheme based on date and time, if
            files would have the same name, adds integer counter to filename to avoid overwrite,
            e.g. 'Yukawa3body_20220101_123456_1.obj'.
        '''
        # create directory if it doesn't exist
        if not os.path.exists(directoryname):
            os.makedirs(directoryname)
        # create filename using date and time
        timestr = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{directoryname}/Yukawa3body_{timestr}.obj"
        # check if filename already exists, add integer counter to filename to avoid overwrite
        counter = 0
        filename_to_check = filename
        while os.path.exists(filename_to_check):
            counter += 1
            filename_to_check = filename[:-4] + f"_{counter}.obj"
        # save data to file
        with open(filename_to_check, 'wb') as f:
            pkl.dump(self, f)


# END CLASS
###################################################################################################


###################################################################################################
# Functions
###################################################################################################

def multiple_simulate(duration=3e-1, dt=1e-4, n_trajectories:int=10,
                      potential_type:str='attractive', rng:np.random.Generator=None, 
                      save_data:bool=True, directoryname:str=None):
    '''
    Syntax: multiple_simulate()
    Description: simulate integer number 'n_trajectories' of equal duration 'duration' with
        timestep 'dt' for a potential type of 'potential_type'conditions, returns list of
        Yukawa3body objects. Can also input a random number generator object using kwarg 'rng' as a
        numpy.random.Generator object. If bool 'save_data' is True, saves data to .obj file in 
        folder 'data/YYMMDD_runs'.
    '''
    default_seed = 346734
    if rng is None:
        rng = np.random.default_rng(seed=default_seed)

    sim_list = []
    for i in range(n_trajectories):
        print("calculating trajectory", i)
        # randomize initial conditions and append to list
        sim = Yukawa3body(potential_type=potential_type, rng=rng)
        sim.generate_init_cond()
        sim.simulate(duration,dt=dt)
        sim_list.append(sim)
        # save data if desired
        if save_data:
            print("saving data for trajectory", i)
            if directoryname is None:
                # generate directoryname
                timestr = time.strftime("%Y%m%d")
                directoryname = f"data/{timestr}_runs"
            sim.save_data(directoryname=directoryname)
    return sim_list


def load_data(directory_of_pkls_only:str):
    # '''
    # Description: loads data from .obj files in directory given by relative path (project working
    #     directory is 'C:\Users\zacha\Box\Graduate School\Research\Code') in kwarg 
    #     'directory_of_pkls_only' and returns as a list of Yukawa3body objects.
    # '''
    sim_list = []
    for filename in os.listdir(directory_of_pkls_only):
        with open(f"{directory_of_pkls_only}/{filename}", 'rb') as f:
            sim = pkl.load(f)
            sim_list.append(sim)
    return sim_list


def plot_multiple(sim_list:list, num_plots:int=9, which:str='position', fontsize:int=12):
    '''
    Description: plots first 9 trajectories of x
    Inputs:
        sim_list: list of sim objects

    '''
    # desc: plots first 9 trajectories of x
    # identify which to plot using string var 'which'
    if which == 'position':
        loop_start = 0
    elif which == 'velocity':
        loop_start = 1
    else:
        raise ValueError("kwarg which which must be either 'position' or 'velocity'")
    
    # check if sim_list is a list of Yukawa3body objects
    # not working on saved data for some reason
    # if not all(isinstance(sim, Yukawa3body) for sim in sim_list):
    #     raise TypeError("sim_list must be a list of Yukawa3body objects")
    
    # plot first 'num_plots' trajectories of x
    plt.rcParams['font.size'] = str(fontsize)
    num_columns = 3
    num_rows = int(np.ceil(num_plots/num_columns))
    figsize = (12,4*num_rows)
    fig, axs = plt.subplots(num_rows, num_columns, sharex=True, sharey=True, figsize=figsize)
    colors = ['C0','C1','C2']
    axs.resize((num_rows*num_columns,))
    for i in range(num_plots):
        for j in range( loop_start, sim_list[0].x.shape[1], 4 ):
            box=[-0.4,0.4]
            axs[i].set_xlim(box)
            axs[i].set_ylim(box)
            label = f"particle {j//4}"
            # plot particle trajectories
            axs[i].plot(sim_list[i].x[:,j], sim_list[i].x[:,j+2], colors[j//4], label=label)
            # plot dots for ptcl init position
            if sim_list[i].is_subtracted:
                indices = np.tile(np.arange(sim_list[i].x.shape[1]),2)
                init_xdiff = sim_list[i].init_cond[indices[j  ]] - sim_list[i].init_cond[indices[j+4]]
                init_ydiff = sim_list[i].init_cond[indices[j+2]] - sim_list[i].init_cond[indices[j+6]]
                axs[i].plot(init_xdiff, init_ydiff, colors[j//4] + 'o')
            else:
                init_x = sim_list[i].init_cond[j]
                init_y = sim_list[i].init_cond[j+2]
                axs[i].plot(init_x, init_y, colors[j//4] + 'o')#, label=label + " start")
            # plot arrows for ptcl init velocity
            # axs[i].arrow(sim_list[i].init_cond[j],sim_list[i].init_cond[j+2],5e-2*sim_list[i].init_cond[j+1],5e-2*sim_list[i].init_cond[j+3], color=colors[j//4], width=0.0005)#, label=label + " init. vel.")
    axs[0].legend(loc='upper left',prop={'size': fontsize})
    fig.tight_layout()
    return fig, axs

def generate_3body_library(use_weak:bool=False, spatiotemporal_grid=None, K=100):
    # define custom library of terms with only yukawa (rational) terms
    """
    Description: generates a custom library of terms with only yukawa (rational) terms
        for 3-body simulations. The yukawa library is then combined with an identity library
        to create a generalized library, which is used for fitting the 3-body SINDy model. This 
        library contains only terms that are necessary to describe the equations of motion. Uses
        weak form if kwarg 'use_weak' is True, uses strong form by default. Note: var 
        'spatiotemporal_grid' must be provided if 'use_weak' is True.

    Returns:
        A generalized library used for 3-body SINDy model fitting
    """
    library_functions = [
        lambda x, y: x * np.exp( -np.sqrt( x**2 + y**2 ) ) / ( x**2 + y**2 ),
        lambda x, y: y * np.exp( -np.sqrt( x**2 + y**2 ) ) / ( x**2 + y**2 ),
        lambda x, y: x * np.exp( -np.sqrt( x**2 + y**2 ) ) / ( x**2 + y**2 )**(3/2),
        lambda x, y: y * np.exp( -np.sqrt( x**2 + y**2 ) ) / ( x**2 + y**2 )**(3/2)
    ]
    library_function_names = [
        lambda x,y: "(" + x + ") exp( -sqrt((" + x + ")^2+(" + y + ")^2) ) / ((" + x + ")^2+(" + y + ")^2)",
        lambda x,y: "(" + y + ") exp( -sqrt((" + x + ")^2+(" + y + ")^2) ) / ((" + x + ")^2+(" + y + ")^2)",
        lambda x,y: "(" + x + ") exp( -sqrt((" + x + ")^2+(" + y + ")^2) ) / ((" + x + ")^2+(" + y + ")^2)^(3/2)",
        lambda x,y: "(" + y + ") exp( -sqrt((" + x + ")^2+(" + y + ")^2) ) / ((" + x + ")^2+(" + y + ")^2)^(3/2)"
    ]

    # generate custom library using weak formulation if desired, else use regular custom and identity libraries (strong form)
    if use_weak:
        if spatiotemporal_grid is None:
            raise ValueError("spatiotemporal_grid must be provided if using weak form")
        yukawa_library = ps.WeakPDELibrary(
            library_functions=library_functions, 
            function_names=library_function_names,
            spatiotemporal_grid=spatiotemporal_grid,
            is_uniform=True,
            K=K        )

        identity_library = ps.WeakPDELibrary(
            library_functions=[lambda x: x],
            function_names=[lambda x: x],
            spatiotemporal_grid=spatiotemporal_grid,
            is_uniform=True,
            K=K
        )
    else:
        yukawa_library = ps.CustomLibrary(
            library_functions=library_functions, 
            function_names=library_function_names
        )

        # create identity library for the definition terms x' = v, etc.
        identity_library = ps.IdentityLibrary()

    # input only velocities to first library and only positions to other three libraries
    num_features:int = 12 # x_train.shape[1] # need to change this later to be general
    pos_idxs = [i for i in range(0,num_features,2)]
    vel_idxs = [i+1 for i in range(0,num_features,2)]
    inputs_per_library = np.array([vel_idxs,pos_idxs[0:2]*3,pos_idxs[2:4]*3,pos_idxs[4:6]*3])
    generalized_library = ps.GeneralizedLibrary(
        [identity_library] + 3*[yukawa_library],
        inputs_per_library=inputs_per_library
    )
    return generalized_library

def generate_3body_library_codified():
    # define custom library of terms with only yukawa (rational) terms
    """
    Description: generates a custom library of terms with only yukawa (rational) terms
        for 3-body simulations, grouping terms by cartesian index. The yukawa library is then
        combined with an identity library to create a generalized library, which is used for 
        fitting the 3-body SINDy model. This library combines the terms used in the above func
        'generate_3body_library' so that there are less terms in the library. Created for testing
        purposes.

    Returns:
        A generalized library used for 3-body SINDy model fitting
    """
    library_functions = [
        lambda x, y: x * np.exp( -np.sqrt( x**2 + y**2 ) ) / ( x**2 + y**2 ) + x * np.exp( -np.sqrt( x**2 + y**2 ) ) / ( x**2 + y**2 )**(3/2),
        lambda x, y: y * np.exp( -np.sqrt( x**2 + y**2 ) ) / ( x**2 + y**2 ) + y * np.exp( -np.sqrt( x**2 + y**2 ) ) / ( x**2 + y**2 )**(3/2)
    ]
    library_function_names = [
        lambda x,y: x + "w(" + x + ", " + y + ")",
        lambda x,y: y + "w(" + x + ", " + y + ")"
    ]
    yukawa_library = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=library_function_names
    )

    # create identity library for the definition terms x' = v, etc.
    identity_library = ps.IdentityLibrary()

    # input only velocities to first library and only positions to second library
    num_features:int = 12 # x_train.shape[1] # need to change this later to be general
    pos_idxs = [i for i in range(0,num_features,2)]
    vel_idxs = [i+1 for i in range(0,num_features,2)]
    inputs_per_library = np.array([vel_idxs,pos_idxs[0:2]*3,pos_idxs[2:4]*3,pos_idxs[4:6]*3])
    generalized_library = ps.GeneralizedLibrary(
        [identity_library] + 3*[yukawa_library],
        inputs_per_library=inputs_per_library
    )
    return generalized_library

def print_SINDy_nice_OLD(model: ps.SINDy, noise=None, save=False):
    '''
    Syntax: print_SINDy_nice(model)
    Description: prints SINDy model in readable format, where each equation is printed with each 
        term on a separate line.
    '''
    # print header
    print(100*'-')
    print('STLSQ threshold:', model.optimizer.threshold)
    print('complexity:', model.complexity)
    print(100*'-')
    # get equations and feature names from model object
    eqns = model.equations()
    feature_names = model.feature_names
    # print equations formatted nicely
    for i, eqn in enumerate(eqns):
        terms = eqn.split(' + ')
        padding = (11 - len(feature_names[i]))*' '
        print('(' + feature_names[i] + ')\'' + padding + '= ' + terms[0])
        if len(terms) > 1:
            for term in terms[1:]:
                print(14*' ' + '+ ' + term)
        print()
        if (i+1) % 4 == 0:
            print(100*'-')
    
    if save:
        with open('SINDy_model.txt', 'w') as f:
            f.write(model.equations())

def print_SINDy_nice(model: ps.SINDy, sim_list: list):
    '''
    Syntax: print_SINDy_nice(model)
    Description: prints SINDy model in readable format, where each equation is printed with each 
        term on a separate line.
    '''
    output = SINDy_results_nice(model=model, sim_list=sim_list)
    print('\n'.join(output))


def SINDy_results_nice(model: ps.SINDy, sim_list: list):
    n_trajectories = len(sim_list)
    noise_level = sim_list[0].noise_level
    # save eqns, feature_names, noise_level, n_trajectories, complexity
    output = []
    # Header
    output.append(100*'=')
    output.append('noise level: ' + str(noise_level))
    output.append('number of trajectories: '+ str(n_trajectories))
    output.append('STLSQ threshold: '+ str(model.optimizer.threshold))
    output.append('complexity: ' + str(model.complexity))
    output.append(100*'=')
    # SINDy model
    for i, eqn in enumerate(model.equations()):
        terms = eqn.split(' + ')
        padding = (11 - len(model.feature_names[i]))*' '
        output.append('(' + model.feature_names[i] + ')\'' + padding + '= ' + terms[0])
        if len(terms) > 1:
            for term in terms[1:]:
                output.append(14*' ' + '+ ' + term)
        output.append('')
        if (i+1) % 4 == 0:
            output.append('\n' + 100*'-')

    return output

def save_SINDy_model(model:ps.SINDy, sim_list: list,
                     directoryname: str='data/basic_noisy/SINDy_results'
                     ):
    '''
    Description: saves SINDy model as human readable .txt files and as .obj files for future use.
        Creates directory for specific noise level ), saves .txt files with SINDy
        model there. Creates a subdirectory for .obj files (if it doesn't exist), saves .obj files
        there.
    '''
    # extract noise level and threshold from inputs, convert to strings of uniform format
    noise_level = sim_list[0].noise_level
    threshold = model.optimizer.threshold
    noise_rounded = np.format_float_positional(noise_level, unique=False, precision=5)
    threshold_rounded = np.format_float_positional(threshold, unique=False, precision= 3)

    # create directory for models with a specific noise level if it doesn't exist
    noise_directoryname = directoryname + '/noise_' + noise_rounded
    if not os.path.exists(noise_directoryname):
        os.makedirs(noise_directoryname)

    # save .txt file
    filename = f'threshold_{threshold_rounded}_results.txt'
    filepath = noise_directoryname + '/' + filename
    with open(filepath, 'w') as f:
        f.writelines(line + '\n' for line in SINDy_results_nice(model, sim_list))

    # create directory for .obj files specifically if it doesn't exist
    obj_directoryname = noise_directoryname + '/model_objs'
    if not os.path.exists(obj_directoryname):
        os.makedirs(obj_directoryname)
    
    # save .obj file in subdirectory of noise level directory
    filename = f'threshold_{threshold_rounded}.obj'
    filepath = obj_directoryname + '/' + filename
    with open(filepath, 'wb') as f:
        dill.dump(model, f)


def load_SINDy_models(directory_of_dillpkls_only:str):
    model_list = []
    for filename in os.listdir(directory_of_dillpkls_only):
        with open(f"{directory_of_dillpkls_only}/{filename}", 'rb') as f:
            model = dill.load(f)
            model_list.append(model)
    return model_list


def main():
    # initialize variables
    noise_level = 0
    potential_type = 'repulsive'
    save_data = False
    directoryname = 'data/basic_noisy'
    # define rng for reproducibility
    rng = np.random.default_rng(seed=346734)
    sim_list = multiple_simulate(duration=1e-1,n_trajectories=10,potential_type=potential_type,
                                  rng=rng, save_data=save_data, directoryname=directoryname
                                  )
    # plot_multiple(sim_list=sim_list)
    generalized_library = generate_3body_library()
    opt = ps.STLSQ(threshold=0.4)
    # loop through sim_list to transform data and build list x_train_subtracted and extract out 
    # labels
    x_train_subtracted = []
    for i, sim in enumerate(sim_list):
        sim.subtract_data()
        if noise_level != 0:
            sim.add_gaussian_noise(noise_level=noise_level)
        x_train_subtracted.append(sim.x)
        sim_list[i] = sim # modify list element in place
    x_train_labels = sim.labels # extract labels from last sim in loop
    dt = sim.dt # extract dt from last sim in loop

    # fit a SINDy model
    model = ps.SINDy(optimizer=opt, feature_names=x_train_labels, 
                     feature_library=generalized_library
                     )
    model.fit(x_train_subtracted, t=dt, multiple_trajectories=True)
    # model.print()
    result = SINDy_results_nice(model, sim_list=sim_list)
    for line in result:
        print(line)

    model_filename = f'model_noise_{noise_level}.txt'
    model_directory = 'SINDy_results'
    model_path = directoryname + '/' + model_directory + '/' + model_filename

    with open(model_path, 'wb') as f:
        f.writelines(result)
    for term in model.get_feature_names():
        print(term)

if __name__ == "__main__":
    print("running main function")
    main()