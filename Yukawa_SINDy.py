'''
File:         Yukawa_SINDy.py
Written by:   Brooks Howe
Last updated: 2025/06/02
Description:  Python script containing functions used in
              the file 'YukawaEOM_basic.ipynb'.
'''

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.integrate import solve_ivp
import pysindy as ps
from pysindy.differentiation import FiniteDifference
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Plotting parameters
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'figure.figsize': (8,6)})

# create 'integrator_keywords' dict for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12 # set relative tolerance
integrator_keywords['method'] = 'LSODA' # Livermore Solver for Ordinary Differential Equations with Automatic Stiffness Adjustment
integrator_keywords['atol'] = 1e-12 # set absolute tolerance

# import scaling constant from working directory and declare as global variable
from pickle import load
with open('scaling_const.float','rb') as f:
    A = load(f)

class Simulation:
    '''
    Description: Base class for simulation objects.
    Attributes:
        Simulation parameters
            duration: duration of simulation, positional arg
            dt: timestep, default 0.001, kwarg
        boolean flags:
            is_noisy: bool, is data noisy? default False
            is_subsampled: bool, is data subsampled? default False
        noise_level: standard deviation of gaussian noise added to data,
            default 0 (implying no noise)
        sample_frac: fraction of data which was subsampled,
            default 1 (implying no subsampling)
        data variables:
            t: time data, default None
            t_full: full time data (if subsampled), default None
            x: position, velocity data, default None
            x_clean: clean data (if noise added), default None
            x_fullnoisydata: full noisy data (if subsampled after 
                noise added), default None


    Methods (in the same order they are written in this class):
        add_gaussian_noise(noise_level=0.01)
            adds gaussian noise to the data with standard deviation noise_level. Does not add
            if already noisy. Moves clean data to attribute x_clean, and stores noisy data in 
            attribute x.
        delete_noise()
            deletes the noise added to the data and restores the clean data to the attribute x.
        subsample(fraction=0.1)
            subsamples the data to a fraction of the original data. If data is clean, moves to
            attribute x_clean. If data is noisy, moves to attribute x_fullnoisydata. Subsampled
            data is saved in attribute x.
        restore_data()
            restores the full, unsubsampled data to the attribute x.
        plot()
            plots the data. If clean, plots only the clean data; if noisy, plots both the clean
            and noisy data.
    '''
    ###############################################################################################
    # Class Constructor
    ###############################################################################################

    def __init__(self):
        self.duration = None
        self.dt = None
        self.is_noisy = False
        self.noise_level = 0
        self.is_subsampled = False
        self.sample_frac = 1

        # is this needed? maybe good for storage?
        self.t=None
        self.t_full = None
        self.x=None
        self.x_clean=None
        self.x_fullnoisydata=None

    ###############################################################################################
    # Getters and setters
    ###############################################################################################

    @property
    def duration(self):
        # print("duration getter called") # for testing
        return self._duration
    @duration.setter
    def duration(self, duration):
        # print("duration setter called") # for testing
        # if duration >= 10:
        #     raise ValueError("duration must be less than 10")
        self._duration = duration
    
    @property
    def dt(self):
        return self._dt
    @dt.setter
    def dt(self, dt):
        self._dt = dt

    # Not working for some reason:
    # @property
    # def is_noisy(self):
    #     return self._is_noisy
    
    # @is_noisy.setter
    # def noisy(self, is_noisy):
    #     self._is_noisy = is_noisy

    @property
    def noise_level(self):
        return self._noise_level
    @noise_level.setter
    def noise_level(self, noise_level):
        self._noise_level = noise_level

    @property
    def sample_frac(self):
        return self._sample_frac
    @sample_frac.setter
    def sample_frac(self, sample_frac):
        self._sample_frac = sample_frac
        
    @property
    def t(self):
        return self._t
    @t.setter
    def t(self, t):
        self._t = t

    @property
    def t_full(self):
        if self._t_full is None:
            print("t is not subsampled.")
        return self._t_full
    @t_full.setter
    def t_full(self, t_full):
        self._t_full = t_full

    @property
    def x(self):
        if self._x is None:
            print("No simulation performed. Use simulate() first.")
            return
        return self._x
    @x.setter
    def x(self, x):
        self._x = x
    
    @property
    def x_clean(self):
        if self._x_clean is None:
            print("No noise added to x, x is clean.")
        return self._x_clean
    @x_clean.setter
    def x_clean(self, x_clean):
        self._x_clean = x_clean

    @property
    def x_fullnoisydata(self):
        if self._x_fullnoisydata is None:
            print("x is not subsampled.")
        return self._x_fullnoisydata
    @x_fullnoisydata.setter
    def x_fullnoisydata(self, x_fullnoisydata):
        self._x_fullnoisydata = x_fullnoisydata

    ###############################################################################################
    # Class Methods
    ###############################################################################################

    def add_gaussian_noise(self, noise_level=0.01):
        if self.x is None:
            raise Exception("No simulation performed. Use .simulate() first.")
        if self.is_noisy:
            raise Exception("Data is already noisy, no new noise added.")
        if self.is_subsampled:
            raise Exception("Cannot add noise to subsampled data, " \
                            + "create new sim object to add noise.")
        # do nothing if noise_level is zero
        if noise_level == 0:
            return self
        # Adds noise to data
        # generate noise
        dims = np.shape(self.x)
        rng = np.random.default_rng(seed=2673) # add seed number for reproducibility
        noise =  rng.normal(loc=0,scale=noise_level,size=dims)
        '''code for normalizing noise to avg dispacement/timestep
        # generate noise for position
        std_dev = (self.x[:,0].max() - self.x[:,0].min()) / self.x.shape[0] * noise_level
        x_noise =  rng.normal(loc=0,scale=std_dev,size=dims[0])
        # generate noise for velocity
        std_dev = (self.x[:,1].max() - self.x[:,1].min()) / self.x.shape[0] * noise_level
        v_noise =  rng.normal(loc=0,scale=std_dev,size=dims[0])
        noise =  np.column_stack((x_noise, v_noise))
        '''
        # add noise to data
        self.x_clean = self.x
        self.x = self.x + noise
        self.is_noisy = True
        self.noise_level = noise_level
        return self
    
    def delete_noise(self):
        if self.is_noisy:
            self.x = self.x_clean
            self.x_clean = None
            self.is_noisy = False
            self.noise_level = 0
        else:
            raise Exception("Data is already clean, no noise removed.")
        return self
    
    def subsample(self, sample_frac=0.1):
        if self.x is None:
            raise Exception("No simulation performed. Use simulate() first.")
        if self.is_subsampled:
            raise Exception("Data is already subsampled, no new subsampling performed.")
        # Subsample data
        idx = np.arange(self.x.shape[0])
        rng = np.random.default_rng(seed=2673) # add seed number for reproducibility
        ridx = rng.choice(idx, size = (int(idx.shape[0]*sample_frac)), replace=False, shuffle=False)
        ridx.sort()
        x_subsampled = self.x[ridx]
        t_subsampled = self.t[ridx]
        # save data as attributes
        if self.is_noisy:
            self.x_fullnoisydata = self.x
        else:
            self.x_clean = self.x
        self.x = x_subsampled
        self.t_full = self.t
        self.t = t_subsampled
        self.is_subsampled = True
        self.sample_frac = sample_frac
        return self
    
    def restore_data(self):
        if self.is_subsampled:
            if self.is_noisy:
                self.x = self.x_fullnoisydata
                self.x_fullnoisydata = None
            else:
                self.x = self.x_clean
                self.x_clean = None
            self.is_subsampled = False
            self.sample_frac = 1
            self.t = self.t_full
            self.t_full = None
        else:
            raise Exception("Data was not subsampled.")
        return self
###############################################################################################
# END OF Simulation CLASS
###############################################################################################


class Yukawa_simulation(Simulation):
    '''
    Description: Class for simulating the Yukawa equation of motion for a two-body system.
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
        self.x0 = None
        self.v0 = None
        self.is_scaled=None
        self.random_a_mu = None
        self.random_a_sigma = None

    @property
    def x0(self):
        return self._x0
    @x0.setter
    def x0(self, x0):
        self._x0 = x0
    
    @property
    def v0(self):
        return self._v0
    @v0.setter
    def v0(self, v0):
        self._v0 = v0

    @property
    def is_scaled(self):
        return self._is_scaled
    @is_scaled.setter
    def is_scaled(self, is_scaled:bool):
        self._is_scaled = is_scaled

    @property
    def random_a_mu(self):
        return self._random_a_mu
    @random_a_mu.setter
    def random_a_mu(self, random_a_mu:float):
        self._random_a_mu = random_a_mu

    @property
    def random_a_sigma(self):
        return self._random_a_sigma
    @random_a_sigma.setter
    def random_a_sigma(self, random_a_sigma:float):
        self._random_a_sigma = random_a_sigma

    ###############################################################################################
    # Class Methods
    ###############################################################################################
    def __Yukawa_EOM(self, t, x): 
        global A
        return [x[1], A * ( 1/x[0] + 1/x[0]**2 ) * np.exp( -x[0] ) ]

    def __Yukawa_EOM_stochastic(self, t, x):
        global A
        rng = np.random.default_rng()
        random_a = rng.normal(self.random_a_mu, self.random_a_sigma)
        return [x[1], A * ( 1/x[0] + 1/x[0]**2 ) * np.exp( -x[0] ) + random_a ]

    def __Yukawa_EOM_unscaled(self, t, x): 
       return [x[1], ( 1/x[0] + 1/x[0]**2 ) * np.exp( -x[0] ) ]
    
    def __harmonic_oscilator(self, t, x):
        # included for testing purposes
        return[x[1],-x[0]]

    def simulate(self, duration, dt=0.001, x0=1.0, v0=0.01, scaled=False):
        # syntax: simulate(3, dt=0.001, x0=1, v0=0.01)
        # Generate measurement data
        t = np.arange(0, duration, dt)
        t_span = (t[0], t[-1])

        if scaled:
            func = self.__Yukawa_EOM
            self.is_scaled = True
        else:
            func = self.__Yukawa_EOM_unscaled
            self.is_scaled = False

        x0_train = [x0, v0]
        x_clean = solve_ivp(func, t_span, x0_train, t_eval=t, **integrator_keywords).y.T
        # save parameters as attributes
        self.duration = duration
        self.dt = dt
        self.x0 = x0
        self.v0 = v0
        # save data as attributes
        self.t=t
        self.x=x_clean
        return self
    
    def plot(self):
        if self.x is None:
            raise Exception("No simulation performed. Use simulate() first.")
        labels = np.array([["noisy position", "clean position"], 
                           ["noisy velocity", "clean velocity"]]
                           )
        plt.xlabel("time (s)")
        for i in range(self.x.shape[1]):
            if self.is_noisy:
                plt.plot(self.t, self.x[:,i], label=labels[i,0])
                if self.is_subsampled:
                    plt.plot(self.t_full, self.x_clean[:,i], label=labels[i,1])
                    plt.title(f"Randomly sampled {self.sample_frac*100}% of Noisy Data")
                else:
                    plt.plot(self.t, self.x_clean[:,i], label=labels[i,1])
                    plt.title("Noisy Data")
            else:
                plt.plot(self.t, self.x[:,i], label=labels[i,1])
                if self.is_subsampled:
                    plt.title(f"Randomly sampled {self.sample_frac*100}% of Clean Data")
                else:
                    plt.title("Clean Data")
        plt.legend()

###############################################################################################
# END OF Yukawa_simulation CLASS
###############################################################################################


###############################################################################################
# SINDy functions
###############################################################################################
def set_optimizer(opt_str: str, hparam: float):
# Syntax: set_optimizer("stlsq", 0.1)
# Description: Sets the optimizer for SINDy model based on the string "opt_str"
    if opt_str=="stlsq":
        opt = ps.STLSQ(threshold=hparam,alpha=0.05,verbose=False)
        hparam_str = "threshold"
    elif opt_str=='sr3':
        opt = ps.SR3(threshold=hparam, thresholder="l1")
        hparam_str = "threshold"
    elif opt_str=='lasso':
        opt = Lasso(alpha=hparam, max_iter=20000, fit_intercept=False)
        hparam_str = "alpha"
    else:
        raise TypeError("optimizer not recognized")
    return opt, hparam_str


def generate_Yukawa_library():
    # Syntax: generate_Yukawa_library()
    # Description: Generates a library of custom functions
    # that can be used for SINDy analysis
    # Note: DEPRECATED

    # Create library of coefs
    library_functions = [
        # lambda x: 1.0, get rid of this term because it is being duplicated and causing an error
        lambda x: x,
        lambda x: np.exp(-x) / x,
        lambda x: np.exp(-x) / x**2,
        lambda x: np.exp(-x) / x**3,
        lambda x: np.exp(-x) / x**4,
    ]
    library_function_names = [
        # lambda x: 1,
        lambda x: x,
        lambda x: "exp(-" + x + ") / " + x,
        lambda x: "exp(-" + x + ") / " + x + "^2",
        lambda x: "exp(-" + x + ") / " + x + "^3",
        lambda x: "exp(-" + x + ") / " + x + "^4",
    ]
    custom_library = ps.CustomLibrary(
        library_functions=library_functions, function_names=library_function_names
    )
    return custom_library

def generate_libraries(t_for_weak):
    # define lambdas for library functions
    library_functions = [
        # lambda x: 1.0, get rid of this term because it is being duplicated and causing an error
        lambda x: x,
        lambda x: np.exp(-x) / x,
        lambda x: np.exp(-x) / x**2,
        lambda x: np.exp(-x) / x**3,
        lambda x: np.exp(-x) / x**4,
    ]

    # define names for library functions
    library_function_names = [
        # lambda x: 1,
        lambda x: x,
        lambda x: "exp(-" + x + ") / " + x,
        lambda x: "exp(-" + x + ") / " + x + "^2",
        lambda x: "exp(-" + x + ") / " + x + "^3",
        lambda x: "exp(-" + x + ") / " + x + "^4",
    ]

    # generate weak form library
    np.random.seed(120398)
    # Note: WSINDy uses a random selection of integration subdomains, so
    # we fix a seed number so we use the same set of subdomains each time
    # this function is called.
    weak_lib = ps.WeakPDELibrary(
        library_functions=library_functions,
        spatiotemporal_grid=t_for_weak,
        function_names=library_function_names)

    # generate strong form library
    strong_lib = ps.CustomLibrary(
        library_functions=library_functions, 
        function_names=library_function_names)
    
    return weak_lib, strong_lib


def fit_Yukawa_model(sim_obj: Yukawa_simulation,opt_str: str='stlsq', hparam: float=0.1, 
                     return_hparam_str: bool=False):
    # Syntax: fit_Yukawa_model(sim1, 'stlsq', 0.1)
    # Description: Fits SINDy model for the Yukawa equation
    # with given optimizer and hyperparameter. Optionally 
    # returns the optimizer used as a string

    # set optimizer based on input; library, feature names set for Yukawa
    custom_library = generate_Yukawa_library()
    feature_names = ["x", "v"] # feature names to position (x) and velocity (v)
    opt, hparam_str = set_optimizer(opt_str, hparam)
    
    # fit SINDy model
    model = ps.SINDy(feature_names=feature_names, optimizer=opt, feature_library=custom_library)
    model.fit(sim_obj.x, t=sim_obj.t)
    if return_hparam_str:
        return model, hparam_str
    else:
        return model

###############################################################################################
# Plotting functions
###############################################################################################
def plot_derivatives(sim_obj: Yukawa_simulation, model: ps.SINDy):
    # Syntax: plot_derivatives(sim1, model)
    # Description: Plots the predicted and computed
    # derivatives of the SINDy model

    # Predict derivatives using the learned model
    x_dot_test_predicted = model.predict(sim_obj.x)

    # Compute derivatives with a finite difference method, for comparison
    x_dot_test_computed = model.differentiate(sim_obj.x, t=sim_obj.t)

    fig, axs = plt.subplots(sim_obj.x.shape[1], 1, sharex=True, figsize=(7, 9))
    for i in range(sim_obj.x.shape[1]):
        axs[i].plot(sim_obj.t, x_dot_test_computed[:, i], "k", label="numerical derivative")
        axs[i].plot(sim_obj.t, x_dot_test_predicted[:, i], "r--", label="model prediction")
        axs[i].legend()
        axs[i].set(xlabel="t", ylabel=r"$\dot x_{}$".format(i))
    fig.show()


def plot_coefs(model: ps.SINDy, hparam=None, std_dev=None, figsize=(12,5)):
# Syntax: plot_coefs(model)
# Description: Plots bar chart of coefficient values for each feature in the model.
    
    title_str = 'Coefficient Values'
    # get coefficents and feature names
    coefs = model.coefficients().T
    coef_names = model.get_feature_names()
    # create bar plot of coefficient values
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)#, sharey=True)
    for i in range(coefs.shape[1]):
        axs[i].bar(coef_names, coefs[:,i])
        # axs[i].tick_params(axis='x', labelsize=14) # change label size
        # axs[i].tick_params(axis='y', labelsize=14)
        axs[i].axhline(y=0, lw=0.75, color='k')
        axs[i].set_title('(' + model.feature_names[i] + ')\' equation')
    if hparam is not None:
        title_str += "\nHyperparameter = " + str(hparam)
        # fig.suptitle('Coefficient Values\n' + 'hparam = ' + str(hparam), fontsize=16)
    if std_dev is not None:
        title_str += ", Standard deviation of noise= " + str(std_dev)

    fig.suptitle(title_str, fontsize=16)
    fig.tight_layout(pad=2.0)


def plot_coef_hist(hspace, 
                   coefs_array, 
                   feature_names=[
                                    r'$r$',
                                    r'$v$',
                                    r'$\frac{e^{-r}}{r}$',
                                    r'$\frac{e^{-v}}{v}$',
                                    r'$\frac{e^{-r}}{r^2}$',
                                    r'$\frac{e^{-v}}{v^2}$',
                                    r'$\frac{e^{-r}}{r^3}$',
                                    r'$\frac{e^{-v}}{v^3}$',
                                    r'$\frac{e^{-r}}{r^4}$',
                                    r'$\frac{e^{-v}}{v^4}$'
                                    ],
                   figsize=(15,12)):
    
    # create lists for column and row titles
    col_labels = [r'$\dot{r}$ Equation', r'$\dot{v}$ Equation']
    row_labels = feature_names
    # create array of labels for subplots
    lets = np.array(list(map(chr, range(97, 123)))).reshape((13,2))
    periods = np.full_like(lets, '.')
    labels = np.char.add(lets, periods)
    # set axis limits
    ymin = 0
    ymax = 2
    xmin = min(hspace) - 0.05
    if xmin <= 0:
        xmin = 0
    xmax = max(hspace)
    # plot coefficients as threshold changes, label subplots
    fig, axs = plt.subplots(coefs_array.shape[1], coefs_array.shape[2],
                            figsize=figsize, sharex=True)#, sharey=True)
    for j in range(coefs_array.shape[2]):
        for i in range(coefs_array.shape[1]):
            axs[i,j].plot(hspace, coefs_array[:,i,j],lw=3)
            axs[i,j].set_ylim(ymin=ymin,ymax=ymax)
            axs[i,j].set_xlim(xmin=xmin,xmax=xmax)
            axs[i,j].tick_params(axis='both', labelsize=14)
            # place a text box in upper left in axes coords
            textstr = labels[i,j]
            axs[i,j].text(xmin + 0.01, (2/3)*ymax, textstr,fontsize=14) #  transform=axs[i,j].transAxes,
    # add column and row titles, set x-axis label
    for ax, col_label in zip(axs[0], col_labels): # code from https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
        ax.set_title(col_label,fontsize=18)
    for ax in axs[-1]:
        ax.set_xlabel('Threshold',fontsize=18)
    for ax, row_label in zip(axs[:,0], row_labels): 
        ax.set_ylabel(row_label, rotation=0, loc='bottom', labelpad=50, fontsize=18)

    fig.tight_layout()


def plot_complexity(complexity, hparams, first, last, step):
    # Note: DEPRECATED
    # ticklist = np.arange(first, last+step, step)
    # while len(ticklist) > 10:
    #     ticklist = ticklist[0::2] # slice list by taking every other element
        # the below does the same thing (maybe?) as the above but with more lines of code
        # step = step*2
        # ticklist = np.arange(first, last+step, step)
    # ticklist = np.append(ticklist, last) # doesn't line up with pattern
    plt.figure()
    plt.xlabel("Threshold Value")
    plt.ylabel("Number of terms")
    # plt.xticks(ticklist)
    ymax = np.max(complexity)
    plt.yticks(np.arange(0, ymax+1, 2))
    plt.plot(hparams, complexity, '.')
    plt.show()

def plot_complexity_objs(model_list: list, figsize=(8,6), num_terms_simulation=42):
    '''
    Description: Updated version of plot_complexity() which takes in list of ps.SINDy() objects and
        plots complexity vs. threshold
    '''
    # extract hparams and complexity from each model
    thresholds = np.array([model.optimizer.threshold for model in model_list])
    complexities = np.array([model.complexity for model in model_list])
    # plot complexity vs. thresholds
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.plot(thresholds, complexities, 'o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Number of terms')
    ax.hlines(num_terms_simulation, 0, np.max(thresholds), colors='k',
              linestyles='dashed', lw=1.5, label="Simulation Equations")
    ax.legend()
    fig.tight_layout()
    return fig, ax

# Merge with above func 'plot_complexity'
def plot_complexities(thresholds, complexities, noise_level:float):
    fig, ax = plt.subplots(1,1)
    labels = ["Weak model", "Strong model"]
    for i in range (2):
        ax.plot(thresholds, complexities[i], 'o', label=labels[i])
    ax.hlines(3, min(thresholds), max(thresholds),linestyles='dashed', lw=1.5, label="Simulation Equations")
    ax.set_ylim(0, None)
    ax.yaxis.set_major_locator(MultipleLocator(base=2))
    ax.legend()
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Number of terms")
    ax.set_title("Noise level: " + str(np.round(noise_level, 2)))
    fig.tight_layout()
    return fig, ax


def plot_pareto(train_sim: Yukawa_simulation, test_sim: Yukawa_simulation, threshold_scan, 
                plot_prediction: bool = False):
    # calculate x_dot with finite differences from clean test trajectory
    fd = FiniteDifference()
    xdot_test = fd._differentiate(test_sim.x, test_sim.t)

    # create empty list to store rmse values
    rmse_values = []
    for threshold in threshold_scan:
        # fit SINDy model to noisy data
        model = fit_Yukawa_model(train_sim, opt_str="stlsq", hparam=threshold)
        # predict with test trajectory
        xdot_test_predicted = model.predict(test_sim.x, test_sim.t)
        # compare trajectory predicted by SINDy model with clean test trajectory
        rmse = mean_squared_error(xdot_test, xdot_test_predicted, squared=False)
        rmse_values.append(rmse)

    if plot_prediction:
        _, ax = plt.subplots()
        ax.plot(test_sim.t, xdot_test)
        ax.plot(test_sim.t, xdot_test_predicted, linestyle='--')

    _, ax = plt.subplots(figsize=(5, 5))
    ax.plot(threshold_scan, rmse_values)
    ax.set(xlabel="Threshold", ylabel="RMSE")
    ax.set_ylim(ymin=0, ymax=5)

def plot_pos_maxes(sims:list):
    positions = np.array([sim.x[:,0] for sim in sims])
    avg_positions = np.average(positions, axis=1)
    sim_duration = sims[0].duration
    fig, ax = plt.subplots(1,1)
    ax.plot(avg_positions,"ko")
    ax.set_title("simulation duration: " + str(sim_duration) + "$\omega_{pd}$")
    print(sims[0].duration)
    ax.set_xlabel("trajectory index")
    ax.set_ylabel("max ptcl sep ($\lambda_{De}$)")
    return fig, ax

def plot_pos_avgs(avgs:np.ndarray, cutoffs, durations):
    # determine number of rows and columns for subplots based on number of cutoffs to be tested
    ncolumns = int(np.ceil(np.sqrt(len(cutoffs))))
    nrows = int(np.ceil(len(cutoffs)/ncolumns))
    # set min and max of 

    # initialize plot
    fig, axs = plt.subplots(nrows,ncolumns,figsize=(15,8), sharex=True, sharey=True)
    axs = axs.reshape(nrows*ncolumns)
    for i in range(avgs.shape[0]):
        axs[i].plot(durations, avgs[i], "ko")
        axs[i].set_title("Cutoff: " + str(cutoffs[i]) + "$\lambda_{De}$", fontsize=14)
    # common x label
    fig.supxlabel('duration ($\omega_{pd}^{-1}$)')
    # common y label
    fig.supylabel('avg ptcl separation ($\lambda_{De}$)')
    fig.tight_layout()
    return fig, axs

###############################################################################################
# MAIN FUNCTIONS
###############################################################################################
def explore_thresholds(sim_obj: Yukawa_simulation,
                       first, last, step,
                       verbose=False, plot=False,
                       opt_str="stlsq",
                       feature_names=None):
# Syntax: explore_thresholds(sim1, 0.0, 2.0, 0.1)
# Description: fits a SINDy model for thresholds in the parameter space provided
# by the arguments, then plots number of terms vs. threshold used.
# Note: DEPRECATED

    # create arrays to store complexity, hyperparameters
    # feature_names = ['x', 'v']
    complexity = np.array([])
    hspace = np.arange(first, last+step, step)

    coefs = np.empty((0,10,2))
    
    # Identify optimizer being used
    if verbose:
        print('optimizer:', opt_str)
        print('Std. dev. of noise:', sim_obj.noise_level)
        print('------------------------------')
    
    # scan through different hyperparameters
    for hparam in hspace:
        noisymodel, hparam_str = fit_Yukawa_model(sim_obj, opt_str=opt_str, hparam=hparam, 
                                                  return_hparam_str=True)
        new_coefs = noisymodel.coefficients().T # extract coefficient array from model
        new_coefs = new_coefs[np.newaxis, ...] # resize to append
        coefs = np.append(coefs, new_coefs, axis=0) # append to 'coefs' array
        complexity = np.append(complexity,noisymodel.complexity)
        if verbose:
            print(hparam_str, '=', round(hparam,3))
            noisymodel.print()
            print()
        # if plot:
        #     feature_names = noisymodel.get_feature_names()
        #     plot_coefs(noisymodel,hparam=round(hparam,3),std_dev=round(sim_obj.noise_level, 4))

        # commented this out because there have been some local minima which are < model.complexity
        # stop the loop if noisy model complexity is less than model complexity - 1

        # if noisymodel.complexity <= model.complexity-1:
        #     break
    # ensure that the complexity array is the same size as the hspace array
    hspace.resize(complexity.shape)
    # take absolute values of all coefs
    coefs = np.abs(coefs)

    if plot:
        if feature_names is None:
            # feature_names = noisymodel.get_feature_names()
            latex_feature_names = [
            r'$r$',
            r'$v$',
            r'$\frac{e^{-r}}{r}$',
            r'$\frac{e^{-v}}{v}$',
            r'$\frac{e^{-r}}{r^2}$',
            r'$\frac{e^{-v}}{v^2}$',
            r'$\frac{e^{-r}}{r^3}$',
            r'$\frac{e^{-v}}{v^3}$',
            r'$\frac{e^{-r}}{r^4}$',
            r'$\frac{e^{-v}}{v^4}$'
            ]
            feature_names = latex_feature_names
        plot_complexity(complexity, hspace, first, last, step)
        plot_coef_hist(hspace, coefs, feature_names=feature_names)
    
    # return complexity, hspace, coefs
    return

# Need to merge with above func 'explore_thresholds'
def scan_thresholds(data, thresholds, verbose=False):
    global A
    print(A)
    # data input must be a ys.Yukawa_simulation object or list of these
    # initialize variables
    if isinstance(data, list): 
        multiple_trajectories = True
        x_train = [sim.x for sim in data]
        t_train = data[0].t
        if not data[0].is_scaled:
            A = 1
    else: 
        multiple_trajectories = False
        x_train = data.x
        t_train = data.t
        if not data.is_scaled:
            A = 1

    weak_lib, strong_lib = generate_libraries(t_train)

    precision = 5
    num_simeq = 3
    complexities = np.empty((2,0))
    fit_weak, fit_strong = 2*(True,)

    # loop through thresholds and fit weak and strong SINDy models to the given dataset
    for thresh in thresholds:
        # set optimizer
        opt = ps.STLSQ(threshold=thresh)
        if verbose:
            print("STLSQ threshold:", np.round(thresh/A, decimals=3), "A")
        # fit weak model
        if fit_weak:
            weak_model = ps.SINDy(feature_names=["x", "v"],
                                feature_library=weak_lib, optimizer=opt)
            weak_model.fit(x_train, multiple_trajectories=multiple_trajectories)
            if verbose:
                print("\nWeak model")
                weak_model.print(precision=precision)
            weak_complexity = weak_model.complexity
        else:
            weak_complexity = -1

        # fit strong model
        if fit_strong:
            strong_model = ps.SINDy(feature_names=["x", "v"],
                                    feature_library=strong_lib, optimizer=opt)
            strong_model.fit(x_train, t=t_train, multiple_trajectories=multiple_trajectories)
            if verbose:
                print("\nStrong model")
                strong_model.print(precision=precision)
                print("\n" + 100*"=" + "\n")
            strong_complexity = strong_model.complexity
        else:
            strong_complexity = -1
        
        # append new complexities to list
        new_complexities = np.array([weak_complexity, strong_complexity]).reshape((2,1))
        complexities = np.append(complexities, new_complexities, axis=1)

        # stop fitting models if complexity is below num_simeq
        if weak_complexity <= num_simeq:
            fit_weak = False
        if strong_complexity <= num_simeq:
            fit_strong = False
        if not fit_weak and not fit_strong:
            break

    # change threshold to have the same dim as complexities
    num_complexities = complexities.shape[1]
    thresholds = thresholds[0:num_complexities]
        
    return thresholds, complexities


def explore_noises(sim_obj: Yukawa_simulation,
                   first, last, step, 
                   verbose=False, plot=False,
                   hparam = 0.1,
                   opt_str="stlsq",
                   ):
    # create array of noises to try
    noisespace = np.arange(first, last+step, step)
    # print optimizer, hparam if desired
    if verbose:
        print('optimizer:', opt_str)
        print('hparam =', hparam)
        print('----------------------------')
    # scan through different noise levels and fit SINDy model for each one
    for noise in noisespace:
        # delete noise if any
        if sim_obj.is_noisy:
            sim_obj.delete_noise()
        # add noise with current noise level and fit
        sim_obj.add_gaussian_noise(std_dev=noise)
        noisymodel = fit_Yukawa_model(sim_obj, opt_str=opt_str, hparam=hparam)
        # print model, if desired
        if verbose:
            print('standard deviation =', round(noise,4))
            noisymodel.print()
            print()
        # plot model coefs, if desired
        if plot:
            plot_coefs(noisymodel,hparam=round(hparam,3),std_dev=round(noise,4))


# added for training with multiple trajectories
def generate_training_data(n_sims=200, duration=5, dt=0.001, noise_level=0.01, mu_x0s=1, mu_v0s=0.01, scaled=False):
    # generates list of multiple trajectories with random initial conditions
    # generate init cond
    rng = np.random.default_rng(seed=4862039)
    x0s = rng.normal(mu_x0s, 0.2, n_sims)
    v0s = rng.choice([-1,1], n_sims) * rng.normal(-mu_v0s, 0.002, n_sims)
    # simulate
    sims = []
    for i in range(n_sims):
        sim = Yukawa_simulation()
        sim.simulate(duration, dt=dt, x0=x0s[i], v0=v0s[i], scaled=scaled)
        if noise_level != 0:
            sim.add_gaussian_noise(noise_level=noise_level)
        sims.append(sim)
    
    return sims


def determine_avg_f(sim_duration=5, n_debyes=5, plot=False):
    global A
    data = generate_training_data(n_sims=200, duration=sim_duration, dt=0.001, noise_level=0, 
                                  mu_x0s=1, mu_v0s=0.01, scaled=True)

    n_datapoints = len(data) * len(data[0].x)
    xs = np.array([sim.x[:,0] for sim in data]).reshape(n_datapoints)

    # print(xs.shape, "before truncation")

    # truncate any positions outside of n_debyes debye lengths
    large_xs = [i for i, x in enumerate(xs) if x >= n_debyes]
    xs = np.delete(xs, large_xs)

    # print(xs.shape, "after truncation")

    accels = A * ( 1/xs + 1/xs**2 ) * np.exp( -xs )

    if plot:
        fig, ax = plot_pos_maxes(data)
        fig.show()

    return np.average(accels)

def test_diff_durations(cutoffs, durations):
    avgs = np.zeros((len(cutoffs),len(durations)))
    # loop through cutoff values
    for i, cutoff in enumerate(cutoffs):
        print("cutoff", cutoff)
        # loop through simulation durations
        for j, duration in enumerate(durations):
            print("duration", duration)
            avgs[i,j] = determine_avg_f(sim_duration=duration, n_debyes=cutoff, plot=False)
    return avgs