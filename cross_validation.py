'''
File:         cross_validation.py
Written by:   Brooks Howe
Last updated: 2025/10/24
Description:  Library of python functions used for cross-validation analysis
'''

# import function file
import Yukawa_SINDy as ys
# import libraries
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import pickle as pkl
import numpy as np
import xarray as xr
import pysindy as ps
from pysindy.feature_library.base import BaseFeatureLibrary
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ignore warnings generated from using LaTeX coding in matplotlib label strings
from warnings import filterwarnings
filterwarnings('ignore', message = 'invalid escape sequence')

# import scaling constant from working directory and declare as global variable
from pickle import load
with open('scaling_const.float','rb') as f:
    SCALING_CONST = load(f)

def same_times(list_of_sims:list):
    '''
    Description: Helper function to check if all simulations in a list of 
    Yukawa_SINDy.Yukawa_simulation objs have the same time grid. returns 
    True or False.
    '''
    same_times:bool = True
    t_check = list_of_sims[0].t
    for sim in list_of_sims[1:]:
        if not np.all(t_check == sim.t):
            same_times = False
            break
    return same_times


def kfold_training(
    x_train:np.ndarray, 
    t_data:np.ndarray, 
    n_folds:int, 
    SINDy_model:ps.SINDy, 
    verbose:bool=False
):
    '''
    Description: takes in training data and associated time data and performs k-fold
    cross-validation with k=n_folds. Takes in a ps.SINDy object to extract the library
    of terms, feature names, and optimizer. Prints all model if verbose is True.
    Returns rmse score from all models and a 3D array of coefficients from all models.
    '''
    # check dimension of training_data and t_data arrays
    if x_train.ndim!=3:
        raise Exception('training data has wrong dimensions')
    if x_train.shape[1]!=t_data.shape[0]:
        raise Exception('time data has wrong dimensions')
    # get SINDy parameters from input SINDy model
    feature_list = SINDy_model.get_feature_names()
    n_features = len(feature_list)

    # check if feature_library is weak or strong, don't need
    # to pass time as an arg into the 'fit' method of ps.SINDy.
    # if weak, need to recreate weak library inside loop to get
    # different random sampling of subdomains of integration
    use_weak = None
    K = None
    if isinstance(SINDy_model.feature_library, ps.WeakPDELibrary):
        use_weak = True
        t_to_fit = None
        K = SINDy_model.feature_library.K
    else:
        use_weak = False
        t_to_fit = t_data

    feature_names = SINDy_model.feature_names
    library_functions = SINDy_model.feature_library.functions
    library_function_names = SINDy_model.feature_library.function_names
    
    # perform KFold CV
    all_rmse = np.array([])
    all_models = np.array([])
    kf = KFold(n_splits=n_folds)
    for train, test in kf.split(x_train):
        # split training data
        x_train_kf = [traj for traj in x_train[train]]
        x_test_kf  = [traj for traj in x_train[test]]
        
        # fit SINDy model using given threshold
        if use_weak:
            feature_library = ps.WeakPDELibrary(
                library_functions = library_functions,
                function_names = library_function_names,
                spatiotemporal_grid = t_data,
                K = K
            )

        else:
            feature_library = ps.CustomLibrary(
                library_functions = library_functions,
                function_names = library_function_names
            )
            
        # instantiate and fit SINDy model
        opt = ps.STLSQ(threshold = SINDy_model.optimizer.threshold)
        model = ps.SINDy(
            optimizer = opt, 
            feature_library = feature_library, 
            feature_names = feature_names
        )
        model.fit(x_train_kf, t_to_fit, multiple_trajectories=True)
        if verbose: model.print()

        # append to list of models
        all_models = np.hstack((all_models,model))

        # validate model against test data
        # print(f'test traj shape: {x_test_kf[0].shape}') # included for testing
        # print(f'coefficients shape: {model.coefficients().shape}') # included for testing
        rmse = model.score(x_test_kf, t=t_data, multiple_trajectories=True, metric=root_mean_squared_error)
        all_rmse = np.hstack((all_rmse, rmse))

    # pull out coefs with the lowest error from cross val
    best_model = all_models[all_rmse.argmin()]

    return all_models, all_rmse


def test_on_withhold(
    x_withhold:np.ndarray, 
    t_data:np.ndarray, 
    feature_library:ps.feature_library.base.BaseFeatureLibrary, 
    coefs:np.ndarray
):
    '''
    Description: This function tests the SINDy model described by 'coefs' on multiple trajectories 
    data passed with 'x_withhold'. This is done by computing the rmse value between the prediction
    of x_dot generated by the SINDy model with the calculated x_dot using finite difference.
    '''
    n_features = x_withhold.shape[-1]
    lib_funcs = feature_library.functions
    lib_for_pred = ps.CustomLibrary(library_functions=lib_funcs)
    fd = ps.FiniteDifference()
    all_x_dot_pred = np.empty((0,n_features))
    all_x_dot_calc = np.empty_like(all_x_dot_pred)
    for traj in x_withhold:
        # predicted x_dot
        lib_for_pred.fit(traj)
        Phi = lib_for_pred.transform(traj)
        x_dot_pred = Phi@coefs.T
        all_x_dot_pred = np.array(np.vstack((all_x_dot_pred,x_dot_pred)))
        # calculated x_dot
        x_dot_calc = fd._differentiate(traj,t_data)
        all_x_dot_calc = np.array(np.vstack((all_x_dot_calc,x_dot_calc)))

    rmse = root_mean_squared_error(all_x_dot_calc,all_x_dot_pred)

    return rmse


def cross_validate(
    all_data: list, 
    threshold: float, 
    feature_library: BaseFeatureLibrary, 
    feature_names, 
    n_folds=10
):
    '''
    Description: This function performs k-fold cross-validation (cv) with k specified by the 'n_folds'
    (default 10) argument. Gets help from the 'sklearn.model_selection.KFold' object. Takes a list 
    of Yukawa_SINDy.Yukawa_simulation objects, a SINDy STLSQ threshold, a feature library 
    ('pysindy.BaseFeatureLibrary' child objs), and feature names as args. Returns a rank 3 numpy
    array of coefficients from the best two models: the one with the lowest error and the average
    coefficients of all models generated during k-fold cv. Generates coefficients using the weak
    library and makes predictions using those coefficients and the strong library's 'transform'
    method.
    '''
    # check if list of sim objects
    for item in all_data:
        if not isinstance(item, ys.Yukawa_simulation):
            raise TypeError("Argument 'all_data' should be list of 'Yukawa_SINDy.Yukawa_simulation' objects")
    # check if all time grids are the same
    if not same_times(all_data):
        raise Exception("All simulations do not have the same time grid.")
    
    # extract data from sim objects
    x_data = np.array([sim.x for sim in all_data])
    t_data = all_data[0].t
    n_timesteps = t_data.shape[0]
    n_features = all_data[0].x.shape[1]
    # print(f'shape and ndims of t_data: {t_data.shape}, {t_data.ndim}') # included for testing

    # split data into withhold(testing) and training data
    n_trajectories = len(all_data)
    rng = np.random.default_rng(seed=10235783)
    withhold_idxs = rng.choice(x_data.shape[0], np.floor(0.25 * n_trajectories).astype(int), replace=False)
    withhold_idxs.sort()
    train_idxs = np.delete(np.arange(len(all_data)), withhold_idxs)
    x_train = x_data[train_idxs]
    x_withhold = x_data[withhold_idxs]

    # declare optimizer with given threshold
    opt = ps.STLSQ(threshold=threshold)

    # get number of terms in library
    rand_data = np.random.random((n_timesteps,n_features))
    test_model = ps.SINDy(optimizer=opt, feature_library=feature_library, feature_names=feature_names)
    test_model.fit(rand_data)

    # perform kfold cv
    # best_coefs, avg_coefs = kfold_training(x_train,t_data,n_folds,test_model)
    all_models, best_model = kfold_training(x_train,t_data,n_folds,test_model)

    # delete unnecessary vars
    del test_model, rand_data

    # calculate rmse with withhold data
    # best_rmse = test_on_withhold(x_withhold, t_data, feature_library, best_coefs)
    # avg_rmse  = test_on_withhold(x_withhold, t_data, feature_library, avg_coefs)
    best_model_score = best_model.score(
        x_withhold,
        t=t_data, 
        multiple_trajectories=True, 
        metric=root_mean_squared_error
    )

    
    return best_model, best_model_score


def two_body_param_scan(noise_space, threshold_space):
    '''
    Description: This function does a sweep through the noises given by 'noise_space', 
    generates noisy data at the different noise levels. At each noise level, a SINDy
    analysis is performed using the library created in ys.generate_Yukawa_library and
    ys.generate_weak_Yukawa_library. 10-fold cross-validation (cv) is then performed at 
    each level of threshold specified by 'threshold_space'. The coefficients from the 
    best (lowest rmse) and average models from the cv are then collected and stored in 
    'xr.DataArray' objects.
    '''
    # convert args to iterables if they are just one number
    if not hasattr(noise_space, '__iter__'):
        noise_space = [noise_space]
    if not hasattr(threshold_space, '__iter__'):
        threshold_space = [threshold_space]


    # define number of folds in kfold cv
    n_folds = 10

    # define spaces not characterized by arguments
    feature_names = ['x', 'v']
    formulations = ['weak', 'strong']
    model_selection_criteria = ['best', 'average']

    # count for formatting the dimensions of results arrays
    n_thresholds = len(threshold_space)
    n_formulations = len(formulations)
    n_CV_selections = 2
    n_equations = 2
    n_library_terms = 10
    n_models = n_thresholds * n_formulations * n_CV_selections
    n_noises = len(noise_space)

    # create empty arrays for coefficients and errors
    empty_results_array = np.zeros((
        n_thresholds,
        n_formulations,
        n_CV_selections,
        n_equations,
        n_library_terms
    ))
    empty_error_array = np.zeros((
        n_thresholds,
        n_formulations,
        n_CV_selections
    ))

    # loop through all noises in noise_space and perform SINDy analysis with cross-validation
    all_coefs = []
    all_rmses = []
    for noise in noise_space:
        # set up DataArray structures to save coefficients and rmses of both
        # the best and average model from cross-validation
        print(f'\n\nnoise = {noise}')

        # common dimensions
        param_dims=[
            "threshold",
            "formulation",
            "cv_selection"
        ]
        coords={
            "threshold": threshold_space,
            "formulation": formulations,
            "cv_selection": model_selection_criteria
        }

        # coefficients array
        SINDy_coefficients = xr.DataArray(
            empty_results_array,
            dims = param_dims + [
                "equation",
                "library term"
            ],
            coords = coords
        )

        # error array
        SINDy_model_rmses = xr.DataArray(
            empty_error_array,
            dims = param_dims,
            coords = coords
        )

        # save noise level as array attribute
        SINDy_coefficients.attrs["noise_level"] = noise
        SINDy_model_rmses.attrs["noise_level"] = noise

        # generate training data
        sim_list = ys.generate_training_data(
            n_sims=200,
            duration=5,
            dt=1e-3, 
            noise_level=noise,
            mu_x0s=0.5, 
            mu_v0s=0.01, 
            scaled=True
        )

        # generate weak and strong form libraries
        strong_library = ys.generate_Yukawa_library()
        weak_library = ys.generate_weak_Yukawa_library(sim_list[0].t)
        libraries = (weak_library, strong_library)

        # loop through thresholds and weak and strong formulations of SINDy
        for i, threshold in enumerate(threshold_space):
            print(f'\nthreshold = {threshold}')
            for j, lib in enumerate(libraries):
                best_coefs, best_rmse, avg_coefs, avg_rmse = cross_validate(
                    sim_list,
                    threshold,
                    lib,
                    feature_names,
                    n_folds=n_folds
                )
                # save best model info
                SINDy_coefficients[i, j, 0] = best_coefs
                SINDy_model_rmses[i, j, 0] = best_rmse
                # save average model info
                SINDy_coefficients[i, j, 1] = avg_coefs
                SINDy_model_rmses[i, j, 1] = avg_rmse

        # save all results for this noise level
        all_coefs.append(SINDy_coefficients)
        all_rmses.append(SINDy_model_rmses)

    return all_coefs, all_rmses


def plot_pareto(coefs, rmses, noise_level):
    n_plots = len(coefs.formulation)
    fig, axs = plt.subplots(1, n_plots, figsize=(16,9))
    fig.suptitle(f'Noise = {noise_level:.5f}')
    for ax, form in zip(axs, coefs.formulation):
        ax.set_title(f'{form.to_numpy()}')

        # plot best model pareto plot
        best = 'best'
        best_model_coefs = coefs.sel(formulation=form, cv_selection=best)
        best_model_num_terms = np.count_nonzero(best_model_coefs, axis=(1,2))
        best_model_rmses = rmses.sel(formulation=form, cv_selection=best)

        ax.plot(best_model_num_terms, best_model_rmses, 'o', label=best)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # plot average model on pareto plot
        average = 'average'
        avg_model_coefs = coefs.sel(formulation=form, cv_selection=average)
        avg_model_num_terms = np.count_nonzero(avg_model_coefs, axis=(1,2))
        avg_model_rmses = rmses.sel(formulation=form, cv_selection=average)

        ax.plot(avg_model_num_terms, avg_model_rmses, 'o', label=average)
        ax.legend()
    fig.supxlabel('Number of terms')
    fig.supylabel('RMS Error')
    fig.tight_layout()

    return fig, axs


def main():
    noise_space = np.logspace(-4,-1,10)
    threshold_space = np.arange(0,1,0.01) # placeholder
    coefs, rmses = two_body_param_scan(noise_space,threshold_space)

    # save coefs and rmses using pickle
    data_directory = 'final_results/'

    coefs_filename = 'coefs.pkl'
    with open(data_directory + coefs_filename, 'wb') as f:
        pkl.dump(coefs, f)

    rmses_filename = 'rmses.pkl'
    with open(data_directory + rmses_filename, 'wb') as f:
        pkl.dump(rmses, f)

    for coef, rmse, noise in zip(coefs, rmses, noise_space):
        plot_pareto(coef,rmse, noise)

def test_plot():
    noise_space = [1e-4, 1e-2]
    threshold_space = [0., 0.5]
    all_coefs, all_rmses = two_body_param_scan(noise_space, threshold_space)
    noise_to_plot = noise_space[0]
    plot_pareto(all_coefs[0], all_rmses[0], noise_to_plot)


def test_kfold_training():
    sim_list = ys.generate_training_data(mu_x0s=0.5, noise_level=0.1, scaled=True)

    x_train = [sim.x for sim in sim_list[0:150]]
    x_test = [sim.x for sim in sim_list[150:200]]

    threshold = 0.7
    feature_names = ['x', 'v']
    opt = ps.STLSQ(threshold=threshold)

    weak_library = ys.generate_weak_Yukawa_library(sim_list[0].t)
    model = ps.SINDy(
        optimizer=opt, 
        feature_library=weak_library, 
        feature_names=feature_names
    )
    model.fit(x_train, t=sim_list[0].t, multiple_trajectories=True)
    coefs = model.coefficients()

    model.score(x_test, t=sim_list[0].t, multiple_trajectories=True)



if __name__ == '__main__':
    test_kfold_training()