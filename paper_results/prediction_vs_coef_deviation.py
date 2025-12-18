# import modules from parent dir
# from <https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder>
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pysindy as ps
import numpy as np
import matplotlib.pyplot as plt

import Yukawa_SINDy as ys
import cross_validation as cv

import pickle as pkl
with open('scaling_const.float', 'rb') as f:
    SCALING_CONST = pkl.load(f)


def generate_cv_models(threshold, noise_level):
    # generate data
    sim_list = ys.generate_training_data(noise_level=noise_level) # using default values

    # format data for kfold_training
    x_train = np.array([sim.x for sim in sim_list])
    if cv.same_times(sim_list):
        t_train = sim_list[0].t
    else:
        raise Exception('sims need to have the same time grid')

    # define SINDy parameters
    feature_library = ys.generate_weak_Yukawa_library(t_train, K=500)
    feature_names = ['x', 'v']

    # fit 'dummy model' to pass into kfold_training
    opt = ps.STLSQ(threshold=threshold)
    n_timesteps = x_train[0].shape[0]
    n_features = x_train[0].shape[1]
    rand_data = np.random.random((n_timesteps, n_features))
    test_model = ps.SINDy(
        optimizer=opt, feature_library=feature_library, feature_names=feature_names
    )
    test_model.fit(rand_data)


    # define true coefficients to calculate deviation
    true_coefficients = np.array(
        [[0., 1., 0.,            0., 0.,            0., 0., 0., 0., 0.],
        [0., 0., SCALING_CONST, 0., SCALING_CONST, 0., 0., 0., 0., 0.]
        ]
    )

    # pull all models generated during cv
    models, coef_devs, rmses = cv.kfold_training(
        x_train,
        t_train,
        n_folds=10,
        SINDy_model=test_model,
        true_coefficients=true_coefficients
    )

    return models, coef_devs, rmses


def repeat_cv(threshold, n_repeats, noise_level):
    all_models = np.array([])
    all_coef_devs = np.array([])
    all_rmses = np.array([])
    for i in range(n_repeats):
        print(f"\riter: {i}\n")
        models, coef_devs, rmses = generate_cv_models(threshold, noise_level)
        all_models = np.hstack((all_models, models))
        all_coef_devs = np.hstack((all_coef_devs, coef_devs))
        all_rmses = np.hstack((all_rmses, rmses))
    
    return all_models, all_coef_devs, all_rmses


def threshold_scan(threshold_space, n_repeats, noise_level):

    if not hasattr(threshold_space, '__iter__'):
        threshold_space = [threshold_space]

    for thresh in threshold_space:
        # printing Threshold
        print(f"\rThreshold: {thresh:.2f}\n")

        # do multiple cv analyses
        models, coef_devs, rmses = repeat_cv(thresh, n_repeats, noise_level)

        # define data directory
        threshold_str = f'{thresh:.2f}'.replace('.', '_')
        directory_prefix = 'paper_results/prediction_vs_coef_deviation/threshold_'
        directory = directory_prefix + threshold_str
        
        # put data and filenames in lists
        coefficients = [model.coefficients() for model in models]
        stuff_to_save = [
            coefficients,
            coef_devs.tolist(),
            rmses.tolist()
        ]
        file_names = [
            'model_coefficients.pickle',
            'coefficient_devs.pickle',
            'prediction_errors.pickle'
        ]

        # save data
        for data, file_name in zip(stuff_to_save, file_names):
            ys.pickle_data(
                data, 
                directory,
                file_name,
                overwrite=True
            )
    
    return 


def main():
    threshold_space = np.arange(0.4, 1., 0.1)
    n_repeats = 10
    noise_level = 0.1
    threshold_scan(threshold_space, n_repeats, noise_level)


if __name__ == '__main__':
    main()