###########################################################
# File: weak_noise_robustness.py                          #
# Created by: Z. B. Howe                                  #
# Description: Performs and saves Weak SINDy analysis     #
#   method on different levels of noisy data and saves    #
#   the learned coefficients with the two different       #
#   selection criteria: lowest prediction error on with-  #
#   hold data and smallest deviation from actual coef-    #
#   ficients.                                             #
###########################################################

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

import pysindy as ps
import numpy as np
import matplotlib.pyplot as plt

import pickle as pkl
with open('scaling_const.float', 'rb') as f:
    SCALING_CONST = pkl.load(f)

def noise_analysis(noise_level: float, threshold_space: np.ndarray):
    # set up place to save results
    noise_name = f'{noise_level:.3f}'.replace('.', '_')
    directory_name = 'weak_noise_robustness/noise_' + noise_name

    # generate 200 trajectories with random initial conditions on which to run SINDy analysis
    simulation_list = ys.generate_training_data(
        n_sims = 200,
        duration = 5,
        dt = 1e-3,
        noise_level = noise_level,
        mu_x0s = 1,
        mu_v0s = 0.01,
        scaled = True
    )

    all_best_models = []
    all_best_model_scores = []
    all_truest_models = []
    all_truest_model_scores = []
    for threshold in threshold_space:
        # generate weak library
        feature_library = ys.generate_weak_Yukawa_library(simulation_list[0].t)

        feature_names = ['x', 'v']
        true_coefficients = np.array(
            [[0., 1., 0.,            0., 0.,            0., 0., 0., 0., 0.],
            [0., 0., SCALING_CONST, 0., SCALING_CONST, 0., 0., 0., 0., 0.]]
        )

        # perform cross-validation
        CV_output = cv.cross_validate(
            simulation_list,
            threshold,
            feature_library,
            feature_names,
            true_coefficients,
            n_folds = 10
        )

        # unpack and append to lists
        best_model, best_model_score, truest_model, truest_model_score = CV_output

        all_best_models.append(best_model)
        all_best_model_scores.append(best_model_score)
        all_truest_models.append(truest_model)
        all_truest_model_scores.append(truest_model_score)
    
    # Save coefficients, scores, and thresholds
    all_best_coefficients = [model.coefficients() for model in all_best_models]
    all_truest_coefficients = [model.coefficients() for model in all_truest_models]
    data_to_save = [
        all_best_coefficients,
        all_best_model_scores,
        all_truest_coefficients,
        all_truest_model_scores,
        threshold_space
    ]
    file_names = [
        'best_coefs.pickle',
        'best_model_prediction_errors.pickle',
        'truest_coefs.pickle',
        'truest_model_prediction_errors.pickle',
        'thresholds.pickle'
    ]

    for data, file_name in zip(data_to_save, file_names):
        ys.pickle_data(
            data,
            directory_name,
            file_name,
            overwrite = False
        )

    return

def main():
    # set up noises and thresholds to scan through
    threshold_space = np.arange(0., 1.2, 0.2)
    noise_space = np.arange(0.2, 0.5, 0.1)

    # loop through and save data
    for noise_level in noise_space:
        noise_analysis(noise_level, threshold_space)

def testmain():
    noise_level = 0.1
    threshold_space = np.arange(0., 1.2, 0.2)
    noise_analysis(noise_level, threshold_space)

if __name__ == '__main__':
    main()