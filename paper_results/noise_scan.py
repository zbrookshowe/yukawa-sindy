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

def noise_scan(thresholds, noise_space):
    # change thresholds to be an iterable with the same size as noise_space if it is not
    if not hasattr(thresholds, '__iter__'):
        thresholds = np.repeat(thresholds, len(noise_space))
    if not len(thresholds) == len(noise_space):
        raise ValueError("\'thresholds\' must have the same length as \'noise_space\'")

    truest_model_coefs = []
    truest_model_scores = []
    for threshold, noise_level in zip(thresholds, noise_space):
        sim_list = ys.generate_training_data(noise_level = noise_level)
        feature_library = ys.generate_weak_Yukawa_library(sim_list[0].t, K=500)
        # define true coefs
        true_coefficients = np.array(
            [[0., 1., 0.,            0., 0.,            0., 0., 0., 0., 0.],
             [0., 0., SCALING_CONST, 0., SCALING_CONST, 0., 0., 0., 0., 0.]]
        )
        _, _, truest_model, truest_model_score = cv.cross_validate(
            sim_list, 
            threshold, 
            feature_library, 
            feature_names = ['x','v'], 
            true_coefficients = true_coefficients
        )
        truest_model_coefs.append(truest_model.coefficients())
        truest_model_scores.append(truest_model_score)

    return truest_model_coefs, truest_model_scores

def main():
    # loop through noise levels, hitting each multiple times, at only one threshold.
    num_cross_vals = 10
    noise_space =   num_cross_vals*[1e-4] + num_cross_vals*[1e-3] + num_cross_vals*[1e-2] + num_cross_vals*[1e-1]
    thresholds  = 3*num_cross_vals*[0.5]                                                  + num_cross_vals*[0.7]
    truest_model_coefs, truest_model_scores = noise_scan(thresholds, noise_space)

    # save data
    data_directory = 'paper_results/weak_noise_robustness/noise_scan2'
    data_to_save = [noise_space, thresholds, truest_model_coefs, truest_model_scores]
    file_names = [
        'noises.pickle',
        'thresholds.pickle',
        'truest_coefs.pickle',
        'truest_prediction_errors.pickle'
    ]
    for data, file_name in zip(data_to_save, file_names):
        ys.pickle_data(
            data,
            data_directory,
            file_name
        )

if __name__ == '__main__':
    main()