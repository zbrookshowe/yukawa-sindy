'''
File:         basic_sindy_results_noisy.py
Written by:   Brooks Howe
Last updated: 2025/06/03
Description:  Python script which generates results using the SINDy algorithm for the 3-body Yukawa
    system of point particles with different noise levels added. Also can use weak formulation if 
    use_weak=True.
'''
import Yukawa_SINDy as ys
import Yukawa3body as y3
import numpy as np
import pysindy as ps

# use weak or strong formulation
use_weak = True
K=100 # set K for weak form

# define relative path to data directory where data is or will be stored
data_dir = 'data/basic_noisy/analysis_trajectories'
SINDy_dir = 'data/weak_noisy/K_test/SINDy_results'

# set whether to generate and save data
generate_data = False
save_data = True # only relevant if generate_data is True

# set simulation parameters used if generating data, not relevant if using saved data
n_trajectories = 200
sim_duration = 1e-1
potential_type = 'repulsive'
dt=1e-4

# define noise levels
noise_levels = [0] # np.arange(0, 1.2e-4, 2e-5)
# define threshold values to use with SINDy
threshold_array = [0.30,0.35] # np.arange(0.0, 1.0, 0.05)

if generate_data:
    # create list of sim objects with different noise levels
    seed_num = 109274
    rng = np.random.default_rng(seed=seed_num)
    sim_list = y3.multiple_simulate(duration=sim_duration, dt=dt, n_trajectories=n_trajectories, 
                                    potential_type=potential_type,rng=rng, save_data=save_data, 
                                    directoryname=data_dir)
else:
    # load saved data
    sim_list = y3.load_data(data_dir)

# generate SINDy library
lib = y3.generate_3body_library(use_weak=use_weak, spatiotemporal_grid=sim_list[0].t, K=K)
# loop through noise levels
for noise_level in noise_levels:
    # transform to subtracted space and add noise to each simulation in sim_list
    x_train = []
    for sim in sim_list:
        # delete noise if present
        if sim.is_noisy:
            sim.delete_noise()
        # subtract data and add noise
        if not sim.is_subtracted:
            sim.subtract_data()
        if noise_level != 0:
            sim.add_gaussian_noise(noise_level=noise_level) # just use one noise level for now
        # collect data into list x_train for SINDy fitting
        x_train.append(sim.x)

    # extract labels and time step from last simulation in sim_list
    x_train_labels = sim.labels
    t = sim.t
    # fit a SINDy model using different thresholds
    for threshold in threshold_array:
        print("fitting model with noise level", noise_level, "and threshold", threshold)
        opt = ps.STLSQ(threshold=threshold)
        model = ps.SINDy(optimizer=opt, feature_names=x_train_labels, feature_library=lib)
        if use_weak:
            model.fit(x_train, multiple_trajectories=True)
        else:
            model.fit(x_train, t=t, multiple_trajectories=True)
        # # for testing:
        # print('STLSQ threshold:', threshold)
        # print('Std. dev. of noise:', noise_levels[0])
        # print('complexity:', model.complexity)
        # y3.print_SINDy_nice(model)
        
        # save/print model as .obj and .txt files
        y3.save_SINDy_model(model, sim_list, directoryname=SINDy_dir)
