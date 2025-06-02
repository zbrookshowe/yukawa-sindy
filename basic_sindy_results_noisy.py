'''
File:         basic_sindy_results_noisy.py
Written by:   Brooks Howe
Last updated: 2025/06/02
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

# define relative path to data directory where data is or will be stored
data_dir = 'data/basic_noisy/analysis_trajectories'
if use_weak:
    SINDy_dir = 'data/weak_noisy/SINDy_results'
else:
    SINDy_dir = 'data/basic_noisy/SINDy_results'

# set whether to generate and save data
generate_data = False
save_data = True # only relevant if generate_data is True

# set simulation parameters used if generating data, not relevant if using saved data
n_trajectories = 200
sim_duration = 1e-1
potential_type = 'repulsive'
dt=1e-4

# define noise levels
noise_levels = [5e-4] # np.arange(0, 1.2e-4, 2e-5)
# define threshold values to use with SINDy
threshold_array = np.arange(0.0, 1.0, 0.05)

if generate_data:
    # create list of sim objects with different noise levels
    seed_num = 109274
    rng = np.random.default_rng(seed=seed_num)
    sim_list = y3.multiple_simulate(duration=sim_duration, dt=dt, n_trajectories=n_trajectories, 
                                    potential_type=potential_type,rng=rng, save_data=save_data, 
                                    directoryname=data_dir
                                    )
else:
    # load saved data
    sim_list = y3.load_data(data_dir)

# generate SINDy library
lib = y3.generate_3body_library(use_weak=use_weak, spatiotemporal_grid=sim_list[0].t)
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
    dt = sim.dt
    # fit a SINDy model using different thresholds
    for threshold in threshold_array:
        print("fitting model with noise level", noise_level, "and threshold", threshold)
        opt = ps.STLSQ(threshold=threshold)
        model = ps.SINDy(optimizer=opt, feature_names=x_train_labels, feature_library=lib)
        model.fit(x_train, t=dt, multiple_trajectories=True)
        # print('STLSQ threshold:', threshold)
        # print('Std. dev. of noise:', noise_levels[0])
        # print('complexity:', model.complexity)
        # y3.print_SINDy_nice(model)
        
        # save/print model as .obj and .txt files
        y3.save_SINDy_model(model, sim_list, directoryname=SINDy_dir)

        # old junk, delete if func 'y3.save_SINDy_model' works
        # save result as txt
        # result = y3.SINDy_results_nice(model, sim_list)
        # noise_rounded = np.format_float_positional(noise_level, unique=False, precision=5)
        # threshold_rounded = np.format_float_positional(threshold, unique=False, precision= 3)
        # print model
        # for line in result:
        #     print(line)

        # save model
        
        # model_filename = f'noise_{noise_rounded}_thresh_{threshold_rounded}_results.txt'
        # model_path = SINDy_dir + '/' + model_filename
        # with open(model_path, 'w') as f:
        #     f.writelines(line + '\n' for line in result)
