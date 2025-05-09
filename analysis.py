import Yukawa3body as y3
import numpy as np
import pysindy as ps
import pickle as pkl
import os

def load_data(directory_of_pkls_only:str):
    sim_list = []
    for filename in os.listdir(directory_of_pkls_only):
        with open(f"{directory_of_pkls_only}/{filename}", 'rb') as f:
            sim = pkl.load(f)
            sim_list.append(sim)
    return sim_list

def check_if_subtracted(sim_list:list):
    some_not_subtracted = False
    for i, sim in enumerate(sim_list):
        if not sim.is_subtracted:
            some_not_subtracted = True
            print(f"simulation {i} is not subtracted")

    if some_not_subtracted:
        print("Warning: not all data is subtracted")

def check_if_noisy(sim_list:list):
    some_not_noisy = False
    for i, sim in enumerate(sim_list):
        if not sim.is_noisy:
            some_not_noisy = True
            print(f"simulation {i} is not noisy")

    if some_not_noisy:
        print("Warning: not all data is noisy")

def generate_data(noise_levels:list, data_dir:str):
    '''
    Description: Generates data using the 3-body Yukawa system of point particles with noises given 
    by argument noise_levels and saves it in data directory.
    '''
    for noise_level in noise_levels:
        # create directory name for this specific noise level
        data_dir_noise_level = data_dir + '/noise_level_' + str(noise_level)
        # create list of sim objects with different noise levels
        seed_num = 109274
        rng = np.random.default_rng(seed=seed_num)
        sim_list = y3.multiple_simulate(duration=1e-1, n_trajectories=200, 
                                        potential_type='repulsive', rng=rng, save_data=True,
                                        directoryname=data_dir_noise_level
                                        )

# define relative path to data directory
data_dir = 'data/basic_noisy'
# define noise levels
noise_levels = [1e-3, 5e-3, 1e-2]
# define threshold values to use with SINDy
threshold_array:np.ndarray = np.arange(0.0, 0.9, 0.1)

# loop through noise levels
for noise_level in noise_levels:
    # create directory name for this specific noise level
    data_dir_noise_level = data_dir + '/noise_level_' + str(noise_level)
    # create list of sim objects with different noise levels
    seed_num = 109274
    rng = np.random.default_rng(seed=seed_num)
    sim_list = y3.multiple_simulate(duration=1e-1, n_trajectories=200, potential_type='repulsive', rng=rng, 
                                    save_data=True, directoryname=data_dir_noise_level)

    # transform to subtracted space and add noise to each simulation in sim_list
    x_train = []
    for sim in sim_list:
        # subtract data and add noise
        sim.subtract_data()
        sim.add_gaussian_noise(noise_level=noise_level) # just use one noise level for now
        # collect data into list x_train for SINDy fitting
        x_train.append(sim.x)

    # extract labels and time step from last simulation in sim_list
    x_train_labels = sim.labels
    dt = sim.dt
    # fit a SINDy model using different thresholds
    lib = y3.generate_3body_library()
    for threshold in threshold_array:
        opt = ps.STLSQ(threshold=threshold)
        model = ps.SINDy(optimizer=opt, feature_names=x_train_labels, feature_library=lib)
        model.fit(x_train, t=dt, multiple_trajectories=True)
        # print('STLSQ threshold:', threshold)
        # print('Std. dev. of noise:', noise_levels[0])
        # print('complexity:', model.complexity)
        y3.print_SINDy_nice(model)
        print(100*'=')


if __name__ == '__main__':
    load_data('data/basic_noisy')