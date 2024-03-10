import numpy as np
from config.values import *
import os

def preprocess(data):
    normalized_data = np.zeros_like(data) # (2000, 9)

    # normalized the data
    for i in range(data.shape[1]):
        if i < 3:
            normalized_data[:, i] = (data[:, i] - DEMO_TRAJECTORY_MIN[i]) / (DEMO_TRAJECTORY_MAX[i] - DEMO_TRAJECTORY_MIN[i])
        else:
            normalized_data[:, i] = (data[:, i] - DEMO_FORCE_TORQUE_MIN[i-3]) / (DEMO_FORCE_TORQUE_MAX[i-3] - DEMO_FORCE_TORQUE_MIN[i-3])

    # [0, 0.9]に正規化
    normalized_data = normalized_data * SCALING_FACTOR

    return normalized_data.T #(9, 2000)


dir = '/root/Research_Internship_at_GVlab/real/step2/data/'

dataset = {}
for sponge in ALL_SPONGES_LIST:
    data = None
    for trial in range(1, DATA_PER_SPONGE+1):
        data_path = dir + sponge + '_' + str(trial) + '.npz'
        if not os.path.exists(data_path):
            print('The data for', sponge, 'does not exist.')
            exit()

        # load data
        pose = np.load(data_path)['pose'] #(2000, 7)
        ft = np.load(data_path)['ft'] #(2000, 6)
        position = pose[:, :3] #(2000, 3)
        # concat position and ft
        position_ft = np.concatenate([position, ft], axis=1) #(2000, 9)

        preprocessed_data = preprocess(position_ft) #(9, 2000)

        # merge the data
        if data is None:
            data = np.expand_dims(preprocessed_data, axis=0) #(1, 9, 2000)
        else:
            data = np.concatenate([data, np.expand_dims(preprocessed_data, axis=0)], axis=0) #(trial, 9, 2000)
    
    dataset[sponge] = data #(DEMO_PER_SPONGE, 9, 2000)

# save as npz
np.savez(dir + 'demo_preprocessed.npz', **dataset)
print('The preprocessed data has been saved at\n', dir + 'demo_preprocessed.npz')