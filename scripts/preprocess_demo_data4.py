import numpy as np
from config.values import *
import os

dir = '/root/Research_Internship_at_GVlab/real/step2/data/'

dataset = {}
for sponge in TRAIN_SPONGES_LIST:
    ft_data, z_diff_data = None, None
    for trial in range(1, DATA_PER_SPONGE+1):
        data_path = dir + sponge + '_' + str(trial) + '.npz'
        if not os.path.exists(data_path):
            print('The data for', sponge, 'does not exist.')
            exit()

        # load data
        pose = np.load(data_path)['pose'] #(100, 7)
        ft = np.load(data_path)['ft'] #(100, 6)
        # 1/20にダウンサンプリング
        pose = pose[::20]
        ft = ft[::20]
        # z displacement
        z_diff = np.diff(pose[:, 2]) #(99,)

        if z_diff_data is None:
            z_diff_data = np.expand_dims(z_diff, axis=0) #(1, 99)
            ft_data = np.expand_dims(ft, axis=0) #(1, 100, 6)
        else:
            z_diff_data = np.concatenate([z_diff_data, np.expand_dims(z_diff, axis=0)], axis=0) #(trial, 99)
            ft_data = np.concatenate([ft_data, np.expand_dims(ft, axis=0)], axis=0) #(trial, 100, 6)
    dataset[sponge] = {'z_diff': z_diff_data, 'ft': ft_data}

# calculate the max and min values
min_z_diff, max_z_diff = np.min(z_diff_data), np.max(z_diff_data)
min_ft, max_ft = [], []
for i in range(ft_data.shape[2]):
    max_ft.append(np.max(ft_data[:, :, i]))
    min_ft.append(np.min(ft_data[:, :, i]))

# normalize the data
normalized_dataset = {}
for sponge in TRAIN_SPONGES_LIST:
    normalized_z_diff = (dataset[sponge]['z_diff'] - min_z_diff) / (max_z_diff - min_z_diff) #(trial, 99)
    normalized_ft = (dataset[sponge]['ft'] - min_ft) / (max_ft - min_ft) #(trial, 100, 6)
    # [0, 0.9]に正規化
    normalized_z_diff = normalized_z_diff * SCALING_FACTOR
    normalized_ft = normalized_ft * SCALING_FACTOR
    # (trial, 100, 6) -> (trial, 6, 100)
    normalized_ft = normalized_ft.transpose(0, 2, 1)
    normalized_dataset[sponge] = {'z_diff': normalized_z_diff, 'ft': normalized_ft}

# save as npz
np.savez(dir + 'demo_preprocessed4.npz', **normalized_dataset)
print('Copy the value below and paste it to config/values.py')
print('DEMO_Z_DIFF_MIN =', min_z_diff)
print('DEMO_Z_DIFF_MAX =', max_z_diff)
print('DEMO_FORCE_TORQUE_MIN =', min_ft)
print('DEMO_FORCE_TORQUE_MAX =', max_ft)
