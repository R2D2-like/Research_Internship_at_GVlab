import numpy as np
from config.values import *
import os

dir = '/root/Research_Internship_at_GVlab/data0402/real/step2/data/'
save_dir = '/root/Research_Internship_at_GVlab/data0402/real/step2/data/'

dataset = {}
ft_data_log, z_diff_data_log = None, None

for sponge in TRAIN_SPONGES_LIST:
    ft_data, z_diff_data = None, None
    for trial in range(1, DATA_PER_SPONGE+1):
        # trial = 4
        data_path = dir + sponge + '_' + str(trial) + '.npz'
        if not os.path.exists(data_path):
            print('The data for', sponge, 'does not exist.')
            exit()

        # load data
        pose = np.load(data_path)['pose'] #(100, 7)
        ft = np.load(data_path)['ft'] #(100, 6)
        # 1/20にダウンサンプリング
        pose = pose[::20][20:]#[20:]
        ft = ft[::20][20:]#[20:]
        # z displacement
        z_diff = np.diff(pose[:, 2]) #(99,)

        if z_diff_data is None:
            z_diff_data = np.expand_dims(z_diff, axis=0) #(1, 99)
            ft_data = np.expand_dims(ft, axis=0) #(1, 100, 6)
        else:
            z_diff_data = np.concatenate([z_diff_data, np.expand_dims(z_diff, axis=0)], axis=0) #(trial, 99)
            ft_data = np.concatenate([ft_data, np.expand_dims(ft, axis=0)], axis=0) #(trial, 100, 6)
    print('ft_data', ft_data.shape)
    if ft_data_log is None:
        ft_data_log = ft_data 
        z_diff_data_log = z_diff_data
    else:
        ft_data_log = np.concatenate([ft_data_log, ft_data], axis=0)
        z_diff_data_log = np.concatenate([z_diff_data_log, z_diff_data], axis=0)
    print('ft_data_log', ft_data_log.shape)


    dataset[sponge] = {'z_diff': z_diff_data, 'ft': ft_data}

# calculate the max and min values
min_z_diff, max_z_diff = np.min(z_diff_data_log), np.max(z_diff_data_log)
min_ft, max_ft = [], []
for i in range(ft_data_log.shape[2]):
    max_ft.append(np.max(ft_data_log[:, :, i]))
    min_ft.append(np.min(ft_data_log[:, :, i]))

# normalize the data
z_diff_dataset = {}
ft_dataset = {}
for sponge in TRAIN_SPONGES_LIST:
    normalized_z_diff = (np.array(dataset[sponge]['z_diff']) - min_z_diff) / (max_z_diff - min_z_diff) #(trial, 99)
    normalized_ft = (np.array(dataset[sponge]['ft'] - min_ft)) / (np.array(max_ft) - np.array(min_ft)) #(trial, 100, 6)
    # [0, 0.9]に正規化
    normalized_z_diff = normalized_z_diff * SCALING_FACTOR
    normalized_ft = normalized_ft * SCALING_FACTOR
    # (trial, 100, 6) -> (trial, 6, 100)
    normalized_ft = normalized_ft.transpose(0, 2, 1)
    z_diff_dataset[sponge] = normalized_z_diff
    ft_dataset[sponge] = normalized_ft

# save as npz
np.savez(save_dir + 'demo_preprocessed_z_diff.npz', **z_diff_dataset)
np.savez(save_dir + 'demo_preprocessed_ft.npz', **ft_dataset)
print('Copy the value below and paste it to config/values.py')
print('DEMO_Z_DIFF_MIN =', min_z_diff)
print('DEMO_Z_DIFF_MAX =', max_z_diff)
print('DEMO_FORCE_TORQUE_MIN =', min_ft)
print('DEMO_FORCE_TORQUE_MAX =', max_ft)
