import numpy as np
from config.values import *
import os
from scipy.signal import butter, sosfilt


def preprocess(data):
    # データのサンプリング周波数とカットオフ周波数を設定
    fs = 3000.0  # サンプリング周波数 (Hz)
    fc = 110.0   # カットオフ周波数 (Hz)

    # バターワースローパスフィルタの設計
    sos = butter(N=4, Wn=fc/(fs/2), btype='low', output='sos')

    # フィルタリングされたデータを格納する配列を準備
    filtered_data = np.zeros_like(data) #(400, 6)

    # 各列に対してローパスフィルタを適用
    for i in range(data.shape[1]): # i=0,1,2,3,4,5
        filtered_data[:, i] = sosfilt(sos, data[:, i])

    normalized_data = np.zeros_like(filtered_data)

    # normalized the filtered data
    for i in range(data.shape[1]):
        normalized_data[:, i] = (filtered_data[:, i] - EXPLORATORY_MIN[i]) / (EXPLORATORY_MAX[i] - EXPLORATORY_MIN[i])

    # [0, 0.9]に正規化
    normalized_data = normalized_data * SCALING_FACTOR

    return filtered_data, normalized_data


mode = input('0:step1, 1:rollout: ')
if mode == '0':
    # Load the npy data
    dir = '/root/Research_Internship_at_GVlab/real/step1/data/'
else:
    dir = '/root/Research_Internship_at_GVlab/real/rollout/data/exploratory/'
    DATA_PER_SPONGE = 1

raw_dataset, filtered_dataset, preprocessed_dataset = {}, {}, {}
for sponge in ALL_SPONGES_LIST:
    raw_data, filtered_data, preprocessed_data = None, None, None
    for trial in range(1, DATA_PER_SPONGE+1):
        if mode == '0':
            pressing_path = dir + 'pressing/' + sponge + '_' + str(trial) + '.npz'
            lateral_path = dir + 'lateral/' + sponge + '_' + str(trial) + '.npz'
        else:
            pressing_path = dir + 'pressing/' + sponge + '.npz'
            lateral_path = dir + 'lateral/' + sponge + '.npz'
        if not os.path.exists(pressing_path) or not os.path.exists(lateral_path):
            print('The data for', sponge, 'does not exist.')
            continue

        # load data
        pressing_data = np.load(pressing_path)[sponge] #(200, 6)
        lateral_data = np.load(lateral_path)[sponge] #(200, 6)

        # merge and preprocess the data
        merged = np.concatenate([pressing_data, lateral_data], axis=0) #(400, 6)
        filtered, preprocessed = preprocess(merged) #(400, 6)

        if raw_data is None:
            raw_data = np.expand_dims(merged, axis=0) #(1, 400, 6)
        else:
            raw_data = np.concatenate([raw_data, np.expand_dims(merged, axis=0)], axis=0)

        if filtered_data is None:
            filtered_data = np.expand_dims(filtered, axis=0)
        else:
            filtered_data = np.concatenate([filtered_data, np.expand_dims(filtered, axis=0)], axis=0)

        if preprocessed_data is None:
            preprocessed_data = np.expand_dims(preprocessed, axis=0)
        else:
            preprocessed_data = np.concatenate([preprocessed_data, np.expand_dims(preprocessed, axis=0)], axis=0)
        

    raw_dataset[sponge] = raw_data #(DEMO_PER_SPONGE, 400, 6)
    filtered_dataset[sponge] = filtered_data #(DEMO_PER_SPONGE, 400, 6)
    preprocessed_dataset[sponge] = preprocessed_data #(DEMO_PER_SPONGE, 400, 6)

# save as npz
raw_data_save_path = dir + 'exploratory_action_raw.npz' 
np.savez(raw_data_save_path, **raw_dataset)
print('The raw dataset has been saved at\n', raw_data_save_path)

filtered_data_save_path = dir + 'exploratory_action_filtered.npz'
np.savez(filtered_data_save_path, **filtered_dataset)
print('The filtered dataset has been saved at\n', filtered_data_save_path)

preprocessed_data_save_path = dir + 'exploratory_action_preprocessed.npz'
np.savez(preprocessed_data_save_path, **preprocessed_dataset)
print('The preprocessed dataset has been saved at\n', preprocessed_data_save_path)

