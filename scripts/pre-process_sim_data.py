import numpy as np
from scipy.signal import butter, sosfilt
from config.values import *
import os

# データのサンプリング周波数とカットオフ周波数を設定
fs = 3000.0  # サンプリング周波数 (Hz)
fc = 50.0   # カットオフ周波数 (Hz)

# バターワースローパスフィルタの設計
sos = butter(N=4, Wn=fc/(fs/2), btype='low', output='sos')
sos_t = butter(N=4, Wn=fc/(fs/2), btype='low', output='sos')

dir = '/root/Research_Internship_at_GVlab/data0402/sim/data/'
if not os.path.exists(dir):
    os.makedirs(dir)

data_path = dir + 'sim_data.npy'
data = np.load(data_path) #(1000, 400, 6)

# convert to adapt to real robot configuration
data *= -1
# # idx 0 と 1 を入れ替え
# data = np.concatenate([data[:, :, 1:2], data[:, :, 0:1], data[:, :, 2:]], axis=2)
# # idx 3 と 4 を入れ替え
# data = np.concatenate([data[:, :, :3], data[:, :, 4:5], data[:, :, 3:4], data[:, :, 5:]], axis=2)
# # idx 3 と 4 の200ステップ〜400ステップの符号を反転
# data[:, 200:, 3:5] *= -1
# data[:, :, 3:5] *= -1
filtered_data = np.zeros_like(data)

# 各列に対してローパスフィルタを適用
for i in range(data.shape[0]):
    for j in range(data.shape[2]):
        if j < 3:
            filtered_data[i][:200, j] = sosfilt(sos, data[i][:200, j])
            filtered_data[i][200:, j] = sosfilt(sos, data[i][200:, j])
        else:
            filtered_data[i][:200, j] = sosfilt(sos_t, data[i][:200, j])
            filtered_data[i][200:, j] = sosfilt(sos_t, data[i][200:, j])
            


# normalized the filtered data
max_val = []
min_val = []

for i in range(filtered_data.shape[2]):
    max_val.append(np.max(filtered_data[:, :, i]))
    min_val.append(np.min(filtered_data[:, :, i]))

normalized_data = np.zeros_like(filtered_data) #(1000, 400, 3)

for i in range(data.shape[0]):
    for j in range(data.shape[2]):
        normalized_data[i][:, j] = (filtered_data[i][:, j] - min_val[j]) / (max_val[j] - min_val[j])

# [0, 0.9]に正規化
normalized_data *= SCALING_FACTOR

# save the normalized data
save_path = dir + 'sim_filtered.npy'
np.save(save_path, filtered_data)
save_path = dir + 'sim_preprocessed.npy'
np.save(save_path, normalized_data)
print('Data is saved at\n', save_path)
print('Copy the value below and paste it to config/values.py')
print('EXPLORATORY_MIN =', min_val)
print('EXPLORATORY_MAX =', max_val)

