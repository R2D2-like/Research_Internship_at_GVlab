import numpy as np
from scipy.signal import butter, sosfilt
from config.values import *
import os

# データのサンプリング周波数とカットオフ周波数を設定
fs = 3000.0  # サンプリング周波数 (Hz)
fc = 110.0   # カットオフ周波数 (Hz)

# バターワースローパスフィルタの設計
sos = butter(N=4, Wn=fc/(fs/2), btype='low', output='sos')

dir = '/root/Research_Internship_at_GVlab/sim/data/'
if not os.path.exists(dir):
    os.makedirs(dir)

data_path = dir + 'sim_data_3dim.npy'
data = np.load(data_path) #(1000, 400, 3)
filtered_data = np.zeros_like(data)

# 各列に対してローパスフィルタを適用
for i in range(data.shape[0]):
    for j in range(data.shape[2]):
        filtered_data[i][:, j] = sosfilt(sos, data[i][:, j])

# normalized the filtered data
max_val = []
min_val = []

for i in range(data.shape[2]):
    max_val.append(np.max(data[:, :, i]))
    min_val.append(np.min(data[:, :, i]))

normalized_data = np.zeros_like(filtered_data) #(1000, 400, 3)

for i in range(data.shape[0]):
    for j in range(data.shape[2]):
        normalized_data[i][:, j] = (filtered_data[i][:, j] - min_val[j]) / (max_val[j] - min_val[j])

# [0, 0.9]に正規化
normalized_data *= SCALING_FACTOR

# save the normalized data
save_path = dir + 'sim_preprocessed.npy'
np.save(save_path, normalized_data)
print('Data is saved at\n', save_path)
print('Copy the value below and paste it to config/values.py')
print('EXPLORATORY_MIN =', min_val)
print('EXPLORATORY_MAX =', max_val)

# フィルタリング後のデータをプロット
import matplotlib.pyplot as plt
data = filtered_data[0, 0:3] 

fig, ax = plt.subplots()
ax.plot(data[:, 0], label='Fx')
ax.plot(data[:, 1], label='Fy')
ax.plot(data[:, 2], label='Fz')
ax.set_xlabel('Time')
ax.set_ylabel('Force')
ax.legend()
plt.show()