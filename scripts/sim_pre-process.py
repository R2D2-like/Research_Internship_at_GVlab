import numpy as np
from scipy.signal import butter, sosfilt
from config.values import *

# データのサンプリング周波数とカットオフ周波数を設定
fs = 3000.0  # サンプリング周波数 (Hz)
fc = 110.0   # カットオフ周波数 (Hz)

# バターワースローパスフィルタの設計
sos = butter(N=4, Wn=fc/(fs/2), btype='low', output='sos')


data = np.load('/root/Research_Internship_at_GVlab/sim/data/sim_data_3dim.npy') #(1000, 400, 3)

for i in range(1000):
    # フィルタリングされたデータを格納する配列を準備
    filtered_data = np.zeros_like(data[i])

    # 各列に対してローパスフィルタを適用
    for j in range(data[i].shape[1]):
        filtered_data[:, j] = sosfilt(sos, data[i][:, j])

# normalized the filtered data
max_val = []
min_val = []

for i in range(data.shape[2]):
    max_val.append(np.max(data[:, :, i]))
    min_val.append(np.min(data[:, :, i]))

normalized_data = np.zeros_like(filtered_data)

for i in range(data.shape[2]):
    normalized_data[:, i] = (filtered_data[:, i] - min_val[i]) / (max_val[i] - min_val[i])

# [0, 0.9]に正規化
normalized_data = normalized_data * SCALING_FACTOR

# save the normalized data
np.save('/root/Research_Internship_at_GVlab/sim/data/pre-processed_sim_data.npy', normalized_data)
print('copy the value below and paste it to config/values.py')
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