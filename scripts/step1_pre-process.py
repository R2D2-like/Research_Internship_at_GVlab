import numpy as np
from scipy.signal import butter, sosfilt
from config.values import *

# データのサンプリング周波数とカットオフ周波数を設定
fs = 3000.0  # サンプリング周波数 (Hz)
fc = 110.0   # カットオフ周波数 (Hz)

# バターワースローパスフィルタの設計
sos = butter(N=4, Wn=fc/(fs/2), btype='low', output='sos')


data = np.load('/root/Research_Internship_at_GVlab/real/step1/data/step1_data.npy') #(12, 400, 3)

for i in range(data.shape[0]):
    # フィルタリングされたデータを格納する配列を準備
    filtered_data = np.zeros_like(data[i])

    # 各列に対してローパスフィルタを適用
    for j in range(data[i].shape[1]):
        filtered_data[:, j] = sosfilt(sos, data[i][:, j])

normalized_data = np.zeros_like(filtered_data)

# normalized the filtered data
for i in range(data.shape[2]):
    normalized_data[:, i] = (filtered_data[:, i] - EXPLORATORY_MIN[i]) / (EXPLORATORY_MAX[i] - EXPLORATORY_MIN[i])

# [0, 0.9]に正規化
normalized_data = normalized_data * SCALING_FACTOR

# save the normalized data
np.save('/root/Research_Internship_at_GVlab/real/step1/data/pre-processed_step1_data.npy', normalized_data)

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

