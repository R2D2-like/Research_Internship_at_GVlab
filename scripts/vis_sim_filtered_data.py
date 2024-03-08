import numpy as np
from scipy.signal import butter, sosfilt

# データのサンプリング周波数とカットオフ周波数を設定
fs = 3000.0  # サンプリング周波数 (Hz)
fc = 110.0   # カットオフ周波数 (Hz)

# バターワースローパスフィルタの設計
sos = butter(N=4, Wn=fc/(fs/2), btype='low', output='sos')


data = np.load('/root/Research_Internship_at_GVlab/sim/data/sim_data_3dim.npy')
idx = input('Enter the index of the data you want to visualize: ')
data = data[idx]

# フィルタリングされたデータを格納する配列を準備
filtered_data = np.zeros_like(data)

# 各列に対してローパスフィルタを適用
for i in range(data.shape[1]):
    filtered_data[:, i] = sosfilt(sos, data[:, i])

# フィルタリング後のデータをプロット
import matplotlib.pyplot as plt
data = filtered_data[:, 0:3] #(400, 3)

fig, ax = plt.subplots()
ax.plot(data[:, 0], label='Fx')
ax.plot(data[:, 1], label='Fy')
ax.plot(data[:, 2], label='Fz')
ax.set_xlabel('Time')
ax.set_ylabel('Force')
ax.legend()
plt.show()

