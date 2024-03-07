import numpy as np
from scipy.signal import butter, sosfilt

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

for i in range(data.shape[2]):
    filtered_data[:, i] = (filtered_data[:, i] - min_val[i]) / (max_val[i] - min_val[i])

# [0, 0.9]に正規化
normalized_data = filtered_data * 0.9

# save the normalized data
np.save('/root/Research_Internship_at_GVlab/sim/data/pre-processed_sim_data.npy', normalized_data)
