import numpy as np
from scipy.signal import butter, sosfilt
from config.values import *

# データのサンプリング周波数とカットオフ周波数を設定
fs = 3000.0  # サンプリング周波数 (Hz)
fc = 110.0   # カットオフ周波数 (Hz)

# バターワースローパスフィルタの設計
sos = butter(N=4, Wn=fc/(fs/2), btype='low', output='sos')

mode = input('0:trajectory data, 1:trajectory + ft data: ')
if mode == '0':
    data = np.load('/root/Research_Internship_at_GVlab/real/step2/data/step2_traj.npy') #(N, 2000, 3)
else:
    data = np.load('/root/Research_Internship_at_GVlab/real/step2/data/step2_traj_ft.npy') #(N, 2000, 9)

for i in range(data.shape[0]):
    # フィルタリングされたデータを格納する配列を準備
    filtered_data = np.zeros_like(data[i])

    # 各列に対してローパスフィルタを適用
    for j in range(data[i].shape[1]):
        filtered_data[:, j] = sosfilt(sos, data[i][:, j])

# normalized the filtered data
traj_max_val = []
traj_min_val = []
ft_max_val = []
ft_min_val = []

for i in range(data.shape[2]):
    if i < 3:
        traj_max_val.append(np.max(data[:, :, i]))
        traj_min_val.append(np.min(data[:, :, i]))
    else:
        ft_max_val.append(np.max(data[:, :, i]))
        ft_min_val.append(np.min(data[:, :, i]))

normalized_data = np.zeros_like(filtered_data)

for i in range(data.shape[2]):
    if i < 3:
        normalized_data[:, i] = (filtered_data[:, i] - traj_min_val[i]) / (traj_max_val[i] - traj_min_val[i])
    else:
        normalized_data[:, i] = (filtered_data[:, i] - ft_min_val[i-3]) / (ft_max_val[i-3] - ft_min_val[i-3])

# [0, 0.9]に正規化
normalized_data = normalized_data * SCALING_FACTOR

# save the normalized data
if mode == '0':
    np.save('/root/Research_Internship_at_GVlab/real/step2/data/pre-processed_step2_traj_data.npy', normalized_data)
elif mode == '1':
    np.save('/root/Research_Internship_at_GVlab/real/step2/data/pre-processed_step2_traj_ft_data.npy', normalized_data)
print('copy the value below and paste it to config/values.py')
print('DEMO_TRAJECTORY_MIN =', traj_min_val)
print('DEMO_TRAJECTORY_MAX =', traj_max_val)
if mode == '1':
    print('DEMO_FORCE_TORQUE_MIN =', ft_min_val)
    print('DEMO_FORCE_TORQUE_MAX =', ft_max_val)

# フィルタリング後のデータをプロット
import matplotlib.pyplot as plt
data = filtered_data[0, 0:3] 

fig, ax = plt.subplots()
ax.plot(data[:, 0], label='x')
ax.plot(data[:, 1], label='y')
ax.plot(data[:, 2], label='z')
ax.set_xlabel('Time')
ax.set_ylabel('Position')
ax.legend()
plt.show()
