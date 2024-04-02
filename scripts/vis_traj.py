import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

sponge = input('sponge: ')
height = input('height: ')
# eef_positionだけプロットする
data_path = '/root/Research_Internship_at_GVlab/data0402/real/rollout/data/proposed/predicted/' + sponge +'_' + height+ '_0402.npz'
save_path = '/root/Research_Internship_at_GVlab/data0402/fig/'+ sponge +'_' + height+ '_0402.png'
eef_position_data =  np.load(data_path)['eef_position']
print(eef_position_data.shape)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('End-effector Position')
ax1.plot(eef_position_data[:, 0], label='x')
ax1.plot(eef_position_data[:, 1], label='y')
ax1.plot(eef_position_data[:, 2], label='z')
ax1.set_xlabel('Time')
ax1.set_ylabel('Position')
ax1.legend()
fig.savefig(save_path)
plt.show()