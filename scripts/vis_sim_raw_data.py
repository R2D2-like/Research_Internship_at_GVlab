import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the npy data
data = np.load('/root/Research_Internship_at_GVlab/sim/data/sim_data_3dim.npy')
idx = input('Enter the index of the data you want to visualize: ')
data = data[idx][:, 0:3] #(400, 3)

# 横軸時間，縦軸力（データ）をプロット 
# 各データは0.01秒間隔で取得されていて，t=0.01, 0.02, 0.03, ... となっている
# Fx, Fy, Fzを色分けしてプロット

fig, ax = plt.subplots()
ax.plot(data[:, 0], label='Fx')
ax.plot(data[:, 1], label='Fy')
ax.plot(data[:, 2], label='Fz')
ax.set_xlabel('Time')
ax.set_ylabel('Force')
ax.legend()
plt.show()

# save the plot
fig.savefig('/root/Research_Internship_at_GVlab/sim/data/sim_data_4dim.png')



