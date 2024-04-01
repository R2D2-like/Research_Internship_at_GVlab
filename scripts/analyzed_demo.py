import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


sponge = input('sponge:')
trial = input('trial:')
data_path = '/root/Research_Internship_at_GVlab/data0313/real/step2/data/{}_{}.npz'.format(sponge, trial)
  
pose_data = np.load(data_path)['pose']
print(pose_data.shape)
ft_data = np.load(data_path)['ft']
# ft_data = ft_data[1460::20,:] # with baseline_white_lowtable
# ft_data = ft_data[:500:20,:] # with baseline_black_lowtable
# ft_data = ft_data[1130:1630:20,:] # with baseline_lowtable
# ft_data = ft_data[830:1330:20,:] # with baseline_hightable

print(ft_data.shape)

# count fz < 0
count = 0
sum = 0
for i in range(ft_data.shape[0]):
    if ft_data[i][2] < 0:
        count += 1
    sum += ft_data[i][2]
print('{}/{}'.format(count, ft_data.shape[0]))
print('mean fz: {}'.format(sum/ft_data.shape[0]))
# print('sum fz: {}'.format(sum))
print('std fz: {}'.format(np.std(ft_data[:,2])))





