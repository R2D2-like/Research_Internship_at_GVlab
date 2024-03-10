import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load the npz data
mode = input('0:ft, 1:position: ')
path = input('Enter path: ')
data_name = path.split('/')[-1].split('.')[0] # e.g. 's0f0'
data = np.load(path)
data = data[data_name]
save_dir = '/root/Research_Internship_at_GVlab/fig/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if mode == '0':
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], label='Fx')
    ax.plot(data[:, 1], label='Fy')
    ax.plot(data[:, 2], label='Fz')
    ax.plot(data[:, 3], label='Tx')
    ax.plot(data[:, 4], label='Ty')
    ax.plot(data[:, 5], label='Tz')

    ax.set_xlabel('Time')
    ax.set_ylabel('Force/Torque')
    ax.legend()
    save_dir = '/root/Research_Internship_at_GVlab/fig/ft/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir + data_name + '.png'
    fig.savefig(save_path)
    plt.show()

elif mode == '1':
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], label='x')
    ax.plot(data[:, 1], label='y')
    ax.plot(data[:, 2], label='z')

    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.legend()
    save_dir = '/root/Research_Internship_at_GVlab/fig/position/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir + data_name + '.png'
    fig.savefig(save_path)
    plt.show()



