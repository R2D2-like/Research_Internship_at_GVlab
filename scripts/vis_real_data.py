import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load the npz data
mode = input('0:step1, 1:step2, 2:rollout: ')
if mode == '0':
    mode = 'step1'
elif mode == '1':
    mode = 'step2'
else:
    mode = 'rollout'
data_type = input('0:exploratory, 1:trajectory: ')
stiffness = input('stiffness level (1, 2, 3, 4): ')
friction = input('friction level (1, 2, 3): ')
sponge = 's' + stiffness + 'f' + friction
data_dir = '/root/Research_Internship_at_GVlab/real/' + mode + '/data/'
save_dir = '/root/Research_Internship_at_GVlab/fig/' + mode + '/' 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if data_type == '0':
    # Load the npz data
    raw_data_path = data_dir + sponge + '_raw.npz'
    filtered_data_path = data_dir + sponge + '_filtered.npz'
    raw_data = np.load(raw_data_path)[sponge] #(400, 6)
    filtered_data = np.load(filtered_data_path)[sponge] #(400, 6)

    # rawのforce(ax1), rawのtorque(ax2)とfilteredのforce(ax3), filteredのtorque(ax4)を並べてプロット
    fig = plt.figure()
    # rawのforce
    ax1 = fig.add_subplot(221)
    ax1.set_title('Raw Force')
    ax1.plot(raw_data[:, 0], label='Fx')
    ax1.plot(raw_data[:, 1], label='Fy')
    ax1.plot(raw_data[:, 2], label='Fz')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Force')
    ax1.legend()
    # rawのtorque
    ax2 = fig.add_subplot(223)
    ax2.set_title('Raw Torque')
    ax2.plot(raw_data[:, 3], label='Tx')
    ax2.plot(raw_data[:, 4], label='Ty')
    ax2.plot(raw_data[:, 5], label='Tz')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Torque')
    ax2.legend()
    # filteredのforce
    ax3 = fig.add_subplot(222)
    ax3.set_title('Filtered Force')
    ax3.plot(filtered_data[:, 0], label='Fx')
    ax3.plot(filtered_data[:, 1], label='Fy')
    ax3.plot(filtered_data[:, 2], label='Fz')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Force')
    ax3.legend()
    # filteredのtorque
    ax4 = fig.add_subplot(224)
    ax4.set_title('Filtered Torque')
    ax4.plot(filtered_data[:, 3], label='Tx')
    ax4.plot(filtered_data[:, 4], label='Ty')
    ax4.plot(filtered_data[:, 5], label='Tz')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Torque')
    ax4.legend()
    save_path = save_dir + sponge + '_ft.png'
    fig.savefig(save_path)
    plt.show()

elif data_type == '1':
    # Load the npy data
    if mode == 'step2':
        trial = input('trial (1, 2, 3, 5, 6): ')
        data_path = data_dir + sponge + '_' + trial + '.npz'
        save_path = save_dir + sponge + '_trajectory_' + trial + '.png'
    elif mode == 'rollout':
        data_path = data_dir + sponge + '.npz'
        save_path = save_dir + sponge + '_trajectory.png'
    pose_data = np.load(data_path)['pose']
    ft_data = np.load(data_path)['ft']
    # position と orientation と force と torque を並べてプロット
    fig = plt.figure()
    # position
    ax1 = fig.add_subplot(221)
    ax1.set_title('Position')
    ax1.plot(pose_data[:, 0], label='x')
    ax1.plot(pose_data[:, 1], label='y')
    ax1.plot(pose_data[:, 2], label='z')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position')
    ax1.legend()
    # orientation
    ax2 = fig.add_subplot(222)
    ax2.set_title('Orientation')
    ax2.plot(pose_data[:, 3], label='x')
    ax2.plot(pose_data[:, 4], label='y')
    ax2.plot(pose_data[:, 5], label='z')
    ax2.plot(pose_data[:, 6], label='w')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Orientation')
    ax2.legend()
    # force
    ax3 = fig.add_subplot(223)
    ax3.set_title('Force')
    ax3.plot(ft_data[:, 0], label='Fx')
    ax3.plot(ft_data[:, 1], label='Fy')
    ax3.plot(ft_data[:, 2], label='Fz')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Force')
    ax3.legend()
    # torque
    ax4 = fig.add_subplot(224)
    ax4.set_title('Torque')
    ax4.plot(ft_data[:, 3], label='Tx')
    ax4.plot(ft_data[:, 4], label='Ty')
    ax4.plot(ft_data[:, 5], label='Tz')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Torque')
    ax4.legend()
    fig.savefig(save_path)
    plt.show()    
    



