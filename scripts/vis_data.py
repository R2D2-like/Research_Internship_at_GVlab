import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load the npz data
env = input('0:sim, 1:real: ')
if env == '0':
    idx = int(input('idx: '))
    raw_data_path = '/root/Research_Internship_at_GVlab/sim/data/sim_data_3dim.npy' #(100, 400, 6)
    filtered_data_path = '/root/Research_Internship_at_GVlab/sim/data/sim_filtered.npy' #(100, 400, 6)
    preprocessed_data_path = '/root/Research_Internship_at_GVlab/sim/data/sim_preprocessed.npy' #(100, 400, 6)
    raw_data = np.load(raw_data_path)[idx]
    filtered_data = np.load(filtered_data_path)[idx]
    preprocessed_data = np.load(preprocessed_data_path)[idx]

    save_dir = '/root/Research_Internship_at_GVlab/fig/sim/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # rawのforce(ax1), rawのtorque(ax2)とfilteredのforce(ax3), filteredのtorque(ax4)とpreprocessedのforce(ax5), preprocessedのtorque(ax6)を並べてプロット
    fig = plt.figure()
    # rawのforce
    ax1 = fig.add_subplot(231)
    ax1.set_title('Raw Force')
    ax1.plot(raw_data[:, 0], label='Fx')
    ax1.plot(raw_data[:, 1], label='Fy')
    ax1.plot(raw_data[:, 2], label='Fz')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Force')
    ax1.legend()
    # rawのtorque
    ax2 = fig.add_subplot(234)
    ax2.set_title('Raw Torque')
    ax2.plot(raw_data[:, 3], label='Tx')
    ax2.plot(raw_data[:, 4], label='Ty')
    ax2.plot(raw_data[:, 5], label='Tz')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Torque')
    ax2.legend()
    # filteredのforce
    ax3 = fig.add_subplot(232)
    ax3.set_title('Filtered Force')
    ax3.plot(filtered_data[:, 0], label='Fx')
    ax3.plot(filtered_data[:, 1], label='Fy')
    ax3.plot(filtered_data[:, 2], label='Fz')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Force')
    ax3.legend()
    # filteredのtorque
    ax4 = fig.add_subplot(235)
    ax4.set_title('Filtered Torque')
    ax4.plot(filtered_data[:, 3], label='Tx')
    ax4.plot(filtered_data[:, 4], label='Ty')
    ax4.plot(filtered_data[:, 5], label='Tz')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Torque')
    ax4.legend()
    # preprocessedのforce
    ax5 = fig.add_subplot(233)
    ax5.set_title('Preprocessed Force')
    ax5.plot(preprocessed_data[:, 0], label='Fx')
    ax5.plot(preprocessed_data[:, 1], label='Fy')
    ax5.plot(preprocessed_data[:, 2], label='Fz')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Force')
    ax5.legend()
    # preprocessedのtorque
    ax6 = fig.add_subplot(236)
    ax6.set_title('Preprocessed Torque')
    ax6.plot(preprocessed_data[:, 3], label='Tx')
    ax6.plot(preprocessed_data[:, 4], label='Ty')
    ax6.plot(preprocessed_data[:, 5], label='Tz')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Torque')
    ax6.legend()
    save_path = save_dir + 'sim_' + str(idx) + '_ft.png'
    fig.savefig(save_path)
    plt.show()
    
else:
    mode = input('0:step1, 1:step2, 2:rollout: ')
    if mode == '0':
        mode = 'step1'
    elif mode == '1':
        mode = 'step2'
    else:
        mode = 'rollout'
    data_dir = '/root/Research_Internship_at_GVlab/real/' + mode + '/data/'
    data_type = input('0:exploratory, 1:trajectory: ')
    stiffness = input('stiffness level (1, 2, 3, 4): ')
    friction = input('friction level (1, 2, 3): ')
    sponge = 's' + stiffness + 'f' + friction
    save_dir = '/root/Research_Internship_at_GVlab/fig/' + mode + '/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if data_type == '0':
        # Load the npz data
        if mode == 'step1':
            trial = int(input('trial (1, 2, 3, 5, 6, 7, 8): ')) - 1
        elif mode == 'rollout':
            trial = 0
            data_dir += 'exploratory/'
        raw_data_path = data_dir + 'exploratory_action_raw.npz'
        filtered_data_path = data_dir + 'exploratory_action_filtered.npz'
        raw_data = np.load(raw_data_path)[sponge][trial] #(400, 6)
        filtered_data = np.load(filtered_data_path)[sponge][trial] #(400, 6)

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
        p_or_r = None
        # Load the npy data
        if mode == 'step2':
            trial = input('trial (1, 2, 3, 5, 6, 7, 8): ')
            data_path = data_dir + sponge + '_' + trial + '.npz'
            save_path = save_dir + sponge + '_trajectory_' + trial + '.png'
        elif mode == 'rollout':
            method = input('0:baseline, 1:proposed: ')
            p_or_r = input('0:predicted, 1:result: ')
            if method == '0':
                if p_or_r == '0':
                    # data_path = data_dir + 'baseline/predicted/' + sponge + '.npz'
                    data_path = '/root/Research_Internship_at_GVlab/real/rollout/data/baseline/predicted/baseline_hightable.npz'
                    save_dir = save_dir + 'baseline/predicted/'
                else:
                    # data_path = data_dir + 'baseline/result/' + sponge + '.npz'
                    data_path = '/root/Research_Internship_at_GVlab/real/rollout/data/baseline/result/baseline_hightable.npz'
                    save_dir = save_dir + 'baseline/result/'
            else:
                if p_or_r == '0':
                    data_path = '/root/Research_Internship_at_GVlab/real/rollout/data/proposed/predicted/proposed_hightable_predicted.npz'
                    save_dir = save_dir + 'proposed/predicted/'
                else:
                    data_path = '/root/Research_Internship_at_GVlab/real/rollout/data/proposed/result/proposed_hightable_result.npz'
                    save_dir = save_dir + 'proposed/result/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = save_dir + 'baseline_hightable.png'
        if p_or_r == '0':
            # eef_positionだけプロットする
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
        else:    
            pose_data = np.load(data_path)['pose']
            print(pose_data.shape)
            ft_data = np.load(data_path)['ft']
            print(ft_data.shape)
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
            # # orientation
            # ax2 = fig.add_subplot(222)
            # ax2.set_title('Orientation')
            # ax2.plot(pose_data[:, 3], label='x')
            # ax2.plot(pose_data[:, 4], label='y')
            # ax2.plot(pose_data[:, 5], label='z')
            # ax2.plot(pose_data[:, 6], label='w')
            # ax2.set_xlabel('Time')
            # ax2.set_ylabel('Orientation')
            # ax2.legend()
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
        



