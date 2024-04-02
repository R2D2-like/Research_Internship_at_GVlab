import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load the npz data
env = input('0:sim, 1:real: ')
if env == '0':
    idx = int(input('idx: '))
    raw_data_path = '/root/Research_Internship_at_GVlab/sim/data/sim_data_3dim_0309.npy' #(100, 400, 6)
    filtered_data_path = '/root/Research_Internship_at_GVlab/sim/data/sim_filtered.npy' #(100, 400, 6)
    preprocessed_data_path = '/root/Research_Internship_at_GVlab/sim/data/sim_preprocessed.npy' #(100, 400, 6)
    raw_data = np.load(raw_data_path)[idx]
    filtered_data = np.load(filtered_data_path)[idx]
    preprocessed_data = np.load(preprocessed_data_path)[idx]

    save_dir = '/root/Research_Internship_at_GVlab/fig/sim/vis/'
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
    save_dir = '/root/Research_Internship_at_GVlab/data0402/fig/' + mode + '/normal_80/' 
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
            height = input('low high slope:')
            save_name = sponge + '_' + height 
            method = input('0:baseline, 1:proposed: ')
            p_or_r = input('0:predicted, 1:result: ')
            if method == '0':
                if p_or_r == '0':
                    # data_path = data_dir + 'baseline/predicted/' + sponge + '.npz'
                    data_path = '/root/Research_Internship_at_GVlab/data0402/real/rollout/data/normal/baseline/predicted/' + save_name +'.npz'
                    save_dir = save_dir + 'baseline/predicted/'
                else:
                    # data_path = data_dir + 'baseline/result/' + sponge + '.npz'
                    data_path = '/root/Research_Internship_at_GVlab/real/rollout/data/baseline/result/baseline_hightable.npz'
                    data_path = '/root/Research_Internship_at_GVlab/data0402/real/rollout/data/normal/baseline/result/' + save_name +'.npz'
                    save_dir = save_dir + 'baseline/result/'
            else:
                if p_or_r == '0':
                    data_path = '/root/Research_Internship_at_GVlab/data0402/real/rollout/data/normal/proposed/predicted/' + save_name +'.npz'
                    save_dir = save_dir + 'proposed/predicted/'
                else:
                    data_path = '/root/Research_Internship_at_GVlab/data0402/real/rollout/data/normal/proposed/result/' + save_name +'.npz'
                    save_dir = save_dir + 'proposed/result/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = save_dir + save_name +'.png'
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
            # ft_data = ft_data[1460::20,:] # with baseline_white_lowtable
            # ft_data = ft_data[:500:20,:] # with baseline_black_lowtable
            print(ft_data.shape)
            # ft_data = ft_data[1130:1630:20,:] # with baseline_lowtable
            # ft_data = ft_data[830:1330:20,:] # with baseline_hightable
            # position と orientation と force と torque を並べてプロット
            fig = plt.figure()
            # position
            ax1 = fig.add_subplot(211) 
            ax1.set_title('Vertical position of the robot\'s end-effector')
            # ax1.plot(pose_data[:, 0], label='x')
            # ax1.plot(pose_data[:, 1], label='y')
            plt.tight_layout()
            ax1.set_ylim(-0.05, 0.13)
            # ax1.set_ylim(-0.05, 0.125)
            # ax1.set_ylim(0.75, 0.85)
            # ax1.plot(pose_data[:, 0], label='x')
            ax1.plot(pose_data[:, 2], label='z')
            ax1.set_xlabel('Time step')
            ax1.set_ylabel('Vertical position (m)')
            ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
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


            ax3 = fig.add_subplot(212)
            ax3.set_title('Robot\'s end-effector contact force')
            # ax3.set_ylim(-65,65)
            ax3.set_ylim(-100,35)
            ax3.plot(ft_data[:, 0], label='Fx', color='blue', linestyle = 'dotted')
            ax3.plot(ft_data[:, 1], label='Fy', color='green', linestyle = 'dotted')
            ax3.plot(ft_data[:, 2], label='Fz', color='red')
            # ax3.set_xlabel('Time')
            # ax3.set_ylabel('Force')
            # ax3.legend()
            # # torque
            # ax4 = fig.add_subplot(224)
            # ax4.set_title('Torque')

            ax3.set_xlabel('Time step')
            # left side ylabel title
            ax3.set_ylabel('Force (N)')
            # right side for torque
            ax4 = ax3.twinx()
            # right side ylabel title
            ax4.set_ylabel('Torque (Nm)')
            # set ylim for right side ylabel
            ax4.set_ylim(-14.3,5)
            # plot torque in right side ylabel change color 
            ax4.plot(ft_data[:, 3], label='Tx', color='orange', linestyle = 'dotted')
            ax4.plot(ft_data[:, 4], label='Ty', color='purple', linestyle = 'dotted')
            ax4.plot(ft_data[:, 5], label='Tz', color='brown', linestyle = 'dotted')
            # fix the position of the legend containig both force and torque
            ax3.legend(ax3.get_lines() + ax4.get_lines(), [l.get_label() for l in ax3.get_lines() + ax4.get_lines()], loc='upper right', bbox_to_anchor=(1, 1))
            # add space between subplots
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            # add space in order not to overlap the ylabel and legend
            # plt.tight_layout(rect=[0,0,1,0.96])
            plt.savefig(save_path, bbox_inches='tight')
            # count fz < 0
            count = 0
            sum = 0
            for i in range(ft_data.shape[0]):
                if ft_data[i][2] < 0:
                    count += 1
                sum += ft_data[i][2]
            print('{}/{}'.format(count, ft_data.shape[0]))
            print('mean fz: {}'.format(sum/ft_data.shape[0]))
            print('sum fz: {}'.format(sum))
            print('std fz: {}'.format(np.std(ft_data[:,2])))
            plt.show()    
        



