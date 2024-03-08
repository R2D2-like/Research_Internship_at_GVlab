import numpy as np
import robosuite as suite
from robosuite.environments.base import register_env
from robosuite.controllers import load_controller_config
from exploratory_task import ExploratoryTask
import xml.etree.ElementTree as ET
from robosuite.utils.buffers import RingBuffer


FORCE_TORQUE_SAMPLING_RATE = 100  # Hz
DATA_NUM = 1000
DATA_SAVE_PATH = 'sim_data.npy'
GRIPPER_PATH = '/root/external/robosuite/robosuite/models/assets/grippers/sponge_gripper.xml'   
global FORCE_OFFSET 
global TORQUE_OFFSET 
SOLIMP_NAME = [
    'wiping_surface1a',
    'wiping_surface1b',
    'wiping_surface1c',
    'wiping_surface1d',
    'wiping_surface1e',
    'wiping_surface1f',
    'wiping_surface1g',
    'wiping_surface2a',
    'wiping_surface2b',
    'wiping_surface2c',
    'wiping_surface2d',
    'wiping_surface2e',
    'wiping_surface2f',
    'wiping_surface2g',
]

def select_number():
    # 0.02から0.2の間の範囲と0.2から0.3の間の範囲の重みを決定
    weights = [0.8, 0.2]  # 0.02から0.2までの範囲が全体の80%、0.2から0.3までが20%
    
    # どの範囲から数値を選択するか決定
    choice = np.random.choice(['0.02-0.2', '0.2-0.3'], p=weights)
    
    # 決定された範囲から数値をランダムに選択
    if choice == '0.02-0.2':
        return round(np.random.uniform(0.02, 0.2), 4)
    else:
        return round(np.random.uniform(0.2, 0.3), 4)
    
def move_end_effector(env, direction, speed, duration):
    """
    Move the end-effector in a specified direction at a specified speed for a certain duration, while recording force and torque data at a higher frequency.
    """
    # Initialize RingBuffer for force and torque data
    buffer_length = int(duration * FORCE_TORQUE_SAMPLING_RATE)  
    force_buffer = RingBuffer(dim=3, length=buffer_length)  
    torque_buffer = RingBuffer(dim=3, length=buffer_length)  

    # Define filter function to record force and torque data
    def filter_fcn_force(corrupted_value):
        force_buffer.push(corrupted_value)
        return corrupted_value
    def filter_fcn_torque(corrupted_value):
        torque_buffer.push(corrupted_value)
        return corrupted_value

    # Modify force and torque observables to use the filter function and increase their sampling rates
    env.modify_observable(
        observable_name="robot0_eef_force",
        attribute="filter",
        modifier=filter_fcn_force
    )
    env.modify_observable(
        observable_name="robot0_eef_force",
        attribute="sampling_rate",
        modifier=FORCE_TORQUE_SAMPLING_RATE
    )
    env.modify_observable(
        observable_name="robot0_eef_torque",
        attribute="filter",
        modifier=filter_fcn_torque
    )
    env.modify_observable(
        observable_name="robot0_eef_torque",
        attribute="sampling_rate",
        modifier=FORCE_TORQUE_SAMPLING_RATE
    )

    control_steps = int(duration * env.control_freq)
    for _ in range(control_steps):
        action = np.zeros(env.action_dim)
        scaling_factor = 15  if duration==2 else 12.5
        action[:3] = scaling_factor * direction * speed  # Adjust action based on the environment's requirements
        obs, _, _, _ = env.step(action)
        env.render()

    # Retrieve recorded data from buffer
    force = [force_buffer.buf[i] for i in range(force_buffer._size)] #(buffer_length, 3)
    if len(force) < 3:
        return None, None
    global FORCE_OFFSET
    force -= FORCE_OFFSET
    # for i in range(len(force)):
    #     print(force[i].reshape(1, 3)[0])
    #     # print('current_position', obs['robot0_eef_pos'])
    #     if force[i][2] > 50:
    #         print('force', force[i].reshape(1, 3))
    #         print('current_position', obs['robot0_eef_pos'])
    #         # print('displacement', touch_pos - obs['robot0_eef_pos'][2])
    # print('obs_force', obs['robot0_eef_force'])
    # print('current_position', obs['robot0_eef_pos'])
    print('force average', np.mean(force, axis=0))
    torque = [torque_buffer.buf[i] for i in range(torque_buffer._size)] #(buffer_length, 3)
    global TORQUE_OFFSET
    torque -= TORQUE_OFFSET
    print('torque average', np.mean(torque, axis=0))
    data_recorder = np.concatenate([force, torque], axis=1) #(buffer_length, 6)

    return data_recorder, obs

def target_pos2action(target_pos, robot, obs, kp, kd):
    
    action = [0 for _ in range(robot.dof)]
    current_pos = obs['robot0_eef_pos'] # 現在のエンドエフェクタの位置を取得
    current_vel = obs['robot0_eef_vel_lin'] # 現在のエンドエフェクタの速度を取得

    for i in range(3):
        action[i] = (target_pos[i] - current_pos[i]) * kp - current_vel[i] * kd
    
    return action

def move_end_effector_to_table(env, obs, target_position, reset=False):
    """
    エンドエフェクタを特定の位置に移動させる関数。
    :param env: robosuite環境
    :param target_position: 目的の位置 [x, y, z]
    :param threshold: 目的の位置に到達したとみなす距離の閾値
    """
    force_log, torque_log = [], []
    for _ in range(1000):  # 最大1000ステップ
        # print('current_position', obs['robot0_eef_pos'])
        # print('force', env.robots[0].ee_force)
        action = target_pos2action(target_position, env.robots[0], obs, 2, 0)
        obs, reward, done, info = env.step(action)
        env.render()

        if reset and obs['robot0_eef_pos'][2] < 1.02:
            global FORCE_OFFSET
            FORCE_OFFSET = np.mean(force_log[:10], axis=0)
            global TORQUE_OFFSET
            TORQUE_OFFSET = np.mean(torque_log[:10], axis=0)
        
        # 接触情報を取得
        if obs['robot0_eef_pos'][2] < 1.00 :
            print("テーブルに触れました。")
            print('現在の位置',obs['robot0_eef_pos'])
            print('force', env.robots[0].ee_force)
            return obs
        
        force_log.append(obs['robot0_eef_force'])
        torque_log.append(obs['robot0_eef_torque'])

def is_eef_touching_table(obs):
    """
    エンドエフェクタがスポンジに触れているかを判定する関数。
    """
    if obs['robot0_contact'] or obs['robot0_eef_pos'][2] < 1.0:
        print("テーブルに触れています。")
        print('force', obs['robot0_eef_force'])
        print('current_position', obs['robot0_eef_pos'])
        return True
    else:
        print("テーブルに触れていません。")
        print('force', obs['robot0_eef_force'])
        print('current_position', obs['robot0_eef_pos'])
        return False

def main():
    register_env(ExploratoryTask)  # Register the custom environment

    # Instantiate the environment with the UR5e robot 
    controller_config = load_controller_config(default_controller="OSC_POSE")
    data_recorder_3dim = None
    data_recorder_4dim = None

    num = 0

    roop_count = 0
    while num < DATA_NUM:
        # load gripper xml file 
        tree = ET.parse(GRIPPER_PATH)
        root = tree.getroot()
        # Sampling friction values
        lateral_friction = np.random.uniform(0, 3.5)
        spin_friction = 0.5
        rolling_friction = 0.0001
        print('friction:', lateral_friction, spin_friction, rolling_friction) 

        # solimp
        width = select_number()

        # Set friction of geom
        for geom in root.iter('geom'):
            if 'friction' in geom.attrib:
                geom.set("friction", f"{lateral_friction} {spin_friction} {rolling_friction}")
            if 'solref' in geom.attrib and geom.attrib['type'] == 'box':
                geom.set("solref", f"{-0.5} -1" )
            if 'solimp' in geom.attrib and geom.attrib['name'] in SOLIMP_NAME:
                geom.set("solimp", "0.001 0.9 {}".format(width))
        # write to xml
        tree.write(GRIPPER_PATH)

        roop_count += 1
        env = suite.make(
            env_name="ExploratoryTask",#"ExploratoryTaskFixed",
            robots="UR5e",
            controller_configs=controller_config,
            gripper_types="SpongeGripper",  # Specify the gripper
            has_renderer=True,  # No on-screen rendering
            has_offscreen_renderer=False,  # Enable off-screen rendering
            use_camera_obs=False,  # Do not use camera observations
            use_object_obs=True,  # Use object observations
            reward_shaping=True,  # Enable reward shaping
            control_freq=20,  # 100Hz control for the robot
            # render_camera='sideview',  # Specify camera type
            horizon=1000,  # 200 timesteps per episode
            ignore_done=True,  # Never terminate the environment
        )
        print('Environment created')
        # print('control_timestep is ', env.control_timestep)

        # シミュレーションをリセットし、スポンジの位置にエンドエフェクタを移動する
        obs = env.reset()
        # table_pos = env.model.mujoco_arena.table_top_abs
        # print('table_pos', table_pos) # 0.9

        # env.robots[0].set_robot_joint_positions([-0.19138369, -0.74438986,  1.71753443, -2.52349095, -1.63481779, -1.70707145])
        # env.robots[0].set_robot_joint_positions([-0.23268999, -0.77181136,  1.95386971, -2.70821434, -1.59245526, -1.75497473])
    
        obs = move_end_effector_to_table(env, obs, [0.1, 0.0, 0.97], reset=True)

        print('current_position', obs['robot0_eef_pos'])
        # Move end-effector downwards (pressing) at 0.01 m/s for 2 seconds
        data_recorder_p, obs = move_end_effector(env, direction=np.array([0, 0, -1]), speed=0.01, duration=2)
        print('current_position', obs['robot0_eef_pos'])
        is_eef_touching_table(obs)

        print('data_recorder_p ({}, {})'.format(len(data_recorder_p), data_recorder_p[0].shape)) #(200, 6)

        # detach from the table
        _, obs = move_end_effector(env, direction=np.array([0, 0, 1]), speed=0.01, duration=2)
        # move to the table
        obs = move_end_effector_to_table(env, obs, [0.1, 0.0, 0.97], reset=False)

        # Move end-effector to the right (lateral motion) at 0.05 m/s for 1 second
        data_recorder_l_1, obs = move_end_effector(env, direction=np.array([0, 1, 0]), speed=0.05, duration=1)
        print('current_position', obs['robot0_eef_pos'])
        is_eef_touching_table(obs)

        print('data_recorder_l_1 ({}, {})'.format(len(data_recorder_l_1), data_recorder_l_1[0].shape)) #(100, 6)
        # Move end-effector to the left (lateral motion) at -0.05 m/s for 1 second
        data_recorder_l_2, obs = move_end_effector(env, direction=np.array([0, -1, 0]), speed=0.05, duration=1)
        print('current_position', obs['robot0_eef_pos'])
        is_eef_touching_table(obs)

        print('data_recorder_l_2 ({}, {})'.format(len(data_recorder_l_2), data_recorder_l_2[0].shape)) #(100, 6)

        data_recorder_l = np.concatenate([data_recorder_l_1, data_recorder_l_2], axis=0) 
        print('data_recorder_l', data_recorder_l.shape) #(200, 6)

        tmp = np.concatenate([data_recorder_p, data_recorder_l], axis=0) #(400, 6)
        tmp = np.expand_dims(tmp, axis=0) #(1, 400, 6)
        if data_recorder_3dim is None:
            data_recorder_3dim = tmp
        else:
            data_recorder_3dim = np.concatenate([data_recorder_3dim, tmp], axis=0)
        print('data_recorder_3dim', data_recorder_3dim.shape) #(idx+1, 400, 6)

        tmp = np.array([data_recorder_p, data_recorder_l]) #(2, 200, 6) 
        tmp = np.expand_dims(tmp, axis=0) #(1, 2, 200, 6)
        if data_recorder_4dim is None:
            data_recorder_4dim = tmp
        else:
            data_recorder_4dim = np.concatenate([data_recorder_4dim, tmp], axis=0)
        print('data_recorder_4dim', data_recorder_4dim.shape) #(idx+1, 2, 200, 6)

        # Close the environment
        env.close()
        num += 1

    print('roop_count', roop_count)

    

    # Save the recorded data
    print('Saving the recorded data...')
    print('data_recorder_3dim', data_recorder_3dim.shape) #(1000, 400, 6)
    np.save('/root/Research_Internship_at_GVlab/sim/data/sim_data_3dim.npy', data_recorder_3dim)
    print('data_recorder_4dim', data_recorder_4dim.shape) #(1000, 2, 200, 6)
    np.save('/root/Research_Internship_at_GVlab/sim/data/sim_data_4dim.npy', data_recorder_4dim)
    print('Data recorded and saved.')

if __name__ == "__main__":
    main()