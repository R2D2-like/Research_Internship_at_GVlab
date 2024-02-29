import numpy as np
import robosuite as suite
from robosuite.environments.base import register_env
from robosuite.controllers import load_controller_config
from exploratory_task_fixed import ExploratoryTaskFixed
import xml.etree.ElementTree as ET
from robosuite.utils.buffers import RingBuffer


FORCE_TORQUE_SAMPLING_RATE = 100  # Hz
DATA_NUM = 1000
DATA_SAVE_PATH = 'sim_data.npy'
GRIPPER_PATH = '/root/external/robosuite/robosuite/models/assets/grippers/sponge_gripper.xml'      

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
        action[:3] = 15 * direction * speed  # Adjust action based on the environment's requirements
        obs, _, _, _ = env.step(action)
        env.render()

    # Retrieve recorded data from buffer
    force = [force_buffer.buf[i] for i in range(force_buffer._size)] #(buffer_length, 3)
    # for i in range(len(force)):
    #     print(force[i].reshape(1, 3))
    # print('obs_force', obs['robot0_eef_force'])
    torque = [torque_buffer.buf[i] for i in range(torque_buffer._size)] #(buffer_length, 3)
    data_recorder = np.concatenate([force, torque], axis=1) #(buffer_length, 6)

    return data_recorder, obs

def target_pos2action(target_pos, robot, obs, kp, kd):
    
    action = [0 for _ in range(robot.dof)]
    current_pos = obs['robot0_eef_pos'] # 現在のエンドエフェクタの位置を取得
    current_vel = obs['robot0_eef_vel_lin'] # 現在のエンドエフェクタの速度を取得

    for i in range(3):
        action[i] = (target_pos[i] - current_pos[i]) * kp - current_vel[i] * kd
    
    return action

def move_end_effector_to_table(env, obs, target_position, threshold=0.01):
    """
    エンドエフェクタを特定の位置に移動させる関数。
    :param env: robosuite環境
    :param target_position: 目的の位置 [x, y, z]
    :param threshold: 目的の位置に到達したとみなす距離の閾値
    """
    for _ in range(1000):  # 最大1000ステップ
        # print('current_position', current_position)
        # print('force', env.robots[0].ee_force)
        action = target_pos2action(target_position, env.robots[0], obs, 2, 0)
        obs, reward, done, info = env.step(action)
        env.render()
        
        # 接触情報を取得
        if is_eef_touching_table(env):
            print("テーブルに触れました。")
            print('現在の位置',obs['robot0_eef_pos'])
            # print('current joint position', env.robots[0]._joint_positions)
            return obs

def is_eef_touching_table(env):
    """
    エンドエフェクタがスポンジに触れているかを判定する関数。
    """
    if env.robots[0].ee_force[2] > 0.0:
        print("Touching テーブルに触れています。")
        print('force', env.robots[0].ee_force)
        return True
    else:
        print("WARNING テーブルに触れていません。")
        print('force', env.robots[0].ee_force)
        return False

def main():
    register_env(ExploratoryTaskFixed)

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
        lateral_friction = np.random.uniform(0, 2)
        spin_friction = np.random.uniform(0, 2)
        rolling_friction = 0.0001
        print('friction:', lateral_friction, spin_friction, rolling_friction) 
        # solref
        solref = np.random.uniform(0.001, 0.1)
        print('solref: {}, 1'.format(solref))

        # Set friction of geom
        for geom in root.iter('geom'):
            if 'friction' in geom.attrib:
                geom.set("friction", f"{lateral_friction} {spin_friction} {rolling_friction}")
            if 'solref' in geom.attrib and geom.attrib['type'] == 'box':
                geom.set("solref", f"{solref} 1" )
        # write to xml
        tree.write(GRIPPER_PATH)

        roop_count += 1
        env = suite.make(
            env_name="ExploratoryTaskFixed",
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
        table_pos = env.model.mujoco_arena.table_top_abs
        table_pos[2] -= 0.2

        # env.robots[0].set_robot_joint_positions([-0.19138369, -0.74438986,  1.71753443, -2.52349095, -1.63481779, -1.70707145])
        # env.robots[0].set_robot_joint_positions([-0.23268999, -0.77181136,  1.95386971, -2.70821434, -1.59245526, -1.75497473])
    
        obs = move_end_effector_to_table(env, obs, table_pos)

        print('current_position', obs['robot0_eef_pos'])
        # Move end-effector downwards (pressing) at 0.01 m/s for 2 seconds
        data_recorder_p, obs = move_end_effector(env, direction=np.array([0, 0, -1]), speed=0.01, duration=2)
        print('current_position', obs['robot0_eef_pos'])
        is_eef_touching_table(env)

        print('data_recorder_p ({}, {})'.format(len(data_recorder_p), data_recorder_p[0].shape)) #(200, 6)

        # Move end-effector to the right (lateral motion) at 0.05 m/s for 1 second
        data_recorder_l_1, obs = move_end_effector(env, direction=np.array([0, 1, 0]), speed=0.05, duration=1)
        print('current_position', obs['robot0_eef_pos'])
        is_eef_touching_table(env)

        print('data_recorder_l_1 ({}, {})'.format(len(data_recorder_l_1), data_recorder_l_1[0].shape)) #(100, 6)
        # Move end-effector to the left (lateral motion) at -0.05 m/s for 1 second
        data_recorder_l_2, obs = move_end_effector(env, direction=np.array([0, -1, 0]), speed=0.05, duration=1)
        print('current_position', obs['robot0_eef_pos'])
        is_eef_touching_table(env)

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
    np.save('sim_data_3dim.npy', data_recorder_3dim)
    print('data_recorder_4dim', data_recorder_4dim.shape) #(1000, 2, 200, 6)
    np.save('sim_data_4dim.npy', data_recorder_4dim)
    print('Data recorded and saved.')

if __name__ == "__main__":
    main()