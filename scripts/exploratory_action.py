import numpy as np
import robosuite as suite
from robosuite.environments.base import register_env
from robosuite.controllers import load_controller_config
import copy
from exploratory_task import ExploratoryTask
import xml.etree.ElementTree as ET
from robosuite.utils.buffers import RingBuffer


FORCE_TORQUE_SAMPLING_RATE = 100  # Hz
DATA_NUM = 1000
DATA_SAVE_PATH = 'sim_data.npy'
GRIPPER_PATH = '/root/external/robosuite/robosuite/models/assets/grippers/wiping_cylinder_gripper.xml'      

# def move_end_effector(env, obs, direction, speed, duration):
#     """
#     Move the end-effector in a specified direction at a specified speed for a certain duration, while recording data.
#     """
#     data_recorder = []
#     observable_steps = int(duration * FORCE_TORQUE_SAMPLING_RATE)  # Use new rate for recording
#     control_steps = int(duration * env.control_freq)  # Control steps as per control_freq
#     step_ratio = observable_steps // control_steps  # Calculate how many observables per control step

#     for _ in range(control_steps):
#         action = np.zeros(env.action_dim)
#         action[:3] = 4 *direction * speed # 4 is a scaling factor for mapping action to velocity(m/s)
#         obs, _, _, _ = env.step(action)

#         # Record force and torque data at the higher frequency
#         for _ in range(step_ratio):
#             force = env.robots[0].ee_force
#             torque = env.robots[0].ee_torque
#             force_torque = np.concatenate([force, torque])

#             data_recorder.append(force_torque)
#         env.render()

#     return data_recorder, obs

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
    obs_sampling_freq = env.control_freq * 5  # Example: 5 times the control frequency
    env.modify_observable(
        observable_name="robot0_eef_force",
        attribute="filter",
        modifier=filter_fcn_force
    )
    env.modify_observable(
        observable_name="robot0_eef_force",
        attribute="sampling_rate",
        modifier=obs_sampling_freq
    )
    env.modify_observable(
        observable_name="robot0_eef_torque",
        attribute="filter",
        modifier=filter_fcn_torque
    )
    env.modify_observable(
        observable_name="robot0_eef_torque",
        attribute="sampling_rate",
        modifier=obs_sampling_freq
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
        # print(force[i].reshape(1, 3))
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

def move_end_effector_to_sponge(env, obs, target_position, threshold=0.01):
    """
    エンドエフェクタを特定の位置に移動させる関数。
    :param env: robosuite環境
    :param target_position: 目的の位置 [x, y, z]
    :param threshold: 目的の位置に到達したとみなす距離の閾値
    """
    for _ in range(1000):  # 最大1000ステップ
        current_position = obs['robot0_eef_pos']  # 現在のエンドエフェクタの位置を取得
        action = target_pos2action(target_position, env.robots[0], obs, 2, 0)
        displacement = target_position - current_position  # 目的の位置までの変位を計算
        # action = np.zeros(env.action_dim)  # 初期化
        # action[:3] = displacement  # 変位に基づいてアクションを設定(15倍はなんとなく）

        obs, reward, done, info = env.step(action)
        env.render()
        
        # エンドエフェクタが目標位置に十分近いかをチェック
        if np.linalg.norm(displacement) < threshold and env.is_eef_touching_sponge():
            print('現在の位置',current_position)
            print("目標位置に到達しました。")
            return obs
        
        # 接触情報を取得
        if env.is_eef_touching_sponge():
            print("スポンジに触れました。")
            print('現在の位置',current_position)
            print('スポンジの位置',obs['sponge_pos'])
            print('current joint position', env.robots[0]._joint_positions)
            return obs

def is_eef_touching_sponge(env):
    """
    エンドエフェクタがスポンジに触れているかを判定する関数。
    """
    if env.is_eef_touching_sponge():
        print("Touching スポンジに触れています。")
        return True
    else:
        print("WARNING スポンジに触れていません。")
        return False


def main():
    register_env(ExploratoryTask)

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
        lateral_friction = np.random.uniform(6, 12.0)
        spin_friction = np.random.uniform(6, 12.0)
        rolling_friction = 12
        print('friction:', lateral_friction, spin_friction, rolling_friction) 

        # Set friction of geom
        for geom in root.iter('geom'):
            if 'friction' in geom.attrib:
                # geom.set("friction", f"{lateral_friction} {spin_friction} {rolling_friction}")
                geom.set("friction", f"{10} {10} {rolling_friction}")
        # write to xml
        tree.write(GRIPPER_PATH)

        roop_count += 1
        env = suite.make(
            env_name="ExploratoryTask",
            robots="UR5e",
            controller_configs=controller_config,
            gripper_types="WipingCylinderGripper",  # Specify the gripper
            has_renderer=True,  # No on-screen rendering
            has_offscreen_renderer=False,  # Enable off-screen rendering
            use_camera_obs=False,  # Do not use camera observations
            use_object_obs=True,  # Use object observations
            reward_shaping=True,  # Enable reward shaping
            control_freq=20,  # 100Hz control for the robot
            # render_camera='sideview',  # Specify camera type
            sampling_rate=FORCE_TORQUE_SAMPLING_RATE,  # 100Hz observation sampling
            horizon=1000,  # 200 timesteps per episode
            ignore_done=True,  # Never terminate the environment
        )
        print('Environment created')
        # print('control_timestep is ', env.control_timestep)

        # シミュレーションをリセットし、スポンジの位置にエンドエフェクタを移動する
        obs = env.reset()
        sponge_pos = obs['sponge_pos']
        print('スポンジの位置：', sponge_pos)
        sponge_pos += np.array([0.04, 0.02, 0])

        # env.robots[0].set_robot_joint_positions([-0.19138369, -0.74438986,  1.71753443, -2.52349095, -1.63481779, -1.70707145])
        # env.robots[0].set_robot_joint_positions([-0.23268999, -0.77181136,  1.95386971, -2.70821434, -1.59245526, -1.75497473])
    
        obs = move_end_effector_to_sponge(env, obs, sponge_pos)

        print('current_position', obs['robot0_eef_pos'])
        # Move end-effector downwards (pressing) at 0.01 m/s for 2 seconds
        data_recorder_p, obs = move_end_effector(env, direction=np.array([0, 0, -1]), speed=0.01, duration=2)
        print('current_position', obs['robot0_eef_pos'])
        if not is_eef_touching_sponge(env):
            # continue
            pass
        print('data_recorder_p ({}, {})'.format(len(data_recorder_p), data_recorder_p[0].shape)) #(200, 6)

        # Move end-effector to the right (lateral motion) at 0.05 m/s for 1 second
        data_recorder_l_1, obs = move_end_effector(env, direction=np.array([0, 1, 0]), speed=0.05, duration=1)
        print('current_position', obs['robot0_eef_pos'])
        if not is_eef_touching_sponge(env):
            # continue
            pass
        print('data_recorder_l_1 ({}, {})'.format(len(data_recorder_l_1), data_recorder_l_1[0].shape)) #(100, 6)
        # Move end-effector to the left (lateral motion) at -0.05 m/s for 1 second
        data_recorder_l_2, obs = move_end_effector(env, direction=np.array([0, -1, 0]), speed=0.05, duration=1)
        print('current_position', obs['robot0_eef_pos'])
        if not is_eef_touching_sponge(env):
            # continue
            pass
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