import numpy as np
import robosuite as suite
from robosuite.environments.base import register_env
from robosuite.controllers import load_controller_config
import copy
from exploratory_task import ExploratoryTask

FORCE_TORQUE_SAMPLING_RATE = 100  # Hz
DATA_NUM = 1000
DATA_SAVE_PATH = 'sim_data.npy'

def move_end_effector(env, direction, speed, duration):
    """
    Move the end-effector in a specified direction at a specified speed for a certain duration, while recording data.
    """
    data_recorder = []
    observable_steps = int(duration * FORCE_TORQUE_SAMPLING_RATE)  # Use new rate for recording
    control_steps = int(duration * env.control_freq)  # Control steps as per control_freq
    step_ratio = observable_steps // control_steps  # Calculate how many observables per control step

    for step in range(control_steps):
        action = np.zeros(env.action_dim)
        action[:3] = direction * speed
        obs, _, _, _ = env.step(action)

        # Record force and torque data at the higher frequency
        for _ in range(step_ratio):
            force = env.robots[0].ee_force
            torque = env.robots[0].ee_torque
            force_torque = np.concatenate([force, torque])

            data_recorder.append(force_torque)
        env.render()

    return data_recorder, obs

def move_end_effector_to_position(env, obs, target_position, threshold=0.001):
    """
    エンドエフェクタを特定の位置に移動させる関数。
    :param env: robosuite環境
    :param target_position: 目的の位置 [x, y, z]
    :param threshold: 目的の位置に到達したとみなす距離の閾値
    """
    for _ in range(1000):  # 最大1000ステップ
        current_position = obs['robot0_eef_pos']  # 現在のエンドエフェクタの位置を取得
        displacement = target_position - current_position  # 目的の位置までの変位を計算
        action = np.zeros(env.action_dim)  # 初期化
        action[:3] = displacement  # 変位に基づいてアクションを設定(15倍はなんとなく）

        obs, reward, done, info = env.step(action)
        env.render()
        
        # エンドエフェクタが目標位置に十分近いかをチェック
        if np.linalg.norm(displacement) < threshold:
            print('現在の位置',current_position)
            print("目標位置に到達しました。")
            break
        
        # 接触情報を取得
        if env.is_eef_touching_sponge():
            print("スポンジに触れました。")
            print('現在の位置',current_position)
            print('スポンジの位置',obs['sponge_pos'])
            break

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

    while num < DATA_NUM:
        env = suite.make(
            env_name="ExploratoryTask",
            robots="UR5e",
            controller_configs=controller_config,
            gripper_types="WipingGripper",  # Specify the gripper
            has_renderer=True,  # No on-screen rendering
            has_offscreen_renderer=False,  # Enable off-screen rendering
            use_camera_obs=False,  # Do not use camera observations
            use_object_obs=True,  # Use object observations
            reward_shaping=True,  # Enable reward shaping
            control_freq=20,  # 100Hz control for the robot
            sampling_rate=FORCE_TORQUE_SAMPLING_RATE,  # 100Hz observation sampling
            horizon=200,  # 200 timesteps per episode
            ignore_done=True,  # Never terminate the environment
        )
        print('Environment created')
        # print('control_timestep is ', env.control_timestep)

        # シミュレーションをリセットし、スポンジの位置にエンドエフェクタを移動する
        obs = env.reset()
        sponge_pos = obs['sponge_pos']
        print('スポンジの位置：', sponge_pos)
    
        move_end_effector_to_position(env, obs, sponge_pos)

        # Move end-effector downwards (pressing) at 0.01 m/s for 2 seconds
        data_recorder_p, obs = move_end_effector(env, direction=np.array([0, 0, -1]), speed=0.01, duration=2)
        if not is_eef_touching_sponge(env):
            continue
        print('data_recorder_p ({}, {})'.format(len(data_recorder_p), data_recorder_p[0].shape)) #(200, 6)

        # Move end-effector to the right (lateral motion) at 0.05 m/s for 1 second
        data_recorder_l_1, _ = move_end_effector(env, direction=np.array([0, 1, 0]), speed=0.05, duration=1)
        if not is_eef_touching_sponge(env):
            continue
        print('data_recorder_l_1 ({}, {})'.format(len(data_recorder_l_1), data_recorder_l_1[0].shape)) #(100, 6)
        # Move end-effector to the left (lateral motion) at -0.05 m/s for 1 second
        data_recorder_l_2, _ = move_end_effector(env, direction=np.array([0, -1, 0]), speed=0.05, duration=1)
        if not is_eef_touching_sponge(env):
            continue
        print('data_recorder_l_2 ({}, {})'.format(len(data_recorder_l_2), data_recorder_l_2[0].shape)) #(100, 6)

        data_recorder_l = np.concatenate([data_recorder_l_1, data_recorder_l_2], axis=0) 
        print('data_recorder_l', data_recorder_l.shape) #(200, 6)

        tmp = np.concatenate([data_recorder_p, data_recorder_l], axis=0)
        print('tmp', tmp.shape) #(400, 6)
        tmp = np.expand_dims(tmp, axis=0)
        print('tmp', tmp.shape) #(1, 400, 6)
        if data_recorder_3dim is None:
            data_recorder_3dim = tmp
        else:
            data_recorder_3dim = np.concatenate([data_recorder_3dim, tmp], axis=0)
        print('data_recorder_3dim', data_recorder_3dim.shape) #(idx+1, 400, 6)

        tmp = np.array([data_recorder_p, data_recorder_l])
        print('tmp', tmp.shape) #(2, 200, 6) 
        tmp = np.expand_dims(tmp, axis=0)
        print('tmp', tmp.shape) #(1, 2, 200, 6)
        if data_recorder_4dim is None:
            data_recorder_4dim = tmp
        else:
            data_recorder_4dim = np.concatenate([data_recorder_4dim, tmp], axis=0)
        print('data_recorder_4dim', data_recorder_4dim.shape) #(idx+1, 2, 200, 6)

        # Close the environment
        env.close()
        num += 1

    

    # Save the recorded data
    print('Saving the recorded data...')
    print('data_recorder_3dim', data_recorder_3dim.shape) #(1000, 400, 6)
    np.save('sim_data_3dim.npy', data_recorder_3dim)
    print('data_recorder_4dim', data_recorder_4dim.shape) #(1000, 2, 200, 6)
    np.save('sim_data_4dim.npy', data_recorder_4dim)
    print('Data recorded and saved.')

if __name__ == "__main__":
    main()