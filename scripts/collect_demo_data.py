"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

from exploratory_task import ExploratoryTask
import xml.etree.ElementTree as ET

FORCE_TORQUE_SAMPLING_RATE = 100  # Hz
GRIPPER_PATH = '/root/external/robosuite/robosuite/models/assets/grippers/wiping_cylinder_gripper.xml' 
DATA_SAVE_PATH = '/root/Research_Internship_at_GVlab/demo_data/demo_data.npy' 

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
        action = target_pos2action(target_position, env.robots[0], obs, 1, 0)
        displacement = target_position - current_position  # 目的の位置までの変位を計算
        # action = np.zeros(env.action_dim)  # 初期化
        # action[:3] = displacement  # 変位に基づいてアクションを設定(15倍はなんとなく）

        obs, reward, done, info = env.step(action)
        env.render()
        
        # エンドエフェクタが目標位置に十分近いかをチェック
        if np.linalg.norm(displacement) < threshold and env.is_eef_touching_sponge():
            print('現在の位置',current_position)
            print("目標位置に到達しました。")
            break
        
        # 接触情報を取得
        if env.is_eef_touching_sponge():
            print("スポンジに触れました。")
            print('現在の位置',current_position)
            print('スポンジの位置',obs['sponge_pos'])
            break

def collect_human_trajectory(env, device, arm, env_configuration):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    obs = env.reset()
    sponge_pos = obs['sponge_pos']
    print('スポンジの位置：', sponge_pos)
    data_recorder = None

    env.robots[0].set_robot_joint_positions([-0.19138369, -0.74438986,  1.71753443, -2.52349095, -1.63481779, -1.70707145])

    move_end_effector_to_sponge(env, obs, sponge_pos)

    print('keyboard control start!')

    device.start_control()

    # Loop for 2000 time steps
    for _ in range(2000):
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # Get the newest action
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm=arm, env_configuration=env_configuration
        )

        # If action is none, then this a reset so we should break
        # if action is None:
        #     break

        # Run environment step
        obs, reward, done, info = env.step(action)
        env.render()

        # record end effector position (x, y, z)
        current_position = obs['robot0_eef_pos']
        if data_recorder is None:
            data_recorder = np.array(current_position)
        else:
            data_recorder = np.concatenate([data_recorder, current_position])

    # cleanup for end of data collection episodes
    env.close()

    return data_recorder

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "tmp/{}".format(str(time.time()).replace(".", "_"))

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
            geom.set("friction", f"{lateral_friction} {spin_friction} {rolling_friction}")
    # write to xml
    tree.write(GRIPPER_PATH)

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
            sampling_rate=FORCE_TORQUE_SAMPLING_RATE,  # 100Hz observation sampling
            horizon=1000,  # 200 timesteps per episode
            ignore_done=True,  # Never terminate the environment
        )

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    trial_num = 1
    data_recorder = None
    # collect demonstrations util pressed ctr+c
    try:
        while True:
            print('starting trial No.:', trial_num)
            data = collect_human_trajectory(env, device, args.arm, args.config) #(2000, 3)
            if data_recorder is None:
                data_recorder = data
            else:
                data = np.expand_dims(data, axis=0) #(1, 2000, 3)
                data_recorder = np.concatenate([data_recorder, data], axis=0) #(trial_num, 2000, 3)

            trial_num += 1
    except KeyboardInterrupt:
        print("Data collection stopped")
    
    print("Data is saved in: ", DATA_SAVE_PATH)
    np.save(DATA_SAVE_PATH, data_recorder)
    

