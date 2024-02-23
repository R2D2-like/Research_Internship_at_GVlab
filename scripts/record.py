from collections import OrderedDict

import numpy as np
import robosuite as suite

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from table_with_sponge_arena import TableWithSpongeArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, TEXTURE_FILES
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.base import register_env
from robosuite.controllers import load_controller_config
from sponge import SpongeObject
import time
from robosuite.models.objects import BoxObject
from robosuite.models.arenas.table_arena import TableArena
import copy

def move_end_effector(env, direction, speed, duration, data_recorder):
    """
    Move the end-effector in a specified direction at a specified speed for a certain duration, while recording data.
    """
    if data_recorder is not None:
        data_recorder_res = copy.deepcopy(data_recorder)
    else:
        data_recorder_res = []
    num_steps = int(duration * env.control_freq)  # Convert duration to number of steps
    for _ in range(num_steps):
        action = np.zeros(env.action_dim)
        action[:3] = direction * speed
        obs, _, _, _ = env.step(action)

        # Record force and torque data
        force = env.robots[0].ee_force
        torque = env.robots[0].ee_torque
        force_torque = np.concatenate([force, torque])

        data_recorder_res.append(force_torque)
        env.render()

    return data_recorder_res, obs

# def move_end_effector(env, direction, speed, duration, data_recorder):
#     """
#     Move the end-effector in a specified direction at a specified speed for a certain duration, while recording data.
#     :param env: The robosuite environment.
#     :param direction: The direction to move (np.array).
#     :param speed: The speed to move at (m/s).
#     :param duration: The duration to move for (seconds).
#     :param _data_recorder: List to record force and torque data.
#     """
#     if data_recorder is not None:
#         data_recorder_res = copy.deepcopy(data_recorder)
#     else:
#         data_recorder_res = []
#     start_time = time.time()
#     while time.time() - start_time < duration:
#         action = np.zeros(6)  # [dx, dy, dz, droll, dpitch, dyaw, grasp]
#         action[:3] = direction * speed  # Set movement direction and speed
#         obs, _, _, _ = env.step(action)
        
#         # Record force and torque data (force_ee)
#         force = env.robots[0].ee_force
#         torque =  env.robots[0].ee_torque
#         force_torque = np.concatenate([force, torque])

#         data_recorder_res.append(force_torque)
        
#         env.render()
        
#     return data_recorder_res


class ExploratoryTask(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (sponge) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        sponge_size=(0.05, 0.05, 0.025),
        sponge_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.sponge_size = sponge_size
        self.sponge_friction = sponge_friction

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the sponge is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the sponge
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the sponge
            - Lifting: in {0, 1}, non-zero if arm has lifted the sponge

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            sponge_pos = self.sim.data.body_xpos[self.sponge_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - sponge_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.sponge):
                reward += 0.25

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Create arena
        mujoco_arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction, table_offset=self.table_offset)

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.sponge = BoxObject(
            name="sponge",
            size=self.sponge_size,
            density=500,
            friction=self.sponge_friction,
            material=CustomMaterial(texture="Sponge",\
                                    tex_name="Sponge", mat_name="Sponge"),
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.sponge)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.sponge,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.sponge,
        )


    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.sponge_body_id = self.sim.model.body_name2id(self.sponge.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # sponge-related observables
            @sensor(modality=modality)
            def sponge_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.sponge_body_id])

            @sensor(modality=modality)
            def sponge_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.sponge_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_sponge_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["sponge_pos"]
                    if f"{pf}eef_pos" in obs_cache and "sponge_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [sponge_pos, sponge_quat, gripper_to_sponge_pos]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the sponge.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the sponge
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.sponge)

    def _check_success(self):
        """
        Check if sponge has been lifted.

        Returns:
            bool: True if sponge has been lifted
        """
        sponge_height = self.sim.data.body_xpos[self.sponge_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        # sponge is higher than the table top above a margin
        return sponge_height > table_height + 0.04
    

    def is_eef_touching_sponge(self):
        """
        Check if the end effector is touching the sponge.

        Returns:
            bool: True if the end effector is touching the sponge
        """
        return self.check_contact(self.robots[0].gripper, self.sponge)
        # return self.check_contact(self.robots[0].robot_model, self.sponge)
    
 
register_env(ExploratoryTask)


# Instantiate the environment with the UR5e robot 
controller_config = load_controller_config(default_controller="OSC_POSE")

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
    control_freq=100,  # 100Hz control for the robot
    horizon=200,  # 200 timesteps per episode
    ignore_done=True,  # Never terminate the environment
)

print(env.control_timestep)

# printout the sponge position
print('Sponge position:', env.sim.data.body_xpos[env.sponge_body_id])


print('Environment created')

env.reset()

'''
Action space low: [-1. -1. -1.]
Action space high: [1. 1. 1.]
'''


# エンドエフェクタの初期位置を指定する
# spongeの位置を取得
sponge_pos = env.sim.data.body_xpos[env.sponge_body_id]
print('sponge',sponge_pos)
# spongeのサイズを取得
sponge_size = env.sponge_size
table_hight = env.model.mujoco_arena.table_offset[2]
print('table_hight',table_hight)


# シミュレーションをリセットし、初期位置にエンドエフェクタを移動する
obs = env.reset()
sponge_pos = obs['sponge_pos']
print('sponge',sponge_pos)
initial_pos = sponge_pos# + np.array([0, 0, sponge_size[2]])
initial_pos[0] += 0.2
initial_orientation = np.array([0, 0, 0])
initial_action = np.concatenate([initial_pos, initial_orientation])

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
        action[:3] = 10*displacement  # 変位に基づいてアクションを設定(15倍はなんとなく）

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
move_end_effector_to_position(env, obs, initial_pos)
# # 1秒間レンダリングしながら静止する
# for _ in range(100):
#     env.render()
#     time.sleep(0.01)

data_recorder = []

# Move end-effector downwards (pressing) at 0.01 m/s for 2 seconds
data_recorder, obs = move_end_effector(env, direction=np.array([0, 0, -1]), speed=0.01, duration=2, data_recorder=data_recorder)
print('data_recorder',len(data_recorder), data_recorder[0].shape) # must be 200, 6
print('env.is_eef_touching_sponge()',env.is_eef_touching_sponge())
# move_end_effector(env, direction=np.array([0, 0, -1]), speed=0.01, duration=2, data_recorder=data_recorder)
# move_end_effector_to_position(env, obs, obs['sponge_pos'])
# check if contact with sponge  

# sponge_pos = env.sim.data.body_xpos[env.sponge_body_id]
# print('sponge',sponge_pos)
# move_end_effector_to_position(env, sponge_pos)

# Move end-effector to the right (lateral motion) at 0.05 m/s for 1 second
data_recorder, _ = move_end_effector(env, direction=np.array([0, 1, 0]), speed=0.05, duration=1, data_recorder=data_recorder)
print('env.is_eef_touching_sponge()',env.is_eef_touching_sponge())
# Move end-effector to the left (lateral motion) at -0.05 m/s for 1 second
data_recorder, _ = move_end_effector(env, direction=np.array([0, -1, 0]), speed=0.05, duration=1, data_recorder=data_recorder)
print('data_recorder',len(data_recorder), data_recorder[0].shape) # must be 400, 6
print('env.is_eef_touching_sponge()',env.is_eef_touching_sponge())


# Close the environment
env.close()
np.save('record.npy', np.array(data_recorder))
print('Data recorded and saved.')



