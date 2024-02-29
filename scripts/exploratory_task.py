import numpy as np
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, update_texture
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.arenas.table_arena import TableArena

class ExploratoryTask(SingleArmEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(0.7, 0.005, 0.01),
        sponge_size=(0.06, 0.03),
        sponge_friction=None,
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
        sampling_rate=100,
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
        solref=None,
        solimp=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # settings for sponge
        self.sponge_size = sponge_size
        if sponge_friction is None:
            # lateral_friction_sponge = np.random.uniform(0.2, 8.0)
            # spin_friction_sponge = np.random.uniform(0.0, 4.0)
            # self.sponge_friction = (lateral_friction_sponge, 5e-3, spin_friction_sponge)
            self.sponge_friction = (1, 0.005, 0.0001)
        else:
            self.sponge_friction = sponge_friction

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # set solref for sponge to be used in the mujoco model
        # if solref is not provided, varying contact stiffness k ∈ [80, 1000] N/m
        if solref is None:
            # stiffness = np.random.uniform(80, 1000)
            time_constant = np.random.uniform(0.001, 0.00001) # corresponding to k(=stiffness)
            self.solref = (0.0001, 1) # (time constant, damping ratio)
            print('solref:', self.solref)
        else:
            self.solref = solref

        if solimp is None:
            self.solimp = (0.97, 0.998, 0.001)
        else:
            self.solimp = solimp

        # sampling rate for object observables
        self.sampling_rate = sampling_rate
            
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
        """
        reward = 0.0
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
        mujoco_arena = TableArena(table_full_size=self.table_full_size,\
                                   table_friction=self.table_friction,\
                                    table_offset=self.table_offset)

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # update_texture("Sponge", "/root/Research_Internship_at_GVlab/scripts/textures/sponge.png")

        self.sponge = CylinderObject(
            name="sponge",
            size=self.sponge_size,
            density=50, # kg/m^3 e.g. water is 1000
            friction=self.sponge_friction,
            solref=self.solref,
            solimp=self.solimp,
            material=CustomMaterial(texture="Sponge",\
                                    tex_name="Sponge",\
                                    mat_name="Sponge"),
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.sponge)
        else:
            # スポンジの辺を机の辺と平行になるように配置
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.sponge,
                x_range=[0, 0],
                y_range=[0, 0],
                rotation=np.pi/2,
                rotation_axis="z",
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
                    sampling_rate=self.sampling_rate,
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