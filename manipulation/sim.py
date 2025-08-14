import ipdb
import numpy as np
import genesis as gs
import gym
from gym.utils import seeding
from gym import spaces
import pickle
import yaml
import os.path as osp
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from manipulation.panda import Panda
from manipulation.ur5 import UR5
from manipulation.sawyer import Sawyer
from manipulation.utils import parse_config, load_env, download_and_parse_objavarse_obj_from_yaml_config
from manipulation.gpt_reward_api import get_joint_id_from_name, get_link_id_from_name


class SimpleEnvGenesis(gym.Env):
    def __init__(self,
                 dt=0.01,
                 config_path=None,
                 gui=False,
                 frameskip=2,
                 horizon=120,
                 restore_state_file=None,
                 rotation_mode='delta-axis-angle-local',
                 translation_mode='delta-translation',
                 max_rotation=np.deg2rad(5),
                 max_translation=0.15,
                 use_suction=True,  # whether to use a suction gripper
                 object_candidate_num=6,  # how many candidate objects to sample from objaverse
                 vhacd=False,  # if to perform vhacd on the object for better collision detection for genesis
                 randomize=0,  # if to randomize the scene
                 obj_id=0,  # which object to choose to use from the candidates
                 ):

        super().__init__()

        # Task
        self.config_path = config_path
        self.restore_state_file = restore_state_file
        self.frameskip = frameskip
        self.horizon = horizon
        self.gui = gui
        self.object_candidate_num = object_candidate_num
        self.solution_path = None
        self.success = False  # not really used, keeped for now
        self.primitive_save_path = None  # to be used for saving the primitives execution results
        self.randomize = randomize
        self.obj_id = obj_id  # which object to choose to use from the candidates

        # physics
        self.gravity = -9.81
        self.contact_constraint = None
        self.vhacd = vhacd

        # action space
        self.use_suction = use_suction
        self.rotation_mode = rotation_mode
        self.translation_mode = translation_mode
        self.max_rotation_angle = max_rotation
        self.max_translation = max_translation
        self.suction_to_obj_pose = 0
        self.suction_contact_link = None
        self.suction_obj_id = None
        self.activated = 0

        # Initialize Genesis
        if self.gui:
            gs.init(backend=gs.gpu)  # Use vulkan for GUI
        else:
            gs.init(backend=gs.cpu)  # Use CUDA for headless mode

        self.asset_dir = osp.join(osp.dirname(osp.realpath(__file__)), "assets/")
        hz = int(1 / dt)
        self.dt = dt

        self.seed()
        self.set_scene()
        self.setup_camera_rpy()
        self.scene_lower, self.scene_upper = self.get_scene_bounds()
        self.scene_center = (self.scene_lower + self.scene_upper) / 2
        self.scene_range = (self.scene_upper - self.scene_lower) / 2

        self.grasp_action_mag = 0.06 if not self.use_suction else 1
        self.action_low = np.array([-1, -1, -1, -1, -1, -1, -1])
        self.action_high = np.array([1, 1, 1, 1, 1, 1, self.grasp_action_mag])

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.base_action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.num_objects = len(self.urdf_ids) - 2  # exclude plane, robot
        distractor_object_num = np.sum(list(self.is_distractor.values()))
        self.num_objects -= distractor_object_num

        ### For RL policy learning, observation space includes:
        # 1. object positions and orientations (6 * num_objects)
        # 2. object min and max bounding box (6 * num_objects)
        # 3. articulated object joint angles (num_objects * num_joints)
        # 4. articulated object link position and orientation (num_objects * num_joints * 6)
        # 5. robot base position (xy)
        # 6. robot end-effector position and orientation (6)
        # 7. gripper suction activated/deactivate or gripper joint angle (if not using suction gripper) (1)
        num_obs = self.num_objects * 12  # obs 1 and 2
        for name in self.urdf_types:
            if self.urdf_types[name] == 'urdf' and not self.is_distractor[name]:  # obs 3 and 4
                num_joints = len(self.genesis_objects[name].joints) if hasattr(self.genesis_objects[name],
                                                                               'joints') else 0
                num_obs += num_joints
                num_obs += 6 * num_joints
        num_obs += 2 + 6 + 1  # obs 5 6 7
        self.base_num_obs = num_obs

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32)
        self.base_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.base_num_obs,), dtype=np.float32)

        self.detected_position = {}  # not used for now, keep it

    def normalize_position(self, pos):
        if self.translation_mode == 'normalized-direct-translation':
            return (pos - self.scene_center) / self.scene_range
        else:
            return pos

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random()

    def get_aabb(self, obj):
        # Get AABB for Genesis object
        if hasattr(obj, 'get_aabb'):
            return obj.get_aabb()
        else:
            # Fallback for rigid bodies
            pos = obj.get_position()
            size = obj.get_size()
            min_aabb = pos - size / 2
            max_aabb = pos + size / 2
            return min_aabb, max_aabb

    def get_aabb_link(self, obj, link_id):
        # For articulated objects, get AABB of specific link
        if hasattr(obj, 'get_link_aabb'):
            return obj.get_link_aabb(link_id)
        else:
            return self.get_aabb(obj)

    def get_scene_bounds(self):
        min_aabbs = []
        max_aabbs = []
        for name, obj in self.genesis_objects.items():
            if name == 'plane': continue
            min_aabb, max_aabb = self.get_aabb(obj)
            min_aabbs.append(min_aabb)
            max_aabbs.append(max_aabb)

        min_aabb = np.min(np.stack(min_aabbs, axis=0).reshape(-1, 3), axis=0)
        max_aabb = np.max(np.stack(max_aabbs, axis=0).reshape(-1, 3), axis=0)
        range = max_aabb - min_aabb
        return min_aabb - 0.5 * range, max_aabb + 0.5 * range

    def clip_within_workspace(self, robot_pos, ori_pos, on_table):
        pos = ori_pos.copy()
        if not on_table:
            # If objects are too close to the robot, push them away
            x_near_low, x_near_high = robot_pos[0] - 0.3, robot_pos[0] + 0.3
            y_near_low, y_near_high = robot_pos[1] - 0.3, robot_pos[1] + 0.3

            if pos[0] > x_near_low and pos[0] < x_near_high:
                pos[0] = x_near_low if pos[0] < robot_pos[0] else x_near_high

            if pos[1] > y_near_low and pos[1] < y_near_high:
                pos[1] = y_near_low if pos[1] < robot_pos[1] else y_near_high
            return pos
        else:
            # Object is on table, should be within table's bounding box
            new_pos = pos.copy()
            new_pos[:2] = np.clip(new_pos[:2], self.table_bbox_min[:2], self.table_bbox_max[:2])
            return new_pos

    def get_robot_base_pos(self):
        robot_base_pos = [1, 1, 0.28]
        return robot_base_pos

    def get_robot_init_joint_angles(self):
        init_joint_angles = [0 for _ in range(len(self.robot.right_arm_joint_indices))]
        if self.robot_name == 'panda':
            init_joint_angles = [0, -1.10916842e-04, 7.33823451e-05, -5.47701370e-01, -5.94950533e-01,
                                 2.62857916e+00, -4.85316284e-01, 1.96042022e+00, 2.15271531e+00,
                                 -7.35304443e-01]
        return init_joint_angles

    def set_scene(self):
        ### simulation preparation
        self.world = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                gravity=(0, 0, self.gravity),
            ),
            show_viewer=True,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
        )

        ### load restore state
        restore_state = None
        if self.restore_state_file is not None:
            with open(self.restore_state_file, 'rb') as f:
                restore_state = pickle.load(f)

        ### load plane
        plane = self.world.add_entity(gs.morphs.Plane())

        ### create and load a robot
        robot_base_pos = self.load_robot(restore_state)

        ### load and parse task config (including semantically meaningful distractor objects)
        self.urdf_ids = {
            "robot": self.robot,
            "plane": plane,
        }
        self.urdf_paths = {}
        self.urdf_types = {}
        self.init_positions = {}
        self.on_tables = {}
        self.simulator_sizes = {}
        self.is_distractor = {
            "robot": 0,
            "plane": 0,
        }
        urdf_paths, urdf_sizes, urdf_positions, urdf_names, urdf_types, urdf_on_table, urdf_movables, \
            use_table, articulated_init_joint_angles, spatial_relationships = self.load_and_parse_config(restore_state)
        ipdb.set_trace()
        ### handle the case if there is a table
        self.load_table(use_table, restore_state)

        ### load each object from the task config
        self.load_object(urdf_paths, urdf_sizes, urdf_positions, urdf_names, urdf_types, urdf_on_table, urdf_movables)

        ### adjusting object positions
        ### place the lowest point on the object to be the height where GPT specifies
        object_height = self.adjust_object_positions(robot_base_pos)

        ### resolve collisions between objects
        self.resolve_collision(robot_base_pos, object_height, spatial_relationships)

        ### handle any special relationships outputted by GPT
        self.handle_gpt_special_relationships(spatial_relationships)

        ### set all object's joint angles to the lower joint limit
        self.set_to_default_joint_angles()

        ### overwrite joint angles specified by GPT
        self.handle_gpt_joint_angle(articulated_init_joint_angles)

        ### record initial joint angles and positions
        self.record_initial_joint_and_pose()

        ### stabilize the scene
        for _ in range(500):
            self.world.step(self.dt)

        ### restore to a state if provided
        if self.restore_state_file is not None:
            load_env(self, self.restore_state_file)

        ### Enable debug rendering
        if self.gui:
            self.world.enable_rendering()

        self.init_state = self.world.save_state()

    def load_robot(self, restore_state):
        robot_classes = {
            "panda": lambda: gs.morphs.MJCF(file = 'xml/franka_emika_panda/panda.xml'),
            # "ur5": lambda: gs.morphs.MJCF(file = 'xml/universal_robots_ur5e/ur5e.xml'),
        }
        robot_names = list(robot_classes.keys())
        self.robot_name = robot_names[np.random.randint(len(robot_names))]
        if restore_state is not None and "robot_name" in restore_state:
            self.robot_name = restore_state['robot_name']
        self.robot_class = robot_classes[self.robot_name]

        # Create robot
        self.robot = self.world.add_entity(self.robot_class())
        self.agents = [self.robot]
        # self.suction_id = self.robot.right_gripper_indices[0]
        jnt_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'joint7',
            'finger_joint1',
            'finger_joint2',
        ]
        self.dofs_idx = [self.robot.get_joint(name).dof_idx_local for name in jnt_names]

    def reset_robot(self):
        # Update robot motor gains
        self.robot.set_dofs_kp(
            kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
            dofs_idx_local=self.dofs_idx,
        )
        # set velocity gains
        self.robot.set_dofs_kv(
            kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
            dofs_idx_local=self.dofs_idx,
        )
        # set force range for safety
        self.robot.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
            dofs_idx_local=self.dofs_idx,
        )

        # Set robot base position & orientation, and joint angles
        robot_base_pos = self.get_robot_base_pos()
        robot_base_orient = [0, 0, 0, 1]
        self.robot_base_orient = robot_base_orient
        end_effector = self.robot.get_link("hand")
        q_pregrasp = self.robot.inverse_kinematics(
            link=end_effector,
            pos=np.array(robot_base_pos),
            quat=np.array(self.robot_base_orient),
        )
        self.robot.control_dofs_position(q_pregrasp[:-2], np.arange(7))

        return robot_base_pos

    def load_and_parse_config(self, restore_state):
        ### select and download objects from objaverse
        res = download_and_parse_objavarse_obj_from_yaml_config(self.config_path,
                                                                candidate_num=self.object_candidate_num,
                                                                vhacd=self.vhacd)
        if not res:
            print("=" * 20)
            print("some objects cannot be found in objaverse, task_build failed, now exit ...")
            print("=" * 20)
            exit()

        self.config = None
        while self.config is None:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        for obj in self.config:
            if "solution_path" in obj:
                self.solution_path = obj["solution_path"]
                break

        ### parse config
        urdf_paths, urdf_sizes, urdf_positions, urdf_names, urdf_types, urdf_on_table, use_table, \
            articulated_init_joint_angles, spatial_relationships, distractor_config_path, urdf_movables = parse_config(
            self.config,
            use_bard=True, obj_id=self.obj_id)
        if not use_table:
            urdf_on_table = [False for _ in urdf_on_table]
        urdf_names = [x.lower() for x in urdf_names]
        for name in urdf_names:
            self.is_distractor[name] = 0

        ### parse distractor object config (semantically meaningful objects that are related but not used for the task)
        if distractor_config_path is not None:
            self.distractor_config_path = distractor_config_path
            res = download_and_parse_objavarse_obj_from_yaml_config(distractor_config_path,
                                                                    candidate_num=self.object_candidate_num,
                                                                    vhacd=self.vhacd)
            with open(distractor_config_path, 'r') as f:
                self.distractor_config = yaml.safe_load(f)
            distractor_urdf_paths, distractor_urdf_sizes, distractor_urdf_positions, distractor_urdf_names, distractor_urdf_types, \
                distractor_urdf_on_table, _, _, _, _, _ = \
                parse_config(self.distractor_config, use_bard=True, obj_id=self.obj_id, use_vhacd=False)
            distractor_urdf_names = [x.lower() for x in distractor_urdf_names]
            if not use_table:
                distractor_urdf_on_table = [False for _ in distractor_urdf_on_table]

            for name in distractor_urdf_names:
                self.is_distractor[name] = 1

            distractor_movables = [True for _ in distractor_urdf_names]

            urdf_paths += distractor_urdf_paths
            urdf_sizes += distractor_urdf_sizes
            urdf_positions += distractor_urdf_positions
            urdf_names += distractor_urdf_names
            urdf_types += distractor_urdf_types
            urdf_on_table += distractor_urdf_on_table
            urdf_movables += distractor_movables

        if restore_state is not None:
            if "urdf_paths" in restore_state:
                self.urdf_paths = restore_state['urdf_paths']
                urdf_paths = [self.urdf_paths[name] for name in urdf_names]
            if "object_sizes" in restore_state:
                self.simulator_sizes = restore_state['object_sizes']
                urdf_sizes = [self.simulator_sizes[name] for name in urdf_names]

        return urdf_paths, urdf_sizes, urdf_positions, urdf_names, urdf_types, urdf_on_table, urdf_movables, \
            use_table, articulated_init_joint_angles, spatial_relationships

    def load_table(self, use_table, restore_state):
        self.use_table = use_table
        if use_table:
            from manipulation.table_utils import table_paths, table_scales, table_poses, table_bbox_scale_down_factors
            self.table_path = table_paths[np.random.randint(len(table_paths))]
            if restore_state is not None:
                self.table_path = restore_state['table_path']

            table_scale = table_scales[self.table_path]
            table_pos = table_poses[self.table_path]
            table_orientation = [np.pi / 2, 0, 0]

            # Create table as rigid body in Genesis
            table = gs.RigidBody.box(size=[1.0, 1.0, 0.1], position=table_pos, orientation=table_orientation)
            self.world.add(table)

            if not self.randomize:
                random_orientation = table_orientation
            else:
                random_orientations = [0, np.pi / 2, np.pi, np.pi * 3 / 2]
                random_orientation = [np.pi / 2, 0, random_orientations[np.random.randint(4)]]

            table.set_orientation(random_orientation)
            self.table_bbox_min, self.table_bbox_max = self.get_aabb(table)

            table_range = self.table_bbox_max - self.table_bbox_min
            self.table_bbox_min[:2] += table_range[:2] * table_bbox_scale_down_factors[self.table_path]
            self.table_bbox_max[:2] -= table_range[:2] * table_bbox_scale_down_factors[self.table_path]
            self.table_height = self.table_bbox_max[2]
            self.simulator_sizes["init_table"] = table_scale
            self.urdf_ids["init_table"] = table
            self.genesis_objects["init_table"] = table
            self.is_distractor['init_table'] = 0

    def load_object(self, urdf_paths, urdf_sizes, urdf_positions, urdf_names, urdf_types, urdf_on_table, urdf_movables):
        for path, size, pos, name, type, on_table, moveable in zip(urdf_paths, urdf_sizes, urdf_positions, urdf_names,
                                                                   urdf_types, urdf_on_table, urdf_movables):
            name = name.lower()
            # by default, all objects movable, except the urdf files
            use_fixed_base = (type == 'urdf' and not self.is_distractor[name])
            if type == 'urdf' and moveable:  # if gpt specified the object is movable, then it is movable
                use_fixed_base = False
            size = min(size, 1.2)
            size = max(size, 0.1)  # if the object is too small, current gripper cannot really manipulate it.

            x_orient = np.pi / 2 if type == 'mesh' else 0  # handle different coordinate axis by objaverse and partnet-mobility
            if self.randomize or self.is_distractor[name]:
                orientation = [x_orient, 0, self.np_random.uniform(-np.pi / 3, np.pi / 3)]
            else:
                orientation = [x_orient, 0, 0]

            if not on_table:
                load_pos = pos
            else:  # change to be table coordinate
                table_xy_range = self.table_bbox_max[:2] - self.table_bbox_min[:2]
                obj_x = self.table_bbox_min[0] + pos[0] * table_xy_range[0]
                obj_y = self.table_bbox_min[1] + pos[1] * table_xy_range[1]
                obj_z = self.table_height
                load_pos = [obj_x, obj_y, obj_z]

            # Load object in Genesis
            if type == 'urdf':
                # For articulated objects, load as articulated body
                obj = gs.ArticulatedBody.load(path, position=load_pos, orientation=orientation,
                                              fixed_base=use_fixed_base)
            else:
                # For mesh objects, load as rigid body
                obj = gs.RigidBody.load(path, position=load_pos, orientation=orientation, fixed=use_fixed_base)

            self.world.add(obj)

            # scale size
            if name in self.simulator_sizes:
                # Remove and reload with saved size
                self.world.remove(obj)
                saved_size = self.simulator_sizes[name]
                if type == 'urdf':
                    obj = gs.ArticulatedBody.load(path, position=load_pos, orientation=orientation,
                                                  fixed_base=use_fixed_base, scale=saved_size)
                else:
                    obj = gs.RigidBody.load(path, position=load_pos, orientation=orientation, fixed=use_fixed_base,
                                            scale=saved_size)
                self.world.add(obj)
            else:
                min_aabb, max_aabb = self.get_aabb(obj)
                actual_size = np.linalg.norm(max_aabb - min_aabb)
                if np.abs(actual_size - size) > 0.05:
                    self.world.remove(obj)
                    scale_factor = size ** 2 / actual_size
                    if type == 'urdf':
                        obj = gs.ArticulatedBody.load(path, position=load_pos, orientation=orientation,
                                                      fixed_base=use_fixed_base, scale=scale_factor)
                    else:
                        obj = gs.RigidBody.load(path, position=load_pos, orientation=orientation, fixed=use_fixed_base,
                                                scale=scale_factor)
                    self.world.add(obj)
                    self.simulator_sizes[name] = scale_factor
                else:
                    self.simulator_sizes[name] = size

            self.urdf_ids[name] = obj
            self.genesis_objects[name] = obj
            self.urdf_paths[name] = path
            self.urdf_types[name] = type
            self.init_positions[name] = np.array(load_pos)
            self.on_tables[name] = on_table

            print("Finished loading object: ", name)

    def adjust_object_positions(self, robot_base_pos):
        object_height = {}
        for name, obj in self.genesis_objects.items():
            if name == 'robot' or name == 'plane' or name == 'init_table': continue
            min_aabb, max_aabb = self.get_aabb(obj)
            min_z = min_aabb[2]
            object_height[obj] = 2 * self.init_positions[name][2] - min_z
            pos = obj.get_position()
            orient = obj.get_orientation()
            new_pos = np.array(pos)
            new_pos = self.clip_within_workspace(robot_base_pos, new_pos, self.on_tables[name])
            new_pos[2] = object_height[obj]
            obj.set_position(new_pos)
            self.init_positions[name] = new_pos

        return object_height

    def resolve_collision(self, robot_base_pos, object_height, spatial_relationships):
        collision = True
        collision_cnt = 1
        while collision:
            if collision_cnt % 50 == 0:  # if collision is not resolved every 50 iterations, we randomly reset object's position
                for name, obj in self.genesis_objects.items():
                    if name == 'robot' or name == 'plane' or name == "init_table": continue
                    pos = self.init_positions[name]
                    orient = obj.get_orientation()
                    new_pos = np.array(pos) + np.random.uniform(-0.2, 0.2, size=3)
                    new_pos = self.clip_within_workspace(robot_base_pos, new_pos, self.on_tables[name])
                    new_pos[2] = object_height[obj]
                    obj.set_position(new_pos)
                    self.world.step(self.dt)

            push_directions = defaultdict(list)  # store the push direction for each object

            # detect collisions between objects
            detected_collision = False
            for name, obj in self.genesis_objects.items():
                if name == 'robot' or name == 'plane' or name == 'init_table': continue
                for name2, obj2 in self.genesis_objects.items():
                    if name == name2 or name2 == 'robot' or name2 == 'plane' or name2 == 'init_table': continue

                    # if gpt specifies obj a and obj b should have some special relationship, then skip collision resolution
                    skip = False
                    for spatial_relationship in spatial_relationships:
                        words = spatial_relationship.lower().split(",")
                        words = [word.strip().lstrip() for word in words]
                        if name in words and name2 in words:
                            skip = True
                            break

                    if skip: continue

                    # Check collision using Genesis collision detection
                    if self.world.check_collision(obj, obj2):
                        # Get collision normal for push direction
                        collision_info = self.world.get_collision_info(obj, obj2)
                        push_direction = collision_info['normal']

                        # both are distractors or both are not, push both objects away
                        if (self.is_distractor[name] and self.is_distractor[name2]) or \
                                (not self.is_distractor[name] and not self.is_distractor[name2]):
                            push_directions[obj].append(-push_direction)
                            push_directions[obj2].append(push_direction)
                        # only 1 is distractor, only pushes the distractor
                        if self.is_distractor[name] and not self.is_distractor[name2]:
                            push_directions[obj].append(push_direction)
                        if not self.is_distractor[name] and self.is_distractor[name2]:
                            push_directions[obj2].append(-push_direction)

                        detected_collision = True

            # collisions between robot and objects, only push object away
            for name, obj in self.genesis_objects.items():
                if name == 'robot' or name == 'plane' or name == 'init_table':
                    continue

                if self.world.check_collision(self.robot.body, obj):
                    collision_info = self.world.get_collision_info(self.robot.body, obj)
                    push_direction = collision_info['normal']
                    push_directions[obj].append(-push_direction)
                    detected_collision = True

            # between table and objects that should not be placed on table
            if self.use_table:
                for name, obj in self.genesis_objects.items():
                    if name == 'robot' or name == 'plane' or name == 'init_table':
                        continue
                    if self.on_tables[name]:
                        continue

                    if self.world.check_collision(self.robot.body, obj):
                        collision_info = self.world.get_collision_info(self.robot.body, obj)
                        push_direction = collision_info['normal']
                        push_directions[obj].append(-push_direction)
                        detected_collision = True

            # move objects
            push_distance = 0.1
            for obj in push_directions:
                for direction in push_directions[obj]:
                    pos = obj.get_position()
                    orient = obj.get_orientation()
                    new_pos = np.array(pos) + push_distance * direction
                    new_pos = self.clip_within_workspace(robot_base_pos, new_pos, self.on_tables[name])
                    new_pos[2] = object_height[obj]

                    obj.set_position(new_pos)
                    self.world.step(self.dt)

            collision = detected_collision
            collision_cnt += 1

            if collision_cnt > 1000:
                break

    def record_initial_joint_and_pose(self):
        self.initial_joint_angle = {}
        for name in self.genesis_objects:
            obj = self.genesis_objects[name.lower()]
            if name == 'robot' or name == 'plane' or name == "init_table": continue
            if self.urdf_types[name.lower()] == 'urdf' and obj.n_dofs > 0:
                self.initial_joint_angle[name] = {}
                for i in range(obj.n_dofs):
                    joint_name = f"joint_{i}"  # Genesis doesn't have joint names by default
                    joint_angle = obj.get_dofs_position()[i]
                    self.initial_joint_angle[name][joint_name] = joint_angle

        self.initial_pos = {}
        self.initial_orient = {}
        for name in self.genesis_objects:
            obj = self.genesis_objects[name.lower()]
            if name == 'robot' or name == 'plane' or name == "init_table": continue
            pos = obj.get_pos()
            quat = obj.get_quat()
            self.initial_pos[name] = pos
            self.initial_orient[name] = quat

    def set_to_default_joint_angles(self):
        for obj_name in self.genesis_objects:
            if obj_name == 'robot' or obj_name == 'plane' or obj_name == "init_table": continue
            obj = self.genesis_objects[obj_name]
            if obj.n_dofs > 0:
                # Set all joints to a small offset from their lower limits
                dof_lower = obj.dof_lower_limits
                dof_upper = obj.dof_upper_limits
                joint_vals = dof_lower + 0.06 * (dof_upper - dof_lower)
                obj.set_dofs_position(joint_vals)

    def handle_gpt_special_relationships(self, spatial_relationships):
        # Handle "on" and "in" relationships - simplified for Genesis
        for spatial_relationship in spatial_relationships:
            words = spatial_relationship.lower().split(",")
            words = [word.strip().lstrip() for word in words]
            if words[0] == "on":
                obj_a = words[1]
                obj_b = words[2]

                if obj_a in self.genesis_objects and obj_b in self.genesis_objects:
                    obj_a_entity = self.genesis_objects[obj_a]
                    obj_b_entity = self.genesis_objects[obj_b]

                    # Get bounding boxes
                    obj_a_aabb = obj_a_entity.get_aabb()
                    obj_b_aabb = obj_b_entity.get_aabb()

                    # Place obj_a on top of obj_b
                    new_pos = (obj_b_aabb[0] + obj_b_aabb[1]) / 2
                    new_pos[2] = obj_b_aabb[1][2] + (obj_a_aabb[1][2] - obj_a_aabb[0][2]) / 2

                    obj_a_entity.set_pos(new_pos)

    def handle_gpt_joint_angle(self, articulated_init_joint_angles):
        for name in articulated_init_joint_angles:
            if name.lower() in self.genesis_objects:
                obj = self.genesis_objects[name.lower()]
                if obj.n_dofs > 0:
                    dof_lower = obj.dof_lower_limits
                    dof_upper = obj.dof_upper_limits

                    for joint_idx, (joint_name, joint_angle) in enumerate(articulated_init_joint_angles[name].items()):
                        if joint_idx < obj.n_dofs:
                            if 'random' not in str(joint_angle):
                                joint_angle = float(joint_angle)
                                joint_angle = max(0.06, min(0.7, joint_angle))
                                joint_angle = dof_lower[joint_idx] + joint_angle * (
                                            dof_upper[joint_idx] - dof_lower[joint_idx])
                            else:
                                joint_angle = self.np_random.uniform(dof_lower[joint_idx], dof_upper[joint_idx])

                            current_dofs = obj.get_dofs_position()
                            current_dofs[joint_idx] = joint_angle
                            obj.set_dofs_position(current_dofs)

    def reset(self):
        # Reset all objects to initial state
        for name, obj in self.genesis_objects.items():
            if name in self.initial_pos:
                obj.set_pos(self.initial_pos[name])
                obj.set_quat(self.initial_orient[name])

        # Reset robot
        if 'robot' in self.genesis_objects:
            robot_obj = self.genesis_objects['robot']
            init_joint_angles = self.get_robot_init_joint_angles()
            if robot_obj.n_dofs > 0:
                robot_obj.set_dofs_position(np.array(init_joint_angles[:robot_obj.n_dofs]))

        self.time_step = 0
        self.success = False
        if self.use_suction:
            self.deactivate_suction()

        return self._get_obs()

    def setup_camera(self, camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=640,
                     camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        # Genesis camera setup - will be handled by renderer
        self.camera_eye = camera_eye
        self.camera_target = camera_target
        self.fov = fov

    def setup_camera_rpy(self, camera_target=[0, 0, 0.3], distance=1.6, rpy=[0, -30, -30], fov=60, camera_width=640,
                         camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        if self.use_table:
            camera_target = np.array([0, 0, 0.3])
        else:
            for name in self.genesis_objects:
                if name in ['robot', 'plane', 'init_table']: continue
                obj = self.genesis_objects[name]
                aabb = obj.get_aabb()
                center = (aabb[0] + aabb[1]) / 2
                camera_target = center
                break

        # Convert rpy to camera position
        import math
        yaw, pitch, roll = np.deg2rad(rpy)
        x = distance * math.cos(pitch) * math.cos(yaw)
        y = distance * math.cos(pitch) * math.sin(yaw)
        z = distance * math.sin(pitch)
        camera_eye = camera_target + np.array([x, y, z])

        self.camera_eye = camera_eye
        self.camera_target = camera_target
        self.fov = fov

    def render(self, mode=None):
        # Genesis rendering
        if hasattr(self, 'camera_eye'):
            # Set up camera for rendering
            self.scene.add_camera(
                pos=self.camera_eye,
                lookat=self.camera_target,
                fov=self.fov,
                res=(self.camera_width, self.camera_height)
            )
            img = self.scene.render()
            return img, None  # Genesis doesn't return depth by default
        return None, None

    def take_step(self, actions, gains=None, forces=None):
        if gains is None:
            gains = [a.motor_gains for a in self.agents]
        elif type(gains) not in (list, tuple):
            gains = [gains] * len(self.agents)
        if forces is None:
            forces = [a.motor_forces for a in self.agents]
        elif type(forces) not in (list, tuple):
            forces = [forces] * len(self.agents)

        action_index = 0
        for i, agent in enumerate(self.agents):
            agent_action_len = self.base_action_space.shape[0]
            action = np.copy(actions[action_index:action_index + agent_action_len])
            action_index += agent_action_len
            action = np.clip(action, self.action_low, self.action_high)

            translation = action[:3]
            rotation = action[3:6]
            suction = action[6]

            # Get current end effector pose from Genesis robot
            if hasattr(agent, 'robot_entity'):
                current_pos = agent.robot_entity.get_pos()
                current_quat = agent.robot_entity.get_quat()

                # Apply translation
                if self.translation_mode == 'delta-translation':
                    target_pos = current_pos + translation * self.max_translation
                elif self.translation_mode == 'normalized-direct-translation':
                    target_pos = translation * self.scene_range + self.scene_center
                elif self.translation_mode == 'direct-translation':
                    target_pos = translation

                    # Apply rotation
                if self.rotation_mode == 'euler-angle':
                    rotation = rotation * np.pi
                    target_quat = gs.math.quat_from_euler(rotation)
                elif 'delta-axis-angle' in self.rotation_mode or 'delta-euler-angle' in self.rotation_mode:
                    target_quat = self.apply_delta_rotation(rotation, current_quat)
                else:
                    target_quat = current_quat

                # Set target pose (simplified - in real implementation you'd use IK)
                agent.robot_entity.set_pos(target_pos)
                agent.robot_entity.set_quat(target_quat)

            # Handle gripper/suction
            if not self.use_suction:
                # Simplified gripper control
                pass
            else:
                if suction >= 0:
                    self.activate_suction()
                else:
                    self.deactivate_suction()

        # Step simulation
        for _ in range(self.frameskip):
            self.world.step()

    def apply_delta_rotation(self, delta_rotation, current_quat):
        # Convert quaternion to rotation matrix and apply delta rotation
        from scipy.spatial.transform import Rotation as R

        if 'delta-axis-angle' in self.rotation_mode:
            dtheta = np.linalg.norm(delta_rotation)
            if dtheta > 0:
                delta_rotation = delta_rotation / dtheta
                dtheta = dtheta * self.max_rotation_angle / np.sqrt(3)
                delta_rotation_matrix = R.from_rotvec(delta_rotation * dtheta).as_matrix()
            else:
                delta_rotation_matrix = np.eye(3)

            current_matrix = R.from_quat(current_quat).as_matrix()

            if self.rotation_mode == 'delta-axis-angle-local':
                new_rotation = current_matrix @ delta_rotation_matrix
            elif self.rotation_mode == 'delta-axis-angle-global':
                new_rotation = delta_rotation_matrix @ current_matrix
            return R.from_matrix(new_rotation).as_quat()
        elif self.rotation_mode == 'delta-euler-angle':
            euler_angle = delta_rotation / np.sqrt(3) * self.max_rotation_angle
            delta_quat = gs.math.quat_from_euler(euler_angle)
            return gs.math.quat_mul(delta_quat, current_quat)

        return current_quat

    def activate_suction(self):
        if not self.activated:
            # Simplified suction activation for Genesis
            # In a full implementation, you'd check for contacts and create constraints
            self.activated = True
            self.suction_obj_id = None  # Would be set based on contact detection

    def deactivate_suction(self):
        self.activated = False
        # Remove any suction constraints

    def step(self, action):
        self.time_step += 1
        self.take_step(action)
        obs = self._get_obs()
        try:
            reward, success = self._compute_reward()
        except:
            reward, success = self.compute_reward()
        self.success = success
        done = self.time_step == self.horizon
        info = self._get_info()
        return obs, reward, done, info

    def _get_info(self):
        return {}

    def _get_obs(self):
        obs = np.zeros(self.base_observation_space.shape[0])

        cnt = 0
        # Object positions and orientations
        for name in self.genesis_objects:
            if name == 'plane' or name == 'robot':
                continue
            if self.is_distractor[name]:
                continue

            obj = self.genesis_objects[name]
            pos = obj.get_pos()
            quat = obj.get_quat()
            euler_angle = gs.math.quat_to_euler(quat)

            obs[cnt:cnt + 3] = self.normalize_position(pos)
            obs[cnt + 3:cnt + 6] = euler_angle
            cnt += 6

        # Object bounding boxes
        for name in self.genesis_objects:
            if name == 'plane' or name == 'robot':
                continue
            if self.is_distractor[name]:
                continue

            obj = self.genesis_objects[name]
            aabb = obj.get_aabb()
            obs[cnt:cnt + 3] = self.normalize_position(aabb[0])
            obs[cnt + 3:cnt + 6] = self.normalize_position(aabb[1])
            cnt += 6

        # Articulated object joint angles and link poses
        for name in self.urdf_types:
            if self.urdf_types[name] == 'urdf' and not self.is_distractor[name]:
                if name in self.genesis_objects:
                    obj = self.genesis_objects[name]
                    if obj.n_dofs > 0:
                        dof_positions = obj.get_dofs_position()
                        for joint_idx in range(obj.n_dofs):
                            obs[cnt] = dof_positions[joint_idx]
                            cnt += 1
                            # For link poses, we'd need to implement link state tracking in Genesis
                            # For now, use dummy values
                            obs[cnt:cnt + 6] = np.zeros(6)
                            cnt += 6

        # Robot base position
        if 'robot' in self.genesis_objects:
            robot_obj = self.genesis_objects['robot']
            robot_base_pos = robot_obj.get_pos()
            robot_base_pos = self.normalize_position(robot_base_pos)
            obs[cnt:cnt + 2] = robot_base_pos[:2]
            cnt += 2

            # Robot end-effector position and orientation
            robot_eef_pos = robot_obj.get_pos()  # Simplified - should get actual EE pose
            robot_eef_quat = robot_obj.get_quat()
            robot_eef_euler = gs.math.quat_to_euler(robot_eef_quat)
            obs[cnt:cnt + 3] = self.normalize_position(robot_eef_pos)
            obs[cnt + 3:cnt + 6] = robot_eef_euler
            cnt += 6

        # Gripper state
        if not self.use_suction:
            # Simplified gripper joint angles
            obs[cnt:cnt + 2] = np.zeros(2)  # Would get actual gripper joint states
            cnt += 2
        else:
            obs[cnt] = int(self.activated)
            cnt += 1

        return obs

    def disconnect(self):
        pass  # Genesis cleanup would go here

    def close(self):
        pass  # Genesis cleanup would go here


SimpleEnv = SimpleEnvGenesis