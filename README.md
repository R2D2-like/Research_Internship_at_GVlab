# Research_Internship_at_GVlab
This is a project of a research internship at GVlab.

## installation and set up
```
git clone https://github.com/R2D2-like/Research_Internship_at_GVlab.git
cd Research_Internship_at_GVlab
make build_docker
make run_docker

#### check ####
# check xeyes
xeyes
# check glxgears
glxgears
# check robosuite
cd external/robosuite
python3 robosuite/demos/demo_random_action.py
```

<!-- ## data collection ~ converting data ~ training
```
cd external/robosuite/

# データを生成
python3 robosuite/scripts/collect_human_demonstrations.py --environment Wipe --robots UR5e --directory /root/Research_Internship_at_GVlab/demo_data/

#適宜データフォルダ名をわかりやすい名前に変更

# データをrobomimicの形式に変換
cd ../robomimic/
python3 robomimic/scripts/conversion/convert_robosuite.py --dataset /root/Research_Internship_at_GVlab/demo_data/test_data/demo.hdf5

# 観測を抽出
python3 robomimic/scripts/dataset_states_to_obs.py --dataset /root/Research_Internship_at_GVlab/demo_data/test_data/demo.hdf5 --output_name modi_demo.hdf5

# 生成したデータの確認
python3 robomimic/scripts/get_dataset_info.py --dataset /root/Research_Internship_at_GVlab/demo_data/test_data/modi_demo.hdf5

# 学習
cd /root/Research_Internship_at_GVlab/
python3 scripts/bc.py --dataset /root/Research_Internship_at_GVlab/demo_data/test_data/modi_demo.hdf5
``` -->

## pre-training in sim (sim data collection ~ pre-training)
```
cd Research_Internship_at_GVlab

# シミュレーターでデータ収集
python3 scripts/exploratory_action_fixed.py

# pre-process
cd /root/Research_Internship_at_GVlab
python3 scripts/pre-process_sim_data.py

# visualize data
cd /root/Research_Internship_at_GVlab
python3 scripts/vis_data.py

# 学習
python3 scripts/train/pre-training.py
```

<!-- ## ros gazebo for UR5e (noetic)
```
# urシリーズの公式repoのREADMEのur5eバージョン
cd catkin_ws/src/universal_robot

# 必ず３つのウィンドウでこの順に実行する
(terminal 1) roslaunch ur_gazebo ur5e_bringup.launch
(terminal 2) roslaunch ur5e_moveit_config moveit_planning_execution.launch sim:=true
(terminal 3) roslaunch ur5e_moveit_config moveit_rviz.launch
``` -->

## real robot execution (UR5e)
```
#### basic nodes (keep running while executing)
# using gazebo
(terminal 1) roslaunch ur_gripper_gazebo ur_gripper_hande_cubes.launch ur_robot:=ur5e grasp_plugin:=1
# using real hardware
(terminal 1) roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=163.220.51.112

(terminal 2) rosrun ur_control ft_filter.py -t wrench
(terminal 3) rosrun ur_control eef_pose_pub.py

#### step1
# exploratory actions
rosrun ur_control pressing.py
rosrun ur_control lateral.py

# pre-process exploratory actions data
cd /root/Research_Internship_at_GVlab
python3 scripts/pre-process_exploratory_actions_data.py

# visualize data
cd /root/Research_Internship_at_GVlab
python3 scripts/vis_data.py


#### step2
rosrun ur_control step2.py

# visualize data
cd /root/Research_Internship_at_GVlab
python3 scripts/vis_data.py

# pre-process demo data
cd /root/Research_Internship_at_GVlab
python3 scripts/pre-process_demo_data.py

#### train
# baseline
python3 scripts/train/training_baseline.py

# proposal
python3 scripts/train/training_proposed.py

#### rollout
# exploratory actions
rosrun ur_control pressing.py
rosrun ur_control lateral.py

# pre-process exploratory actions data
cd /root/Research_Internship_at_GVlab
python3 scripts/pre-process_exploratory_actions_data.py

# visualize data
cd /root/Research_Internship_at_GVlab
python3 scripts/vis_data.py

# execute baseline
rosrun ur_control rollout_baseline_.py

# execute proposed
rosrun ur_control rollout_proposed_.py

# visualize results
cd /root/Research_Internship_at_GVlab
python3 scripts/vis_data.py
```