# Research_Internship_at_GVlab
This is a project of a research internship at GVlab.

## set up
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

## sim (sim data collection ~ pre-training)
```
cd Research_Internship_at_GVlab

# シミュレーターでデータ収集
python3 scripts/exploratory_action_fixed.py

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

## real robot execution (sim ver)
```
(terminal 1) roslaunch ur_gripper_gazebo ur_gripper_hande_cubes.launch ur_robot:=ur5e grasp_plugin:=1
(terminal 2) rosrun ur_control ft_filter.py -t wrench

# step1
rosrun ur_control step1.py

# step2
rosrun ur_control step2.py

# rollout
rosrun ur_control rollout.py
```
