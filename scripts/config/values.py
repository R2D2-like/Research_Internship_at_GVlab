EXPLORATORY_MIN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
EXPLORATORY_MAX = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
DEMO_FORCE_TORQUE_MIN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DEMO_FORCE_TORQUE_MAX = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
DEMO_TRAJECTORY_MIN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DEMO_TRAJECTORY_MAX = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
SCALING_FACTOR = 0.9
ID2SPONGE = {0: 's1f1', 1: 's1f2', 2: 's1f3', \
             3: 's2f1', 4: 's2f2', 5: 's2f3', \
             6: 's3f1', 7: 's3f2', 8: 's3f3', \
             9: 's4f1', 10: 's4f2', 11: 's4f3'}
SPONGE2ID = {v: k for k, v in ID2SPONGE.items()}
ALL_SPONGES_LIST = ['s1f1', 's1f2', 's1f3', 's2f1', 's2f2', 's2f3', 's3f1', 's3f2', 's3f3', 's4f1', 's4f2', 's4f3']
DATA_PER_SPONGE = 8
TRAIN_SPONGES_LIST = ['s1f1', 's1f2', 's1f3', 's2f1', 's2f2', 's2f3', 's3f1', 's3f2', 's3f3']