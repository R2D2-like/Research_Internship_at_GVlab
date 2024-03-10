import numpy as np


mode = input('0:step1, 1:rollout: ')
if mode == '0':
    # Load the npy data
    dir = '/root/Research_Internship_at_GVlab/real/step1/data/'
else:
    dir = '/root/Research_Internship_at_GVlab/real/rollout/data/exploratory/'

while True:
    stiffness = input('stiffness level (0, 1, 2, 3): ')
    friction = input('friction level (0, 1, 2): ')
    file_name = 's' + stiffness + 'f' + friction

    # load data
    pressing_data = np.load(dir + 'pressing/' + file_name + '.npz')[file_name] #(200, 6)
    lateral_data = np.load(dir + 'lateral/' + file_name + '.npz')[file_name] #(200, 6)

    # merge the data
    data = np.concatenate([pressing_data, lateral_data], axis=0) #(400, 6)

    # save the data
    save_path = dir + file_name + '.npz'
    np.savez(save_path, **{file_name: data})
    print('data shape:', data.shape)
    print('The data has been saved at', save_path)

