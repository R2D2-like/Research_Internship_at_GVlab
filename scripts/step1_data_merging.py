import numpy as np

mode = input('0:merge pressing and lateral data, 1:merge step1 data: ')
if mode =='0':
    # Load the npy data
    dir = '/root/Research_Internship_at_GVlab/real/step1/data/'
    path_pressing = input('Enter the name of the pressing data: ')
    path_pressing = dir + 'pressing/' + path_pressing + '.npy'
    path_lateral = input('Enter the name of the lateral data: ')
    path_lateral = dir + 'lateral/' + path_lateral + '.npy'
    save_path = input('Enter the name of the merged data: ')
    save_path = dir + save_path + '.npy'
    data_pressing = np.load(path_pressing) #(200, 6)
    data_lateral = np.load(path_lateral) #(200, 6)
    # merge the data 
    data = np.concatenate([data_pressing, data_lateral], axis=0) #(400, 6)
    np.save(save_path, data)
    print('data shape:', data.shape)
    print('The data has been saved at', save_path)

elif mode == '1':
    # Load the npy data
    dir = '/root/Research_Internship_at_GVlab/real/step1/data/'
    path = []
    while True:
        path.append(dir + input('Enter the name of the data: ') + '.npy')
        if input('Do you want to add more data? (y/n): ') == 'n':
            break
    save_path = input('Enter the name of the merged data: ')
    save_path = dir + save_path + '.npy'

    for i in range(len(path)):
        if i == 0:
            data = np.load(path[i]) #(400,6)
            data = np.expand_dims(data, axis=0) #(1, 400, 6)
        else:
            data = np.concatenate([data, np.expand_dims(np.load(path[i]), axis=0)], axis=0) #(i+1, 400, 6)
    np.save(save_path, data)
    print('data shape:', data.shape)
    print('The data has been saved at', save_path)
