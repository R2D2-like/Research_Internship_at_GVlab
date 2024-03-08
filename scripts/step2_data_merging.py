import numpy as np

# Load the npy data
dir = '/root/Research_Internship_at_GVlab/real/step2/data/'
path = []
while True:
    path.append(dir + input('Enter the name of the data: ') + '.npy')
    if input('Do you want to add more data? (y/n): ') == 'n':
        break
save_path = input('Enter the name of the merged data: ')
save_path = dir + save_path + '.npy'

for i in range(len(path)):
    if i == 0:
        data = np.load(path[i]) #(2000,3)
        data = np.expand_dims(data, axis=0) #(1, 2000, 3)
    else:
        data = np.concatenate([data, np.expand_dims(np.load(path[i]), axis=0)], axis=0) #(i+1, 2000, 3)
np.save(save_path, data)
print('data shape:', data.shape)
print('The data has been saved at', save_path)