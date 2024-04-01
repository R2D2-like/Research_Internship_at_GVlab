import numpy as np
import matplotlib.pyplot as plt
z_diff_data_path = '/root/Research_Internship_at_GVlab/data0402/real/step2/data/demo_preprocessed_z_diff.npz'
ft_data_path = '/root/Research_Internship_at_GVlab/data0402/real/step2/data/demo_preprocessed_ft.npz'
sponge = input('sponge: ')
trial = int(input('trial: ')) -1 
z_diff = np.load(z_diff_data_path)[sponge][trial]#(99,)
print(z_diff.shape)
ft = np.load(ft_data_path)[sponge][trial] #(6, 100)
print(ft.shape)
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.set_title('Z displacement')
# if value of z_diff is less than 0.49074766445156415, plot as blue
# else plot as red
#TODO

# Separate data into two arrays based on the condition
thresh = 0.4500329131968895#0.444298251616438 #0.5128738240251608
z_below_threshold = np.ma.masked_where(z_diff >= thresh, z_diff)
z_above_threshold = np.ma.masked_where(z_diff < thresh, z_diff)

# Plotting Z displacement as continuous lines
time = np.arange(len(z_diff))  # Assuming time steps are equal to the index of z_diff
ax1.plot(time, z_below_threshold, 'b', label='Below threshold')  # Plot as blue line
ax1.plot(time, z_above_threshold, 'r', label='Above threshold')  # Plot as red line

ax1.set_xlabel('Time')
ax1.set_ylabel('Z displacement')
# force
ax2 = fig.add_subplot(222)
ax2.set_title('Force')
ax2.plot(ft[0], label='Fx')
ax2.plot(ft[1], label='Fy')
ax2.plot(ft[2], label='Fz')
ax2.set_xlabel('Time')
ax2.set_ylabel('Force')
ax2.legend()
# torque
ax3 = fig.add_subplot(223)
ax3.set_title('Torque')
ax3.plot(ft[3], label='Tx')
ax3.plot(ft[4], label='Ty')
ax3.plot(ft[5], label='Tz')
ax3.set_xlabel('Time')
ax3.set_ylabel('Torque')
ax3.legend()

save_path = '/root/Research_Internship_at_GVlab/data0402/fig/step2/data0313_demo_preprocessed_z_diff_ft_{}_{}.png'.format(sponge,trial)
fig.savefig(save_path)
plt.show()
