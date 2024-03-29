import numpy as np
import matplotlib.pyplot as plt
z_diff_data_path = '/root/Research_Internship_at_GVlab/real/step2/data/demo_preprocessed_z_diff.npz'
ft_data_path = '/root/Research_Internship_at_GVlab/real/step2/data/demo_preprocessed_ft.npz'
sponge = input('sponge: ')
trial = int(input('trial: '))
z_diff = np.load(z_diff_data_path)[sponge][trial]#(99,)
print(z_diff.shape)
ft = np.load(ft_data_path)[sponge][trial] #(6, 100)
print(ft.shape)
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.set_title('Z displacement')
ax1.plot(z_diff)
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

save_path = '/root/Research_Internship_at_GVlab/fig/demo_preprocessed_z_diff_ft_{}.png'.format(sponge)
fig.savefig(save_path)
plt.show()
