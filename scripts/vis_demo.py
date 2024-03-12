import numpy as np
import matplotlib.pyplot as plt
data_path = '/root/Research_Internship_at_GVlab/real/step2/data/demo_preprocessed3.npz'
data = np.load(data_path)['s2f2'][0] #(9, 100)
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.set_title('Position')
ax1.plot(data[0], label='x')
ax1.plot(data[1], label='y')
ax1.plot(data[2], label='z')
ax1.set_xlabel('Time')
ax1.set_ylabel('Position')
ax1.legend()
# force
ax2 = fig.add_subplot(222)
ax2.set_title('Force')
ax2.plot(data[3], label='Fx')
ax2.plot(data[4], label='Fy')
ax2.plot(data[5], label='Fz')
ax2.set_xlabel('Time')
ax2.set_ylabel('Force')
ax2.legend()
# torque
ax3 = fig.add_subplot(223)
ax3.set_title('Torque')
ax3.plot(data[6], label='Tx')
ax3.plot(data[7], label='Ty')
ax3.plot(data[8], label='Tz')
ax3.set_xlabel('Time')
ax3.set_ylabel('Torque')
ax3.legend()

save_path = '/root/Research_Internship_at_GVlab/fig/demo_preprocessed3_s2f2.png'
fig.savefig(save_path)
plt.show()


