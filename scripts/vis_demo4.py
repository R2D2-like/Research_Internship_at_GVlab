import numpy as np
import matplotlib.pyplot as plt
data_path = '/root/Research_Internship_at_GVlab/real/step2/data/demo_preprocessed4.npz'
data = np.load(data_path)['s2f2']
z_diff = data['z_diff'][0] #(99,)
ft = data['ft'][0] #(6, 100)
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
