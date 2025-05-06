import numpy as np
import matplotlib.pyplot as plt

value_base = np.load(
    '/home/rustam/PR/base_DDPG_2.npy')
values_her = np.load(
    '/home/rustam/PR/DDPG_HER_not_terminated_0.02.npy')
values_her_5 = np.load(
    '/home/rustam/PR/DDPG_HER_not_terminated.npy')
index = np.array([i for i in range(30)])
plt.axis([0, 29, 0, 1])
plt.title("Results")
plt.xlabel("epoch")
plt.ylabel("success_rate")
plt.plot(index, value_base, color="red", label='base_DDPG')
plt.plot(index, values_her, color="blue", label='DDPG+HER, diametr=0.02')
plt.plot(index, values_her_5, color="green", label='DDPG+HER, diametr=0.05')
plt.legend()
plt.show()
