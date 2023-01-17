import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def lin_reg(x, y):
    m = (len(x) * np.sum(x*y) - np.sum(x) * np.sum(y)) / (len(x)*np.sum(x*x) - np.sum(x) ** 2)
    b = (np.sum(y) - m *np.sum(x)) / len(x)
    return [m, b]



timer_1 = np.loadtxt('timers_no_interface.txt')
timer_2 = np.loadtxt('timers_with_interface.txt')
timer_3 = np.loadtxt('timers_setup_no_interface.txt')
timer_4 = np.loadtxt('timers_setup_with_interface.txt')
n_max_samples = len(timer_1)

n_samples = np.arange(0, n_max_samples) + 1
n_samples_2 = np.arange(0, n_max_samples+1)

[m_1, b_1] = lin_reg(n_samples, timer_1)
[m_2, b_2] = lin_reg(n_samples, timer_2)
[m_3, b_3] = lin_reg(n_samples, timer_3)
[m_4, b_4] = lin_reg(n_samples, timer_4)

fig = plt.figure(figsize=(6,4))
plt.plot(n_samples, timer_1, '.', label='Total cost without interface')
plt.plot(n_samples, timer_2, '.', label='Total cost with interface')
plt.plot(n_samples, timer_3, '.', label='Setup cost without interface')
plt.plot(n_samples, timer_4, '.', label='Setup cost with interface')

plt.gca().set_prop_cycle(None)

plt.plot(n_samples_2, m_1*n_samples_2+b_1)#, label='Linear regression without interface')
plt.plot(n_samples_2, m_2*n_samples_2+b_2)#, label='Linear regression with interface')
plt.plot(n_samples_2, m_3*n_samples_2+b_3)#, label='Linear regression setup cost without interface')
plt.plot(n_samples_2, m_4*n_samples_2+b_4)#, label='Linear regression setup cost with interface')

delta_time = 5
plt.ylabel('Cumulated time [sec]')
plt.xlabel('Number of samples of the MC simulation')
plt.gca().set_xlim([0, n_max_samples])
plt.gca().set_ylim([0, np.ceil(np.amax([timer_1, timer_2])/delta_time)*delta_time])
#plt.gca().set_ylim([0, 2.7e-3])
plt.grid(True, which="both")
plt.legend()
fig.tight_layout()
plt.savefig('perf_MC.jpeg', dpi=800)
plt.close()
