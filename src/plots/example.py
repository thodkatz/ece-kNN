from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np


data_seq = genfromtxt('../plots/slurm/v3/NACA0015/output_seq.txt', delimiter=' ')
data_cilk_t = genfromtxt('../plots/slurm/v3/NACA0015/cilk/output.txt', delimiter=' ')
data_openmp_t = genfromtxt('../plots/slurm/v3/NACA0015/openmp/output.txt', delimiter=' ')
data_openmp_static_t = genfromtxt('../plots/slurm/v3/NACA0015/openmp/static/output.txt', delimiter=' ')

threads_cilk = []
data_cilk = []
for i in range(len(data_cilk_t)):
    if (i%2 == 1): 
        data_cilk.append(data_cilk_t[i])
    if (i%2 == 0):
        threads_cilk.append(data_cilk_t[i])

threads_openmp = []
data_openmp = []
for i in range(len(data_openmp_t)):
    if (i%2 == 1): 
        data_openmp.append(data_openmp_t[i])
    if (i%2 == 0):
        threads_openmp.append(data_openmp_t[i])

threads_openmp_static = []
data_openmp_static = []
for i in range(len(data_openmp_static_t)):
    if (i%2 == 1): 
        data_openmp_static.append(data_openmp_static_t[i])
    if (i%2 == 0):
        threads_openmp_static.append(data_openmp_static_t[i])

plt.plot(threads_cilk, data_cilk, 'o', label = "OpenCilk")
plt.plot(threads_openmp, data_openmp, 'o', label = "OpenMP dynamic")
plt.plot(threads_openmp_static, data_openmp_static, 'o', label = "OpenMP static")
plt.plot(1, data_seq, 'o', label = "Serial")

plt.xlabel('Threads', fontsize=12, labelpad=10)
plt.ylabel("Time (s)", fontsize=12, labelpad=10)
plt.legend()
plt.grid(True);
plt.xticks(np.arange(0, 22, 2))
plt.yticks(np.arange(0, 1.45, 0.05))

#plt.title('v3 binary search NACA0015')
plt.show()
