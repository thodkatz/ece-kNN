from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np


data_seq = genfromtxt('../../hpc/slurm/features/output_seq.txt', delimiter=' ')
log_v1 = genfromtxt('../../hpc/slurm/features/v1/output.txt', delimiter=' ')
log_v2 = genfromtxt('../../hpc/slurm/features/v2/output.txt', delimiter=' ')

processes = [8, 12, 16, 20, 24]
neighbors = [10, 40, 70, 100]

data_v0 = np.arange(4, dtype = np.double);
count = -1;
for i in range(len(data_seq)):
    if(i%2 == 1):
        count += 1
        data_v0[count] = data_seq[i]

print(data_v0)

# proc x k
data_v1 = np.arange(20, dtype = np.double).reshape(5,4)
data_v2 = np.arange(20, dtype = np.double).reshape(5,4)

for i in range(len(log_v1)):
    if(i%12 == 0):
        count = 0
    if(i%3 == 0):
        id_proc = int((log_v1[i]-4)/4 -1)
    if(i%3 == 1):
        k = log_v1[i]
    if(i%3 == 2):
        data_v1[id_proc][count] = log_v1[i]
        count += 1

for i in range(len(log_v2)):
    if(i%12 == 0):
        count = 0
    if(i%3 == 0):
        id_proc = int((log_v2[i]-4)/4 -1)
    if(i%3 == 1):
        k = log_v2[i]
    if(i%3 == 2):
        data_v2[id_proc][count] = log_v2[i]
        count += 1



print(data_v1)
print(data_v2)

mean_v0 = np.mean(data_v0)
mean_v1 = np.mean(data_v1, axis=1)
print("\n")
print(mean_v1)
print("\n")
mean_v2 = np.mean(data_v2, axis=1)


plt.subplots_adjust(left=0.11, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.4)

plt.subplot(311)
plt.plot(1, mean_v0, 'o', label = 'V0')
plt.plot(processes, mean_v1,linestyle='--', marker= 'o', label = 'V1')
plt.plot(processes, mean_v2,linestyle='--', marker= 'o', label = 'V2')

plt.xlabel('Processors', fontsize=12, labelpad=10)
plt.ylabel("Time (s)", fontsize=12, labelpad=10)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True);
plt.xticks(np.arange(0, 26, 2))
#plt.yticks(np.arange(0, 40, 5))

plt.subplot(312)
plt.plot(neighbors, data_v1[0], 'o', linestyle='--', marker= 'o',label = '4')
plt.plot(neighbors, data_v1[1], 'o', linestyle='--', marker= 'o',label = '8')
plt.plot(neighbors, data_v1[2], 'o', linestyle='--', marker= 'o',label = '12')
plt.plot(neighbors, data_v1[3], 'o', linestyle='--', marker= 'o',label = '16')
plt.plot(neighbors, data_v1[4], 'o', linestyle='--', marker= 'o',label = '20')

plt.xlabel('k', fontsize=12, labelpad=10)
plt.ylabel("V1 Time (s)", fontsize=12, labelpad=10)
#plt.legend()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True);
plt.xticks(np.arange(0, 110, 10))
#plt.yticks(np.arange(0, 14, 1))

plt.subplot(313)
plt.plot(neighbors, data_v2[0], 'o', linestyle='--', marker= 'o',label = '4')
plt.plot(neighbors, data_v2[1], 'o', linestyle='--', marker= 'o',label = '8')
plt.plot(neighbors, data_v2[2], 'o', linestyle='--', marker= 'o',label = '12')
plt.plot(neighbors, data_v2[3], 'o', linestyle='--', marker= 'o',label = '16')
plt.plot(neighbors, data_v2[4], 'o', linestyle='--', marker= 'o',label = '20')

plt.xlabel('k', fontsize=12, labelpad=10)
plt.ylabel("V2 Time (s)", fontsize=12, labelpad=10)
#plt.legend()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True);
plt.xticks(np.arange(0, 110, 10))
#plt.yticks(np.arange(0, 10, 1))


plt.show()

