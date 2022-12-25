from mpi4py import MPI
import numpy as np
import subprocess
from numpy.random import default_rng
import os
import sys
import time

start = time.time()

# Read in the file
with open('input_scalar_to_update.yaml', 'r') as file :
  filedata = file.read()

N = 50
p_min = -2.
p_max = 2.

# Generate N samples randomly chosen in [p_min, p_max]:
rng = default_rng(seed=42)
p = rng.uniform(p_min, p_max, N)
QoI = np.empty((N,))
timers = np.empty((N,))
timers_setup = np.empty((N,))

# Loop over the N samples and evaluate the quantity of interest:
for i in range(0, N):

    filedata_tmp = filedata.replace('$UPDATE', str(p[i]))

    with open('input_scalar_updated.yaml', 'w') as file:
        file.write(filedata_tmp)

    subprocess.call("Albany input_scalar_updated.yaml &> log.txt", shell=True)
    for line in reversed(list(open("log.txt"))):
        words = line.split()
        if len(words) > 3 and words[0:3] == ['Albany', 'Total', 'Time:']:
            if i == 0:
                timers[i] = float(words[3])
            else:
                timers[i] = float(words[3]) + timers[i-1]
        if len(words) > 4 and words[1:4] == ['Albany:', 'Setup', 'Time:']:
            if i == 0:
                timers_setup[i] = float(words[4])
            else:
                timers_setup[i] = float(words[4]) + timers_setup[i-1]
        if len(words) > 1 and words[0] == "Response[0]":
            QoI[i] = float(words[2])
            break

print(p)
print(QoI)
print(timers)

end = time.time()
elapsed_time = end-start
print(str(timers[-1])+" "+str(elapsed_time)+" "+str(100*timers[-1]/elapsed_time)+"%")
np.savetxt('timers_no_interface.txt', timers)
np.savetxt('timers_setup_no_interface.txt', timers_setup)
