import os
import subprocess
import scipy
import numpy
import scipy.io
import scipy.linalg
import matplotlib.pyplot as plt

## Set the number of matrices to examine. 
filenames = subprocess.check_output("ls -tr *.mm",shell=True)[:-1]
filenames_split = str.split(filenames)
numberMatrices = len(filenames_split)

condNum = numpy.zeros(numberMatrices)
minEigenvalue = numpy.zeros(numberMatrices)

for i in range(numberMatrices):

    filename = 'jac' + str(i) + '.mm'
    a = scipy.io.mmread(filename)
    b = a.todense()
    eigenvalues = scipy.linalg.eigvals(b)
    realEigenvalues = eigenvalues.real
    minEigenvalue[i] = numpy.amin(realEigenvalues)
    condNum[i] = numpy.log10(numpy.amax(realEigenvalues)/numpy.amin(realEigenvalues))

###############
fig, ax = plt.subplots()
ax.plot(range(numberMatrices),condNum[:],color='blue',marker='o',label='scipy')
plt.xlabel('tangents')
plt.ylabel('log10(condition number)')
lg = plt.legend(loc = 3)
lg.draw_frame(False)
#plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.plot(range(numberMatrices),minEigenvalue[:],color='blue',marker='o',label='scipy')
plt.xlabel('tangents')
plt.ylabel('minimum eigenvalue')
lg = plt.legend(loc = 3)
lg.draw_frame(False)
#plt.tight_layout()
plt.show()
