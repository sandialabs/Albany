import scipy
import numpy
import scipy.io
import scipy.linalg
import matplotlib.pyplot as plt

## Set the number of matrices to examine. 
## This can be automated in the future by parsing jac*.mm
## in directory.

numberMatrices = 14

condNum = numpy.zeros(numberMatrices)

for i in range(numberMatrices):

    filename = 'jac' + str(i) + '.mm'
    a = scipy.io.mmread(filename)
    b = a.todense()
    eigenvalues = scipy.linalg.eigvals(b)
    realEigenvalues = eigenvalues.real
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
