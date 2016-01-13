
import numpy as np
a = open('linearsolves', 'r')
data = np.genfromtxt(a, delimiter='')
print 'Total number linear iters: ', sum(data[:,0])
print 'Average number linear iters: ', sum(data[:,0])/len(data[:,0])
a.close()
