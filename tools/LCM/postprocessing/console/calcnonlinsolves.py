
import numpy as np
import matplotlib.pyplot as plt

nonlin = open('nonlinsolves', 'r')
data = np.genfromtxt(nonlin, delimiter='')
j = 1;
count = [];  
for i in range(0, len(data)):
  if (data[i,0] == 1e7):
    count.append(i); 
    j = j+1; 
cells = [];
cells.append(1); 
for i in range(1, len(count)):
  cells.append(len(data[count[i-1]+1:count[i],0])); 
print '# nonlinear solves per loca step:', cells
print 'Average # nonlinear solves (not including first step): ', float(sum(cells[1:]))/float(len(cells[1:]))
plt.plot(cells, 'o')
plt.ylabel('# nonlinear solves')
plt.xlabel('time step')
plt.show()
