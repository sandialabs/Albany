import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

axial_vectors = numpy.loadtxt('axial_vectors.txt')

#print axial_vectors


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(axial_vectors[:,0],axial_vectors[:,1],axial_vectors[:,2],color='blue',marker='o')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
