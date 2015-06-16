# Script creates random rotations

import numpy
from scipy import linalg
import sys
import string

number_blocks = 27


# Exporting data to file
output = open('rotation_matrices.txt','w')
output.write('#\n')
output.write('# Rotation matrix for each block in a genesis file.\n')
output.write('# Format is:  R11 R12 R13 R21 R22 R23 R31 R32 R33.\n')
output.write('#\n')

for i in range(27):

    # create axial vector
    # This allows for a large degree misorientation. 
    #
    axial_vector_1 = numpy.random.uniform(-1.0, 1.0)
    axial_vector_2 = numpy.random.uniform(-1.0, 1.0)
    axial_vector_3 = numpy.random.uniform(-1.0, 1.0)
    axial_vector =  [axial_vector_1, axial_vector_2, axial_vector_3]
    norm_axial_vector = numpy.linalg.norm(axial_vector)
    axial_vector = axial_vector / norm_axial_vector

    skew_symmetric = [[0.0, -axial_vector[2], axial_vector[1]],
                      [axial_vector[2], 0.0, -axial_vector[0]],
                      [-axial_vector[1], axial_vector[0], 0.0]]

    rotation_angle = numpy.random.uniform(0.0, 2.0*numpy.pi)

    # add rotation to the axial vector

    skew_symmetric_rotation = numpy.multiply(skew_symmetric, rotation_angle)

    rotation = linalg.expm(skew_symmetric_rotation)

    # writing data to file
    s = '{0:.15f}  {1:.15f}  {2:.15f}  {3:.15f}  {4:.15f}  {5:.15f}  {6:.15f}  {7:.15f}  {8:.15f}\n'.format(
          rotation[0,0],rotation[0,1],rotation[0,2],
          rotation[1,0],rotation[1,1],rotation[1,2],
          rotation[2,0],rotation[2,1],rotation[2,2])
    output.write(s)

output.close()

# Rotation matrix for each block in a genesis file.
# Format is:  R11 R12 R13 R21 R22 R23 R31 R32 R33
# print axial_vector
# print numpy.dot(axial_vector, axial_vector)
# print skew_symmetric
# print rotation
# print numpy.linalg.det(rotation)
# print numpy.dot(numpy.transpose(rotation),rotation)

