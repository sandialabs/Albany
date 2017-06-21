#!/usr/bin/env python

import numpy
from scipy import linalg
import sys
import string

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "\nUsage:  RandomRotation.py <num_blocks>\n"
        sys.exit(1)


    number_blocks = int(sys.argv[1])

    # Exporting data to file
    output = open('rotation_matrices.txt','w')
    output.write('#\n')
    output.write('# Rotation matrix for each block in a genesis file.\n')
    output.write('# Format is:  R11 R12 R13 R21 R22 R23 R31 R32 R33.\n')
    output.write('#\n')

    # Checking distribution of sphere point picking algorithm
    output_check = open('axial_vectors.txt','w')

    for i in range(number_blocks):

        # Create uniform distribution
        # Please refer to sphere point picking algorithms
        # source: http://mathworld.wolfram.com/SpherePointPicking.html

        # Sample uniform distribution U(0,1) to get random numbers u and v
        u = numpy.random.uniform(0.0, 1.0)
        v = numpy.random.uniform(0.0, 1.0)

        # calculate uniformly distributed spherical coordinates
        theta = 2.0 * numpy.pi * u 
        phi = numpy.arccos(2.0 * v - 1.0)

        # convert to rectangular coordinates
        axial_vector_1 = numpy.cos(theta)*numpy.sin(phi)
        axial_vector_2 = numpy.sin(theta)*numpy.sin(phi)
        axial_vector_3 = numpy.cos(phi)
        axial_vector =  [axial_vector_1, axial_vector_2, axial_vector_3]

        # Ensure that axial vector is a unit vector
        norm_axial_vector = numpy.linalg.norm(axial_vector)
        axial_vector = axial_vector / norm_axial_vector

        skew_symmetric = [[0.0, -axial_vector[2], axial_vector[1]],
                          [axial_vector[2], 0.0, -axial_vector[0]],
                          [-axial_vector[1], axial_vector[0], 0.0]]

        rotation_angle = numpy.random.uniform(0.0, 2.0*numpy.pi)

        # add rotation to the axial vector

        skew_symmetric_rotation = numpy.multiply(skew_symmetric, rotation_angle)

        rotation = linalg.expm(skew_symmetric_rotation)

        # write data to file
        s = '{0:.15f}  {1:.15f}  {2:.15f}  {3:.15f}  {4:.15f}  {5:.15f}  {6:.15f}  {7:.15f}  {8:.15f}\n'.format(
            rotation[0,0],rotation[0,1],rotation[0,2],
            rotation[1,0],rotation[1,1],rotation[1,2],
            rotation[2,0],rotation[2,1],rotation[2,2])
        output.write(s)
        
        s = '{0:.15f}  {1:.15f}  {2:.15f}\n'.format(axial_vector_1,axial_vector_2,axial_vector_3)
        output_check.write(s)

    output.close()
    output_check.close()

    print "\nWrote", number_blocks, "rotation matrices to rotation_matrices.txt\n"
    print "\nWrote", number_blocks, "axial vectors to axial_vector.txt\n" 

# Rotation matrix for each block in a genesis file.
# Format is:  R11 R12 R13 R21 R22 R23 R31 R32 R33
# print axial_vector
# print numpy.dot(axial_vector, axial_vector)
# print skew_symmetric
# print rotation
# print numpy.linalg.det(rotation)
# print numpy.dot(numpy.transpose(rotation),rotation)

