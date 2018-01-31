#!/usr/bin/env python

import argparse
import numpy
import os
from scipy import linalg
import sys
import string
from lcm_preprocess.write_file_material import read_names_block

def create_rotations_random(num_rotations=1, basename=''):

    # Exporting data to file
    filename_rotations = basename + 'Rotations.txt'
    output = open(filename_rotations,'w')
    output.write('#\n')
    output.write('# Rotation matrix for each block in a genesis file.\n')
    output.write('# Format is:  R11 R12 R13 R21 R22 R23 R31 R32 R33.\n')
    output.write('#\n')

    # Checking distribution of sphere point picking algorithm
    output_check = open('axial_vectors.txt','w')

    for i in range(num_rotations):

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

    print '\nWrote', num_rotations, 'rotation matrices to {}\n'.format(filename_rotations)
    print '\nWrote', num_rotations, 'axial vectors to axial_vector.txt\n'

    return True

# Rotation matrix for each block in a genesis file.
# Format is:  R11 R12 R13 R21 R22 R23 R31 R32 R33
# print axial_vector
# print numpy.dot(axial_vector, axial_vector)
# print skew_symmetric
# print rotation
# print numpy.linalg.det(rotation)
# print numpy.dot(numpy.transpose(rotation),rotation)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('num_blocks', help='Specify number of blocks (old style)', nargs='*')
    parser.add_argument('-f', '--filename', help = 'Specify name of exodus file.')
    parser.add_argument('-n', '--num_rotations', help = 'Specify number of rotations.')

    args = parser.parse_args()
    filename = getattr(args, 'filename', None)
    if filename is not None:
        assert(os.path.isfile(filename))
        num_rotations = len(read_names_block(filename))
        basename = os.path.splitext(filename)[0] + '_'
    else:
        basename = ''
        num_rotations = getattr(args, 'num_rotations', None)
        if num_rotations is None:
            num_rotations = getattr(args, 'num_blocks', None)
            if num_rotations is None:
                parser.print_help()
                sys.exit(1)

    success = create_rotations_random(num_rotations=num_rotations, basename=basename)
