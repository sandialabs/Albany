#!/usr/bin/python
'''
LCM_postprocess.py input.e output.e

creates usable output from LCM calculations
'''


import sys
import os
from exodus import exodus
from exodus import copy_mesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.linalg import *



def GetWeights(inFile, num_points):

    weights = dict()
    volume = dict()

    for block in inFile.get_elem_blk_ids():

        weights_block = []

        for point in range(num_points):

            key_weights = 'Weights_' + str(point + 1)

            values_weights = inFile.get_element_variable_values(
                block, 
                key_weights, 
                1)

            weights_block.append(values_weights)

        weights[block] = weights_block

        volume[block] = np.sum(weights_block)

    return weights, volume



def VolumeAverageTensor(
    inFile, 
    outFile, 
    num_points, 
    weights, 
    volume, 
    name_variable_base):

    num_dims = inFile.num_dimensions()
    block_ids = inFile.get_elem_blk_ids()

    values = dict()
    average_value = dict()

    for step in range(len(inFile.get_times())):
        
        average_value_component_step = dict()

        values_step = dict()

        for block in block_ids:

            values_block = dict()

            for dim_i in range(num_dims):

                for dim_j in range(num_dims):

                    index_dim = num_dims * dim_i + dim_j + 1

                    values_component = []

                    for point in range(num_points):

                        values_weights = weights[block][point]

                        index_pt = point * num_dims**2 + index_dim

                        if index_pt < 10:
                            str_index = '0' + str(index_pt)
                        else:
                            str_index = str(index_pt)

                        key_stress = name_variable_base + '_' + str_index

                        values_stress = inFile.get_element_variable_values(
                            block, 
                            key_stress, 
                            step + 1)
                    
                        values_component.extend(
                            [x*y for (x,y) in zip(values_stress, values_weights)])

                    name_component = name_variable_base + '_' + str(dim_i+1) + str(dim_j+1)

                    average_component = np.sum(values_component) / volume[block]

                    # output the volume-averaged quantity
                    outFile.put_element_variable_values(
                        block, 
                        name_component, 
                        step + 1, 
                        average_component * np.ones(len(values_component)))

                    values_block[(dim_i, dim_j)] = average_component

            values_step[block] = values_block

        # end for block in block_ids:

        values[step] = values_step

        for dim_i in range(num_dims):

            for dim_j in range(num_dims):

                average_value_component_step[(dim_i, dim_j)] = \
                    np.sum([values[step][x][(dim_i, dim_j)] * y 
                        for (x, y) in zip(block_ids, volume.values())])

        average_value[step] = average_value_component_step

    # end for step in range(len(times)):

    return average_value

# end def VolumeAverageTensor(weights, name_variable_base):



def postprocess(inFileName, outFileName):

    debug = 0

    # set i/o units 
    inFile = exodus(inFileName,"r")
    if os.path.isfile(outFileName):
      cmdLine = "rm %s" % outFileName
      os.system(cmdLine)
    outFile = copy_mesh(inFileName, outFileName)


    # get number of dimensions
    num_dims = inFile.num_dimensions()

    print "Dimensions"
    print num_dims


    # get times
    times = inFile.get_times()
    num_times = len(times)
    print "Print the times"
    print times[0], times[-1]


    # get list of nodal variables
    node_var_names= inFile.get_node_variable_names()
    print "Printing the names of the nodal variables"
    print node_var_names


    # get list of element variables
    elem_var_names = inFile.get_element_variable_names()
    num_variables_unique = 0
    names_variable_unique = []
    print "Printing the names of the element variables"
    for name in elem_var_names:
        new_name = True
        for name_unique in names_variable_unique:
          if (name.startswith(name_unique+"_")):
            new_name = False
        if (new_name == True):
          indices = [int(s) for s in name if s.isdigit()]
          names_variable_unique.append(name[0:name.find(str(indices[0]))-1])
    print names_variable_unique


    # get number of element blocks and block ids
    block_ids = inFile.get_elem_blk_ids()
    num_blocks = inFile.num_blks()

    # figure out number of integration points
    num_points = 0
    for name in elem_var_names:
      if (name.startswith("Weights_")):
          num_points += 1

    # check that "Weights" exist as an element variable
    if (num_points == 0):
      raise Exception("The weights field is not available...try again.")
          
    print "Number of Integration points"
    print num_points

    # write times to outFile
    for step in range(len(times)):
      outFile.put_time(step+1,times[step])

    # write out displacement vector
    dx = []
    dy = []
    dz = []

    outFile.set_node_variable_number(int(3))
    outFile.put_node_variable_name('displacement_x', 1)
    outFile.put_node_variable_name('displacement_y', 2)
    outFile.put_node_variable_name('displacement_z', 3)

    for step in range(len(times)):

      dx = inFile.get_node_variable_values('displacement_x',step+1)
      dy = inFile.get_node_variable_values('displacement_y',step+1)
      dz = inFile.get_node_variable_values('displacement_z',step+1)

      outFile.put_node_variable_values('displacement_x',step+1,dx)
      outFile.put_node_variable_values('displacement_y',step+1,dy)
      outFile.put_node_variable_values('displacement_z',step+1,dz)

    #
    # create variables in output file
    #
    outFile.set_element_variable_number(2 * num_dims**2)

    for dim_i in range(num_dims):

        for dim_j in range(num_dims):

            outFile.put_element_variable_name(
                'Cauchy_Stress_' + str(dim_i + 1) + str(dim_j + 1), 
                dim_i * num_dims + dim_j + 1)

            outFile.put_element_variable_name(
                'F_' + str(dim_i + 1) + str(dim_j + 1), 
                num_dims**2 + dim_i * num_dims + dim_j + 1)

    #
    # Record the integration point weights
    #
    weights, volume = GetWeights(inFile, num_points)

    print 'Volume: ', np.sum(volume.values())

    #
    # search through variables, find desired quantities, and output them
    #
    for name in names_variable_unique:

        #
        # handle the cauchy stress
        #
        if (name == 'Cauchy_Stress'):

            print name

            stress_cauchy = VolumeAverageTensor(
                inFile,
                outFile,
                num_points,
                weights,
                volume,
                name)

        #
        # handle the deformation gradient
        #
        if (name == 'F'):

            print name

            defgrad = VolumeAverageTensor(
                inFile,
                outFile,
                num_points,
                weights,
                volume,
                name)

        # end for name in names_variable_unique:

    outFile.close()

    strain = dict()
    for step in range(num_times):
        defgrad_step = np.zeros((3,3))
        for dim_i in range(num_dims):
            for dim_j in range(num_dims):
                defgrad_step[dim_i][dim_j] = defgrad[step][(dim_i, dim_j)]
        strain[step] = 0.5 * logm(np.transpose(defgrad_step) * defgrad_step)


    rcParams.update({'figure.autolayout': True})

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=22)

    fig = plt.figure()

    for dim_i in range(num_dims):

        for dim_j in range(num_dims):

            fig.clf()
            plt.plot(
                [strain[step][dim_i][dim_j] for step in range(num_times)],
                [stress_cauchy[step][(dim_i, dim_j)] for step in range(num_times)])
            plt.xlabel('Logarithmic Strain $\epsilon_{'+ str(dim_i + 1) + str(dim_j + 1) +'}$')
            plt.ylabel('Cauchy Stress $\sigma_{'+ str(dim_i + 1) + str(dim_j + 1) +'}$ (MPa)')
            plt.savefig('stress_strain_'+ str(dim_i + 1) + str(dim_j + 1) +'.pdf')

# end def postprocess(inFileName, outFileName):



if __name__ == '__main__':
    inFileName = sys.argv[1]
    outFileName = sys.argv[2]
    postprocess(inFileName, outFileName)