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




    # create any global variables


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


    # create variables in output file

    outFile.set_element_variable_number(int(18))
    outFile.put_element_variable_name('stress_cauchy_11', 1)
    outFile.put_element_variable_name('stress_cauchy_12', 2)
    outFile.put_element_variable_name('stress_cauchy_13', 3)
    outFile.put_element_variable_name('stress_cauchy_21', 4)
    outFile.put_element_variable_name('stress_cauchy_22', 5)
    outFile.put_element_variable_name('stress_cauchy_23', 6)
    outFile.put_element_variable_name('stress_cauchy_31', 7)
    outFile.put_element_variable_name('stress_cauchy_32', 8)
    outFile.put_element_variable_name('stress_cauchy_33', 9)
    outFile.put_element_variable_name('F_11', 10)
    outFile.put_element_variable_name('F_12', 11)
    outFile.put_element_variable_name('F_13', 12)
    outFile.put_element_variable_name('F_21', 13)
    outFile.put_element_variable_name('F_22', 14)
    outFile.put_element_variable_name('F_23', 15)
    outFile.put_element_variable_name('F_31', 16)
    outFile.put_element_variable_name('F_32', 17)
    outFile.put_element_variable_name('F_33', 18)



    #
    # search through variables, find desired quantities, and output them
    #
    for name in names_variable_unique:

        #
        # handle the cauchy stress
        #
        if (name == 'Cauchy_Stress'):

            for step in range(len(times)):

                for block in block_ids:

                    stress_cauchy = []

                    for dim_i in range(num_dims):

                        for dim_j in range(num_dims):

                            index_dim = num_dims * dim_i + dim_j + 1

                            stress_cauchy_component = []

                            for point in range(num_points):

                                key_weights = 'Weights_' + str(point + 1)

                                values_weights = inFile.get_element_variable_values(
                                    block, key_weights, step + 1)

                                index_pt = point * num_dims**2 + index_dim

                                if index_pt < 10:
                                    str_index = '0' + str(index_pt)
                                else:
                                    str_index = str(index_pt)

                                key_stress = 'Cauchy_Stress_' + str_index

                                values_stress = \
                                    inFile.get_element_variable_values(block, key_stress, step+1)
                            
                                stress_cauchy_component.extend(
                                    [x*y for (x,y) in zip(values_stress, values_weights)])

                            if debug != 0:
                                print 'stress_cauchy_'+str(dim_i+1)+str(dim_j+1)
                                print np.sum(stress_cauchy_component)

                            # output the volume-averaged quantity
                            outFile.put_element_variable_values(
                                block,'stress_cauchy_'+str(dim_i+1)+str(dim_j+1),step+1,np.sum(stress_cauchy_component)*np.ones(len(stress_cauchy_component)))

                            stress_cauchy.append(stress_cauchy_component)

                    if debug != 0:
                        print 'Cauchy_Stress', np.shape(stress_cauchy)
                        print stress_cauchy

                # end for step in range(len(times)):

            # end if (name == 'Cauchy_Stress'):

        #
        # handle the deformation gradient
        #
        if (name == 'F'):

            for step in range(len(times)):

                for block in block_ids:

                    defgrad = []

                    for dim_i in range(num_dims):

                        for dim_j in range(num_dims):

                            index_dim = num_dims * dim_i + dim_j + 1

                            defgrad_component = []

                            for point in range(num_points):

                                key_weights = 'Weights_' + str(point + 1)

                                values_weights = inFile.get_element_variable_values(
                                    block, key_weights, step + 1)

                                index_pt = point * num_dims**2 + index_dim

                                if index_pt < 10:
                                    str_index = '0' + str(index_pt)
                                else:
                                    str_index = str(index_pt)

                                key_defgrad = 'F_' + str_index

                                values_defgrad = \
                                    inFile.get_element_variable_values(block, key_defgrad, step+1)
                            
                                defgrad_component.extend(
                                    [x*y for (x,y) in zip(values_defgrad, values_weights)])

                            if debug != 0:
                                print 'defgrad_'+str(dim_i+1)+str(dim_j+1)
                                print np.sum(defgrad_component)

                            # output the volume-averaged quantity
                            outFile.put_element_variable_values(
                                block,'F_'+str(dim_i+1)+str(dim_j+1),step+1,np.sum(defgrad_component)*np.ones(len(defgrad_component)))

                            defgrad.append(defgrad_component)

                    if debug != 0:
                        print 'defgrad', np.shape(defgrad)
                        print defgrad

                # end for step in range(len(times)):

            # end if (name == 'F'):

        # end for name in names_variable_unique:

    outFile.close()

    # end def postprocess(inFileName, outFileName):

if __name__ == '__main__':
    inFileName = sys.argv[1]
    outFileName = sys.argv[2]
    postprocess(inFileName, outFileName)