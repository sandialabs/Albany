#!/usr/bin/env python

import sys
sys.path.append('/ascldap/users/djlittl/ATDM/seacas/seacas_thread_safe_gcc_5.4.0/lib')
import exodus
import string
import math

def analytic_soln(x, y, z):

    U11 = 1.0
    soln = [0.0, 0.0, 0.0]
    soln[0] = U11*x*x

    return soln

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "\nUsage:  OBC_Convergence_PostProcess.py <exodus_file_name>\n"
        sys.exit(1)

    inFileName = sys.argv[1]

    inFile = exodus.exodus(inFileName, mode='r')

    # Print database parameters from inFile
    print " "
    print "Database version:         " + str(round(inFile.version.value,2))
    print "Database title:           " + inFile.title()
    print "Database dimensions:      " + str(inFile.num_dimensions())
    print "Number of nodes:          " + str(inFile.num_nodes())
    print "Number of elements:       " + str(inFile.num_elems())
    print "Number of element blocks: " + str(inFile.num_blks())
    print "Number of node sets:      " + str(inFile.num_node_sets())
    print "Number of side sets:      " + str(inFile.num_side_sets())
    print "Number of time steps:     " + str(inFile.num_times())
    print " "

    numNodes = inFile.num_nodes()
    nodeVariableNames = inFile.get_node_variable_names()
    if 'DisplacementX' not in nodeVariableNames:
        print "\nERROR:  Failed to extract DisplacementX data\n"
        sys.exit(1)

    coord_x, coord_y, coord_z = inFile.get_coords()

    time_step = inFile.num_times()
    disp_x = inFile.get_node_variable_values('DisplacementX', time_step)
    disp_y = inFile.get_node_variable_values('DisplacementY', time_step)
    disp_z = inFile.get_node_variable_values('DisplacementZ', time_step)

    block_id = 1
    volume = inFile.get_element_variable_values(block_id, 'Volume', time_step)
    vol = volume[0]
    for v in volume:
        diff = math.sqrt((vol-v)*(vol-v))
        if diff > 1.0e-12:
            print "ERROR, sphere volumes are not constant"
            sys.exit(1)

    mesh_size = math.pow(vol, 1.0/3.0)

    num_nodes_with_zero_error = 0

    error_norm = 0.0
    for i in range(inFile.num_nodes()):
        computational_solution = [disp_x[i], disp_y[i], disp_z[i]]
        analytic_solution = analytic_soln(coord_x[i], coord_y[i], coord_z[i])

        error = vol * ( (analytic_solution[0] - computational_solution[0])*(analytic_solution[0] - computational_solution[0]) +
                        (analytic_solution[1] - computational_solution[1])*(analytic_solution[1] - computational_solution[1]) +
                        (analytic_solution[2] - computational_solution[2])*(analytic_solution[2] - computational_solution[2]) )

        if error < 1.0e-30:
            num_nodes_with_zero_error += 1
#        else:
#            print computational_solution, analytic_solution, vol, mesh_size, error, error/vol, computational_solution[0] -  analytic_solution[0], computational_solution[1] -  analytic_solution[1], computational_solution[2] -  analytic_solution[2]

        error_norm += error

    error_norm = math.sqrt(error_norm)

    peridigm_error = inFile.get_element_variable_values(block_id, 'Error', time_step)
    peridigm_error_norm = 0.0
    for val in peridigm_error:
        peridigm_error_norm += val
    peridigm_error_norm = math.sqrt(peridigm_error_norm)

    print "NUM NODES", inFile.num_nodes(), " NUM ZERO ERROR", num_nodes_with_zero_error, "FRACTION",  float(num_nodes_with_zero_error)/inFile.num_nodes(), "EXPECTED FRACTION", ((0.05*20)*(0.05*10)*(0.05*10) - (0.05*16)*(0.05*6)*(0.05*6))/((0.05*20)*(0.05*10)*(0.05*10))

    inFile.close()

    outFileLabel = string.splitfields(inFileName, '.')[0] + "_error_norm.txt"
    outFile = open(outFileLabel, 'w')
    outFile.write(str(mesh_size) + " " + str(error_norm) + " " + str(peridigm_error_norm) + " " + str(math.log10(mesh_size)) + " " + str(math.log10(error_norm)) + " " + str(math.log10(peridigm_error_norm)) + "\n")
    outFile.close()

    print "\nError data:", mesh_size, error_norm, peridigm_error_norm, math.log10(mesh_size), math.log10(error_norm), math.log10(peridigm_error_norm)
    print "Error norm written to", outFileLabel, "\n"



