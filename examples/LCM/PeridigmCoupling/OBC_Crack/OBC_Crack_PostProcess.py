#!/usr/bin/env python

# This script requires exodus.py
# It can be tricky to get exodus.py to work because it requires libnetcdf.so and libexodus.so
# Here's one approach that worked on the CEE LAN:
# 1) Append sys.path, as shown below, to include the bin subdirectory of your Trilinos install directory.
#    The file exodus.py is in this directory.
# 2) Edit exodus.py as follows (approximately line 71):
#    accessPth = "/projects/seacas/linux_rhel6/current"
#    The path above is valid on the CEE LAN.  On other systems, you need to provide a path to a SEACAS build
#    that includes shared libraries.
import sys
sys.path.append('/ascldap/users/djlittl/Albany_TPL/trilinos/trilinos-votd/GCC_4.7.2_OPT/bin')
import exodus

import string

if __name__ == "__main__":

    inFileName = "OBC_Crack_Analysis.e"
    inFile = exodus.exodus(inFileName, mode='r')

    outFileLabel = string.splitfields(inFileName, '.')[0]

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
    print " "

    # Extract nodal displacements and forces
    numNodes = inFile.num_nodes()
    numTimeSteps = inFile.num_times()
    nodeVariableNames = inFile.get_node_variable_names()
    coords = inFile.get_coords()
    num_nodes = len(coords[0])
    if 'displacement_x' not in nodeVariableNames:
        print "\nERROR:  Failed to extract displacement_x data\n"
        sys.exit(1)

    print "\nProcessing", num_nodes, "nodes...\n"

    pd_target_y = 0.21875
    pd_target_z = 0.03125
    fem_target_y = 0.25
    fem_target_z = 0.0
    tol = 1.0e-3

    pd_initial_data_top = []
    fem_initial_data_top = []
    pd_initial_data_bottom = []
    fem_initial_data_bottom = []    
    timeStep = 1
    displacement_x = inFile.get_node_variable_values('displacement_x', timeStep)
    for i in range(num_nodes):
        x = coords[0][i]
        y = coords[1][i]
        z = coords[2][i]
        disp = displacement_x[i]

        if abs(y - pd_target_y) < tol and abs(z - pd_target_z) < tol:
            pd_initial_data_top.append([x, y, z, disp])
        if abs(y - fem_target_y) < tol and abs(z - fem_target_z) < tol:
            fem_initial_data_top.append([x, y, z, disp])

        if abs(y + pd_target_y) < tol and abs(z - pd_target_z) < tol:
            pd_initial_data_bottom.append([x, y, z, disp])
        if abs(y + fem_target_y) < tol and abs(z - fem_target_z) < tol:
            fem_initial_data_bottom.append([x, y, z, disp])                        

    pd_final_data_top = []
    fem_final_data_top = []
    pd_final_data_bottom = []
    fem_final_data_bottom = []
    timeStep = numTimeSteps - 1
    displacement_x = inFile.get_node_variable_values('displacement_x', timeStep)
    for i in range(num_nodes):
        x = coords[0][i]
        y = coords[1][i]
        z = coords[2][i]
        disp = displacement_x[i]

        if abs(y - pd_target_y) < tol and abs(z - pd_target_z) < tol:
            pd_final_data_top.append([x, y, z, disp])
        if abs(y - fem_target_y) < tol and abs(z - fem_target_z) < tol:
            fem_final_data_top.append([x, y, z, disp])

        if abs(y + pd_target_y) < tol and abs(z - pd_target_z) < tol:
            pd_final_data_bottom.append([x, y, z, disp])
        if abs(y + fem_target_y) < tol and abs(z - fem_target_z) < tol:
            fem_final_data_bottom.append([x, y, z, disp])                        
                
    inFile.close()

    pd_initial_data_top.sort()
    pd_initial_data_bottom.sort()
    for i in range(len(pd_initial_data_top)):
        pd_initial_data_top[i][3] = 0.0
        pd_initial_data_bottom[i][3] = 0.0
    pd_final_data_top.sort()
    pd_final_data_bottom.sort()

    fem_initial_data_top.sort()
    fem_initial_data_bottom.sort()
    for i in range(len(fem_initial_data_top)):
        if i > 0 and i < len(fem_initial_data_top)-1:
            fem_initial_data_top[i][3] = 0.0
            fem_initial_data_bottom[i][3] = 0.0
    fem_final_data_top.sort()
    fem_final_data_bottom.sort()

    outFile = open(outFileLabel + "_pd_top.txt", 'w')
    for i in range(len(pd_initial_data_top)):
        outFile.write(str(pd_initial_data_top[i][0]) + " " + str(pd_initial_data_top[i][3]) + " " + str(pd_final_data_top[i][3]) + "\n")
    outFile.close()

    outFile = open(outFileLabel + "_pd_bottom.txt", 'w')
    for i in range(len(pd_initial_data_bottom)):
        outFile.write(str(pd_initial_data_bottom[i][0]) + " " + str(pd_initial_data_bottom[i][3]) + " " + str(pd_final_data_bottom[i][3]) + "\n")
    outFile.close()

    outFile = open(outFileLabel + "_fem_top.txt", 'w')
    for i in range(len(fem_initial_data_top)):
        outFile.write(str(fem_initial_data_top[i][0]) + " " + str(fem_initial_data_top[i][3]) + " " + str(fem_final_data_top[i][3]) + "\n")
    outFile.close()

    outFile = open(outFileLabel + "_fem_bottom.txt", 'w')
    for i in range(len(fem_initial_data_bottom)):
        outFile.write(str(fem_initial_data_bottom[i][0]) + " " + str(fem_initial_data_bottom[i][3]) + " " + str(fem_final_data_bottom[i][3]) + "\n")
    outFile.close()    

    print "\nData written to " + outFileLabel + "_pd_top.txt" + " and " + outFileLabel + "_pd_bottom.txt" + " and " + outFileLabel + "_fem_top.txt" + " and " + outFileLabel + "_fem_bottom.txt"
    
    print
    
