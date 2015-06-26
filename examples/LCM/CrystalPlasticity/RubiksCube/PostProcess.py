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

    if len(sys.argv) != 2:
        print "\nUsage:  PostProcess.py <exodus_file_name>\n"
        sys.exit(1)

    inFileName = sys.argv[1]
    inFile = exodus.exodus(inFileName, mode='r')

    outFileLabel = string.splitfields(inFileName, '.')[0] + "_"

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
    if 'displacement_x' not in nodeVariableNames:
        print "\nERROR:  Failed to extract displacement_x data\n"
        sys.exit(1)
    if 'force_x' not in nodeVariableNames:
        print "\nERROR:  Failed to extract force_x data\n"
        sys.exit(1)

    # Read node sets
    nodeSetIds = inFile.get_node_set_ids()
    nodeSetNodes = {}
    for nodeSetId in nodeSetIds:
        nodeIds = inFile.get_node_set_nodes(nodeSetId)
        nodeSetNodes[nodeSetId] = nodeIds[:]

    # In this particular case, we want to plot force-displacement curves
    # where the forces and displacements are on a specific node set

    nodeset_displacement = []
    nodeset_force = []

    for timeStep in range(numTimeSteps):
        
        displacement_x = inFile.get_node_variable_values('displacement_x', timeStep+1)
        force_x = inFile.get_node_variable_values('force_x', timeStep+1)

        # The x-max face is nodeset_2
        nodeSet = nodeSetNodes[2]
        displacement = 0.0
        force = 0.0
        for nodeId in nodeSet:
            displacement += displacement_x[nodeId-1]
            force += force_x[nodeId-1]
        displacement /= len(nodeSet)
        nodeset_displacement.append(displacement)
        nodeset_force.append(force)

    inFile.close()

    print

    outFileName = outFileLabel + 'force_displacement.txt'
    dataFile = open(outFileName, 'w')
    for timeStep in range(numTimeSteps):
        # NOTE:  The displacement is doubled here because the BC on the model actually displace both ends in opposite directions
        dataFile.write(str(2.0*nodeset_displacement[timeStep]) + "  " + str(nodeset_force[timeStep]) + "\n")
    dataFile.close()
    print "Force-displacement data for written to", outFileName

    print
    
