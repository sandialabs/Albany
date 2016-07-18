#!/usr/bin/env python

'''
This script output both the load and diplacement from a node nset

Command: PostProcess.py <exodus_file_name> <nodeset of interest> <direction>

Because the displacements and forces are written with _x, _y, and _z the direction
is specified accordingly.
'''

import sys
import exodus
import string
import numpy
import matplotlib.pyplot as plt

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print "\nUsage:  PostProcess.py <exodus_file_name> <reaction_node_set> <direction>\n"
        sys.exit(1)

    inFileName = sys.argv[1]
    reaction_node_set = int(sys.argv[2])
    direction = str.lower(sys.argv[3])
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
    displacement_label = 'displacement_' + direction
    if displacement_label not in nodeVariableNames:
        print "\nERROR:  Failed to extract " + displacement_label + " data\n"
        sys.exit(1)
    force_label = 'force_' + direction
    if force_label not in nodeVariableNames:
        print "\nERROR:  Failed to extract " + force_label + " data\n"
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
        
        displacement_comp = inFile.get_node_variable_values(displacement_label, timeStep+1)
        force_comp = inFile.get_node_variable_values(force_label, timeStep+1)

        nodeSet = nodeSetNodes[reaction_node_set]
        displacement = 0.0
        force = 0.0
        for nodeId in nodeSet:
            displacement += displacement_comp[nodeId-1]
            force += force_comp[nodeId-1]
        displacement /= len(nodeSet)
        nodeset_displacement.append(displacement)
        nodeset_force.append(force)

    inFile.close()

    outFileName = outFileLabel + 'force_displacement_ns_' + str(reaction_node_set) + '_' + direction + '.dat'
    dataFile = open(outFileName, 'w')
    for timeStep in range(numTimeSteps):
        dataFile.write(str(nodeset_displacement[timeStep]) + "  " + str(nodeset_force[timeStep]) + "\n")
    dataFile.close()
    print "Force-displacement data for written to", outFileName
    
    '''
    fig, ax = plt.subplots()
    ax.plot(nodeset_displacement[:],nodeset_force[:],color='blue',marker='o',label='1 elem/block')
    plt.xlabel('displacement (mm)')
    plt.ylabel('force (N)')
    lg = plt.legend(loc = 4)
    lg.draw_frame(False)
    plt.tight_layout()
    plt.show()
    fig.savefig('load_displacment.pdf')
    '''
