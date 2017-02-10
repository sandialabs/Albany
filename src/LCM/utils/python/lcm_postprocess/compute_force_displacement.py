#!/usr/bin/env python

import sys
import lcm_postprocess
from .lcm_exodus import open_file_exodus
from .lcm_exodus import close_file_exodus
import matplotlib.pyplot as plt
from ._core import InputError


def compute_force_displacement(
    name_file_input,
    reaction_node_set,
    direction,
    verbosity = 0,
    plotting = False,
    write_file = False):

    '''
    This script outputs both the load and displacement from a node nset

    Command: python -m lcm_postprocess.compute_force_displacement <exodus_file_name> <nodeset of interest> <direction>

    Because the displacements and forces are written with _x, _y, and _z, the direction
    is specified accordingly.
    '''
    file_input = open_file_exodus(name_file_input, verbosity = verbosity)

    # Extract nodal displacements and forces
    numNodes = file_input.num_nodes()
    numTimeSteps = file_input.num_times()
    nodeVariableNames = file_input.get_node_variable_names()

    displacement_label = 'displacement_' + direction
    if displacement_label not in nodeVariableNames:
        displacement_label = 'disp_' + direction
    if displacement_label not in nodeVariableNames:
        print "\nERROR:  Failed to extract " + displacement_label + " data\n"
        print "Nodal variable names are:"
        print "  " + "\n  ".join(nodeVariableNames)
        sys.exit(1)
    force_label = 'force_' + direction
    if force_label not in nodeVariableNames:
        force_label = 'resid_' + direction
        if force_label not in nodeVariableNames:
            print "\nERROR:  Failed to extract " + force_label + " data\n"
            print "Nodal variable names are:"
            print "  " + "\n  ".join(nodeVariableNames)
            sys.exit(1)

    # Read node sets
    nodeSetIds = file_input.get_node_set_ids()
    nodeSetNodes = {}
    for nodeSetId in nodeSetIds:
        nodeIds = file_input.get_node_set_nodes(nodeSetId)
        nodeSetNodes[nodeSetId] = nodeIds[:]

    # In this particular case, we want to plot force-displacement curves
    # where the forces and displacements are on a specific node set

    nodeset_displacement = []
    nodeset_force = []

    for timeStep in range(numTimeSteps):
        
        displacement_comp = file_input.get_node_variable_values(displacement_label, timeStep+1)
        force_comp = file_input.get_node_variable_values(force_label, timeStep+1)

        nodeSet = nodeSetNodes[reaction_node_set]
        displacement = 0.0
        force = 0.0
        for nodeId in nodeSet:
            displacement += displacement_comp[nodeId-1]
            force += force_comp[nodeId-1]
        displacement /= len(nodeSet)
        nodeset_displacement.append(displacement)
        nodeset_force.append(force)

    close_file_exodus(file_input)

    if write_file == True:

        outFileLabel = name_file_input.split('.')[0] + '_'
        outFileName = outFileLabel + 'Force_Displacement_' + str(reaction_node_set) + '_' + direction + '.dat'
        dataFile = open(outFileName, 'w')
        for timeStep in range(numTimeSteps):
            dataFile.write(str(nodeset_displacement[timeStep]) + "  " + str(nodeset_force[timeStep]) + "\n")
        dataFile.close()

    if verbosity > 0:
        print 'Force-displacement data for written to', outFileName, '\n'
    
    if plotting == True:
        
        fig, ax = plt.subplots()
        
        ax.plot(
            nodeset_displacement[:],
            nodeset_force[:],
            color = 'blue',
            marker = 'o',
            label = '1 elem/block')

        plt.xlabel('displacement (mm)')
        plt.ylabel('force (N)')
        lg = plt.legend(loc = 4)
        lg.draw_frame(False)
        plt.tight_layout()
        fig.savefig(outFileLabel + 'Force_Displacement' + '_' + str(reaction_node_set) + '_' + direction + '.pdf')
    


if __name__ == "__main__":

    if len(sys.argv) != 4:

        message = 'Required inputs: <exodus_file_name> <reaction_node_set> <direction>'

        sys.tracebacklimit = 0

        raise InputError(message)

    else:

        name_file_input = sys.argv[1]
        reaction_node_set = int(sys.argv[2])
        direction = str.lower(sys.argv[3])

        compute_force_displacement(
            name_file_input,
            reaction_node_set,
            direction,
            verbosity = 1,
            plotting = True,
            write_file = True)
