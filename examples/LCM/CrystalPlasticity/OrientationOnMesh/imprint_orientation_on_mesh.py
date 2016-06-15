#!/usr/bin/env python

# This script requires exodus.py

import os
import sys
import time
import string
import math
sys.path.append('/ascldap/users/djlittl/TPL/trilinos/votd/GCC_4.9.2_OPT/lib')
import exodus

def ListToMatrix(rotation):

    R = [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]]

    R[0][0] = rotation[0]
    R[0][1] = rotation[1]
    R[0][2] = rotation[2]
    R[1][0] = rotation[3]
    R[1][1] = rotation[4]
    R[1][2] = rotation[5]
    R[2][0] = rotation[6]
    R[2][1] = rotation[7]
    R[2][2] = rotation[8]

    return R

def EulerAngles(R):

    transpose_R = False

    if not transpose_R:
        R_11 = R[0][0]
        R_12 = R[0][1]
        R_13 = R[0][2]
        R_23 = R[1][2]
        R_31 = R[2][0]
        R_32 = R[2][1]
        R_33 = R[2][2]
    else:
        R_11 = R[0][0]
        R_12 = R[1][0]
        R_13 = R[2][0]
        R_23 = R[2][1]
        R_31 = R[0][2]
        R_32 = R[1][2]
        R_33 = R[2][2]

    Phi = math.acos(R_33)

    sine_val = math.sin(Phi)

    if sine_val > 1e-10 or sine_val < -1e-10:
        phi1 = math.atan2(R_31/sine_val, -1.0*R_32/sine_val)
        phi2 = math.atan2(R_13/sine_val, R_23/sine_val)

    else:

        # case for sin(Phi) == 0
        # if sine_val == 0, phi2 = 0, and ph1 = atan2(R21,R11)

        phi1 = math.atan2(R_12, R_11)    
        phi2 = 0.0

    euler = [phi1, Phi, phi2]

    return euler

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print "\nUsage:  imprint_orientation_on_mesh.py <mesh.g> <orientations.txt>\n"
        sys.exit(1)

    genesis_input_name = sys.argv[1]
    genesis_input = exodus.exodus(genesis_input_name, mode='r')

    # Read the orientation data

    orientations_input_name = sys.argv[2]
    orientations_file = open(orientations_input_name, 'r')
    orientations_lines = orientations_file.readlines()
    orientations_file.close()
    orientations = []
    for line in orientations_lines:
        vals = string.splitfields(line)
        if len(vals) == 9 and vals[0][0] != '#':
            orientation = []
            for i in range(9):
                orientation.append(float(vals[i]))
            orientations.append(orientation)


    # Print database parameters from genesis_input

    print " "
    print "Database version:         " + str(round(genesis_input.version.value,2))
    print "Database title:           " + genesis_input.title()
    print "Database dimensions:      " + str(genesis_input.num_dimensions())
    print "Number of nodes:          " + str(genesis_input.num_nodes())
    print "Number of elements:       " + str(genesis_input.num_elems())
    print "Number of element blocks: " + str(genesis_input.num_blks())
    print "Number of node sets:      " + str(genesis_input.num_node_sets())
    print "Number of side sets:      " + str(genesis_input.num_side_sets())
    print " "

    # Read the input genesis file

    qa_records = genesis_input.get_qa_records()

    coord_names = genesis_input.get_coord_names()
    x_coord, y_coord, z_coord = genesis_input.get_coords()

    node_id_map = genesis_input.get_node_id_map()
    elem_id_map = genesis_input.get_elem_id_map()

    mesh_has_attributes = False

    block_ids = genesis_input.get_elem_blk_ids()
    block_elem_types = []
    block_num_elem = []
    block_num_nodes_per_elem = []
    block_num_attributes = []
    block_connectivity = {}
    block_attributes = {}
    for block_id in block_ids:
        block_elem_types.append(genesis_input.elem_type(block_id))
        elem_connectivity, num_elem, num_nodes_per_elem = genesis_input.get_elem_connectivity(block_id)
        block_num_elem.append(num_elem)
        block_num_nodes_per_elem.append(num_nodes_per_elem)
        block_num_attributes.append(0)
        block_connectivity[block_id] = elem_connectivity
        elem_attrs = []
        num_attributes = genesis_input.num_attr(block_id)
        if num_attributes > 0:
            mesh_has_attributes = True
            elem_attrs = genesis_input.get_elem_attr(block_id)
        block_attributes[block_id] = elem_attrs

    if mesh_has_attributes:
        print "\nError, input mesh has element attributes.  Meshes with existing attributes are not supported."
        sys.exit(1)

    node_set_ids = genesis_input.get_node_set_ids()

    node_set_params = {}
    for node_set_id in node_set_ids:
        node_set_params[node_set_id] = genesis_input.get_node_set_params(node_set_id)

    node_set_nodes = {}
    for node_set_id in node_set_ids:
        node_ids = genesis_input.get_node_set_nodes(node_set_id)
        node_set_nodes[node_set_id] = []
        for i in range(len(node_ids)):
            node_set_nodes[node_set_id].append(node_ids[i])

    node_set_distribution_factors = {}
    for node_set_id in node_set_ids:
        distribution_factors = genesis_input.get_node_set_dist_facts(node_set_id)
        node_set_distribution_factors[node_set_id] = []
        for i in range(len(distribution_factors)):
            node_set_distribution_factors[node_set_id].append(distribution_factors[i])

    side_set_ids = genesis_input.get_side_set_ids()

    side_set_parameters = {}
    for side_set_id in side_set_ids:
        side_set_parameters[side_set_idd] = genesis_input.get_side_set_params(sideSetId)

    side_set_elements = {}
    side_set_sides = {}
    for side_set_id in side_set_ids:
        (element_ids, side_ids) = genesis_input.get_side_set(side_set_id)
        side_set_elements[side_set_id] = []
        side_set_sides[side_set_id] = []
        for i in range(len(element_ids)):
            side_set_elements[side_set_id].append(element_ids[i])
            side_set_sides[side_set_id].append(side_ids[i])

    side_set_distribution_factors = {}
    for side_set_id in side_set_ids:
        distribution_factors = genesis_input.get_side_set_dist_fact(side_set_id)
        side_set_distribution_factors[side_set_id] = []
        for i in range(len(distribution_factors)):
            side_set_distribution_factors[side_set_id].append(distribution_factors[i])

    # Create attributes for the orientations

    block_euler_angles = {}
    num_attributes = 3

    sorted_block_ids = []
    for block_id in block_ids:
        sorted_block_ids.append(block_id)
    sorted_block_ids.sort()

    block_euler_angles = {}
    for i in range(len(sorted_block_ids)):
        block_id = sorted_block_ids[i]
        block_euler_angles[block_id] = []
        rotation_matrix = ListToMatrix(orientations[i])
        euler_angles = EulerAngles(rotation_matrix)
        for j in range(num_attributes):
            block_euler_angles[block_id].append(euler_angles[j])

    for i in range(len(block_ids)):
        block_id = block_ids[i]
        block_num_attributes[i] = num_attributes
        for j in range(block_num_elem[i]):
            for k in range(num_attributes):
                block_attributes[block_id].append(block_euler_angles[block_id][k])

    # Write ExodusII file

    genesis_output_name = genesis_input_name[:-2] + "_orientation.g"
    if os.path.exists(genesis_output_name):
        os.remove(genesis_output_name)

    genesis_output = exodus.exodus(genesis_output_name,
                                   'w',
                                   'ctype',
                                   genesis_input.title(),
                                   genesis_input.num_dimensions(),
                                   genesis_input.num_nodes(),
                                   genesis_input.num_elems(),
                                   genesis_input.num_blks(),
                                   genesis_input.num_node_sets(),
                                   genesis_input.num_side_sets() )

    qa_software_name = "imprint_orientation_on_mesh.py"
    qa_software_descriptor = "1.0"
    qa_additional_software_data = time.strftime('%Y-%m-%d')
    qa_time_stamp = time.strftime('%H:%M:%S')
    qa_records.append( (qa_software_name, qa_software_descriptor, qa_additional_software_data, qa_time_stamp) )

    genesis_output.put_qa_records(qa_records)

    genesis_output.put_coord_names(coord_names)

    genesis_output.put_coords(x_coord, y_coord, z_coord)

    define_maps = 0
    if len(node_id_map) > 0 or len(elem_id_map) > 0:
        define_maps = 1

    genesis_output.put_concat_elem_blk(block_ids,
                                       block_elem_types,
                                       block_num_elem,
                                       block_num_nodes_per_elem,
                                       block_num_attributes,
                                       define_maps)

    if len(node_id_map) > 0:
        genesis_output.put_node_id_map(node_id_map)

    if len(elem_id_map) > 0:
        genesis_output.put_elem_id_map(elem_id_map)
        
    for block_id in block_ids:
        genesis_output.put_elem_connectivity(block_id, block_connectivity[block_id])
        if len(block_attributes[block_id]) != 0:
            genesis_output.put_elem_attr(block_id, block_attributes[block_id])

    for node_set_id in node_set_ids:
        genesis_output.put_node_set_params(node_set_id, len(node_set_nodes[node_set_id]), len(node_set_distribution_factors[node_set_id]))
        genesis_output.put_node_set(node_set_id, node_set_nodes[node_set_id])
        genesis_output.put_node_set_dist_fact(node_set_id, node_set_distribution_factors[node_set_id])

    for side_set_id in side_set_ids:
        genesis_output.put_side_set_params(side_set_id, len(side_set_elements[side_set_id]), len(side_set_distribution_factors[side_set_id]))
        genesis_output.put_side_set(side_set_id, side_set_elements[side_set_id], side_set_sides[side_set_id])
        genesis_output.put_side_set_dist_fact(side_set_id, side_set_distribution_factors[side_set_id])

    genesis_output.close()
