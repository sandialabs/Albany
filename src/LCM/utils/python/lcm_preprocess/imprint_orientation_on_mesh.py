#!/usr/bin/env python

# This script requires exodus.py

import argparse
import exodus
from lcm_postprocess._core import InputError
import math
import os
import string
import sys
import time

def compute_euler_angles(matrices_rotation=None):
    phis = [np.arccos(m[2,2]) for m in matrices_rotation]
    sines = np.sin(phis)
    phis1 = [np.arctan2(m[2,0]/sine_val, -m[2,1]/sine_val) for m in matrices_rotation]
    phis2 = [np.arctan2(m[0,2]/sine_val, m[1,2]/sine_val) for m in matrices_rotation]
    # case for sin(Phi) == 0: if sine_val == 0, phi2 = 0, and ph1 = atan2(R21,R11)
    phis1[np.abs(sines) < 1e-10] = math.atan2(R_12, R_11)    
    phis2[np.abs(sines) < 1e-10] = 0.0
    return phis1, phis, phis2

def print_info(genesis_input=None):
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

def initialize_genesis(genesis_input=None, genesis_output_name=None):
    if os.path.exists(genesis_output_name):
        os.remove(genesis_output_name)
    num_blocks_input = genesis_input.num_blks()
    if args.combine_blocks == True:
        num_blocks_output = 1
    else:
        num_blocks_output = num_blocks_input
    genesis_output = exodus.exodus(
        genesis_output_name,
       'w',
       'numpy',
       genesis_input.title(),
       genesis_input.num_dimensions(),
       genesis_input.num_nodes(),
       genesis_input.num_elems(),
       num_blocks_output,
       genesis_input.num_node_sets(),
       genesis_input.num_side_sets())
    return genesis_output

def write_qa_records(genesis_input=None, genesis_output=None):
    qa_records = genesis_input.get_qa_records()
    sw_name = 'imprint_orientation_on_mesh.py'
    sw_descriptor = '1.0'
    sw_data = time.strftime('%Y-%m-%d')
    timestamp = time.strftime('%H:%M:%S')
    qa_records.append((sw_name, sw_descriptor, sw_data, timestamp))
    genesis_output.put_qa_records(qa_records)

def transfer_genesis(genesis_input=None, genesis_output_name=None):
    genesis_output = initialize_genesis(
        genesis_input=genesis_input, genesis_output_name=genesis_output_name)
    write_qa_records(genesis_input=genesis_input, genesis_output=genesis_output)

    return genesis_output

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--meshfile', help='Specify mesh filename', default=None)
    parser.add_argument('-c', '--combine_blocks', type=bool, help = 'Combine elements into single block', default=True)
    parser.add_argument('-r', '--rotfile', help = 'Specify rotations filename.', default=None)
    parser.add_argument('-o', '--orientation_format', help = 'Specify "matrix" or "angles" format', default='matrix')

    args = parser.parse_args()

    if args.meshfile is not None:
        if args.rotfile is None:
            basename = args.meshfile.splitext()[0]
            args.rotfile = basename + '_Rotations.txt'
    else:
        if args.rotfile is not None:
            basename = args.rotfile.replace('_Rotations.txt','')
            args.meshfile = basename + '.g'
        else:
            names_file = os.walk('.').next()[2]
            names_potential = [n for n in names_file if n.endswith('_Rotations.txt')]
            if len(names_potential) is 1:
                basename = names_potential[0].replace('_Rotations.txt','')
                args.rotfile = basename + '_Rotations.txt'
                args.meshfile = basename + '.g'
            else:
                raise InputError('Non-unique or missing assumed base file name')

    assert(os.path.isfile(args.meshfile))
    assert(os.path.isfile(args.rotfile))

    genesis_input = exodus.exodus(args.meshfile, mode='r', array_type='numpy')

    # Read the orientation data
    orientations = np.loadtxt(args.rotfile, ndmin=2)
    num_orientations = orientations.shape[0]
    num_dims = int(np.sqrt(orientations.shape[1]))
    assert(num_dims**2 == orientations.shape[1])

    # Print database parameters from genesis_input
    print_info(genesis_input=genesis_input)

    genesis_output_name = os.path.splitext(
        genesis_input_name)[0] + '_orientation.g'

    genesis_output = transfer_genesis(
        genesis_input=genesis_input, genesis_output_name=genesis_output_name)

    

    

    if args.orientation_format == 'matrix':
        num_orientation_attributes = num_dims**2
    elif args.orientation_format == 'angles':
        num_orientation_attributes = num_dims
    else:
        raise Exception('orientation_format must be defined')    

    genesis_output.put_coord_names(genesis_input.get_coord_names())
    genesis_output.put_coords(*genesis_input.get_coords())

    node_id_map = genesis_input.get_node_id_map()
    elem_id_map = genesis_input.get_elem_id_map()
    define_maps = int(len(node_id_map) + len(elem_id_map) > 0)

    if len(node_id_map) > 0:
        genesis_output.put_node_id_map(node_id_map)

    if len(elem_id_map) > 0:
        genesis_output.put_elem_id_map(elem_id_map)

    block_ids_input = genesis_input.get_elem_blk_ids()
    block_num_attributes = [num_orientation_attributes for b in block_ids_input]

    if args.combine_blocks is False:
        block_ids_output = block_ids_input
        block_elem_types = [genesis_input.elem_type(b) for b in block_ids_input]
        block_connectivity, block_num_elem, block_num_nodes_per_elem = zip(
            *[genesis_input.get_elem_connectivity(b) for b in block_ids_input])
    else:
        block_ids_output = [1]
        block_elem_types = [genesis_input.elem_type(block_ids_input[0])]
        block_connectivity, block_num_elem, block_num_nodes_per_elem = zip(
            *[genesis_input.get_elem_connectivity(block_ids_input[0])])
        for b in block_ids_input:
            assert(genesis_input.elem_type[b] == block_elem_types[0])
            bc, be, bn = genesis_input.get_elem_connectivity(b)
            assert(bc == block_connectivity[0])
            assert(bn == block_num_nodes_per_elem[0])




    genesis_output.put_concat_elem_blk(
        block_ids_output,
        block_elem_types,
        block_num_elem,
        block_num_nodes_per_elem,
        block_num_attributes,
        define_maps)
        
    for block_id in block_ids_output:
        genesis_output.put_elem_connectivity(block_id, block_connectivity[block_id])
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

    

    
    block_attributes = {}
    for block_id in block_ids:
        elem_attrs = []
        num_attributes = genesis_input.num_attr(block_id)
        if num_attributes > 0:
            mesh_has_attributes = True
            elem_attrs = genesis_input.get_elem_attr(block_id)
        block_attributes[block_id] = elem_attrs
    if any([genesis_input.num_attr(b) > 0 for b in block_ids_input]):
        raise Exception('\nMeshes with existing attributes are not supported.')

    node_set_ids = genesis_input.get_node_set_ids()
    if len(node_set_ids) > 0:
        node_set_params = {n:genesis_input.get_node_set_params(n) for n in node_set_ids}
        node_set_nodes = {n:genesis_input.get_node_set_nodes(n) for n in node_set_ids}
        node_set_distribution_factors = {n:genesis_input.get_node_set_dist_facts(n) for n in node_set_ids}

    side_set_ids = genesis_input.get_side_set_ids()
    if len(side_set_ids) > 0:
        side_set_parameters = {s:genesis_input.get_side_set_params(s) for s in side_set_ids}
        element_ids, side_ids = zip(*[genesis_input.get_side_set(s) for s in side_set_ids])
        side_set_elements = {i:e for i,e in zip(side_set_ids, element_ids)}
        side_set_sides = {i:s for i,s in zip(side_set_ids, side_ids)}
        side_set_distribution_factors = {s:genesis_input.get_side_set_dist_fact(s) for s in side_set_ids}

    # Create attributes for the orientations

    idx = np.argsort(block_ids_input)

    block_orientations = {}

    sorted_block_ids = []
    for block_id in block_ids:
        sorted_block_ids.append(block_id)
    sorted_block_ids.sort()

    block_orientations = {}
    for i, orientation in enumerate(orientations):
        block_id = sorted_block_ids[i]
        block_orientations[block_id] = []
        if args.orientation_format == 'angles':
            euler_angles = EulerAngles(orientation.reshape((num_dims,num_dims)))
            for j in range(num_orientation_attributes):
                block_orientations[block_id].append(euler_angles[j])
        else:
            # store the transpose of the rotation matrix on the mesh
            block_orientations.extend(
                orientation.reshape((3,3)).T.flatten().tolist())

    for i in range(len(block_ids)):
        block_id = block_ids[i]
        for j in range(block_num_elem[i]):
            for k in range(num_orientation_attributes):
                block_attributes[block_id].append(block_orientations[block_id][k])

    # Combine all blocks with an orientation assigned to them into a single block

    if combine_blocks:

        print "Combining all blocks with an orientation assigned to them into a single block ..."

        # Placeholder for functionality that allows only specific blocks to be combined
        blocks_to_combine = block_ids[:]

        sorted_blocks_to_combine = blocks_to_combine[:]
        sorted_blocks_to_combine.sort()
        combined_block_id = sorted_blocks_to_combine[0]

        grain_elem_types = {}
        grain_num_elem = {}
        grain_num_nodes_per_elem = {}
        grain_num_attributes = {}
        for i in range(len(block_ids)):
            block_id = block_ids[i]
            if block_id in blocks_to_combine:
                grain_elem_types[block_id] = block_elem_types[i]
                grain_num_elem[block_id] = block_num_elem[i]
                grain_num_nodes_per_elem[block_id] = block_num_nodes_per_elem[i]
                grain_num_attributes[block_id] = block_num_attributes[i]

        combined_block_elem_type = grain_elem_types[grain_elem_types.keys()[0]]
        combined_block_num_elem = 0
        combined_block_num_nodes_per_elem = grain_num_nodes_per_elem[grain_num_nodes_per_elem.keys()[0]]
        combined_block_num_attributes = grain_num_attributes[grain_num_attributes.keys()[0]]
        combined_block_connectivity = []
        combined_block_attributes = []

        for block_id in blocks_to_combine:
            combined_block_num_elem += grain_num_elem[block_id]
            combined_block_connectivity.extend(block_connectivity[block_id])
            combined_block_attributes.extend(block_attributes[block_id])

        initial_block_ids = block_ids[:]
        index = len(initial_block_ids) - 1
        for i in range(len(initial_block_ids)):
            block_id = initial_block_ids[index]
            if block_id in blocks_to_combine:
                if block_id != combined_block_id:
                    block_ids.pop(index)
                    block_elem_types.pop(index)
                    block_num_elem.pop(index)
                    block_num_nodes_per_elem.pop(index)
                    block_num_attributes.pop(index)
            index = index - 1

        for i in range(len(block_ids)):
            if block_ids[i] == combined_block_id:
                block_elem_types[i] = combined_block_elem_type
                block_num_elem[i] = combined_block_num_elem
                block_num_nodes_per_elem[i] = combined_block_num_nodes_per_elem
                block_num_attributes[i] = combined_block_num_attributes
                block_connectivity[combined_block_id] = combined_block_connectivity
                block_attributes[combined_block_id] = combined_block_attributes

                print "  combined_block_id", combined_block_id
                print "  num elem", block_num_elem[i]
                print "  num nodes per ele", block_num_nodes_per_elem[i]
                print "  num attr", block_num_attributes[i]
                print "  conn array len", len(block_connectivity[combined_block_id])
                print "  attr array len", len(block_attributes[combined_block_id])
        
    # Write ExodusII file

    

    
