#!/usr/bin/python
'''
read_file_output_exodus.py
'''

# from lcm_postprocess._core import stdout_redirected
from exodus import exodus
import lcm_postprocess
import numpy as np
import re

from lcm_postprocess.lcm_exodus import get_names_variable
from lcm_postprocess.lcm_exodus import get_num_elements_block
from lcm_postprocess.lcm_exodus import get_element_variable_values

#
# Read the Exodus output file
#
# @profile
def read_file_output_exodus(
    filename = None,
    names_variable_read = [
        'Cauchy_Stress',
        'F'],
    build_domain = True):

    with lcm_postprocess.stdout_redirected():
        file_input = exodus(filename,'r')

    # get number of dimensions
    num_dims = file_input.num_dimensions()

    # Get number of elements
    num_elements = dict([(id,file_input.num_elems_in_blk(id)) for id in file_input.get_elem_blk_ids()])
    element_ids = file_input.get_elem_id_map()
    lookup_index_id = dict([(element_ids[index],index) for index in range(len(element_ids))])

    # Get number of nodes
    num_nodes = file_input.num_nodes()
    node_ids = file_input.get_node_id_map()

    # Get output times
    times = file_input.get_times()

    # Get list of nodal variables
    node_var_names = file_input.get_node_variable_names()

    # Get the unique nodal variable names
    names_variable_node = dict()

    for name_node_variable in file_input.get_node_variable_names():

        match = re.search('(.+)_([x-z]+)', name_node_variable)
        
        if match != None:

            name_base = match.group(1)

            name_index = match.group(2)
        
            if name_base not in names_variable_node:

                names_variable_node[name_base] = [name_index]

            else:

                names_variable_node[name_base].append(name_index)

    # Get the unique variable names (deal with integration points and arrays)
    names_variable_element = dict()

    for name_element_variable in file_input.get_element_variable_names():

        match = re.search('(.+)_([0-9]+)', name_element_variable)
        
        if match != None:

            name_base = match.group(1)

            name_index = match.group(2)
        
        if name_base not in names_variable_element:

            names_variable_element[name_base] = [name_index]

        else:

            names_variable_element[name_base].append(name_index)

    # Get number of element blocks and block ids
    block_ids = file_input.get_elem_blk_ids()

    num_blocks = file_input.num_blks()

    # Calculate number of integration points
    num_points = len(names_variable_element['Weights'])

    # Check that "Weights" exist as an element variable
    if (num_points == 0):
      raise Exception("The weights field is not available...try again.")

    #
    # Create and populate the domain object
    #
    if build_domain is True:
        domain = lcm_postprocess.ObjDomain(
            num_dims = num_dims,
            num_elements = np.sum(num_elements.values()),
            num_nodes = num_nodes,
            times = [x for x in times],
            names_variable_node = names_variable_node,
            names_variable_element = names_variable_element)

    coords = file_input.get_coords()

    for index_node, node_id in enumerate(node_ids):

        domain.nodes[node_id] = lcm_postprocess.ObjNode(
            coords = np.array([coords[x][index_node] for x in range(num_dims)]))

        node = domain.nodes[node_id]

        for name_variable_node in names_variable_node:

            indices_variable = names_variable_node[name_variable_node]

            if len(indices_variable) == num_dims: #TODO: write code to store nodal scalars, etc.

                node.variables[name_variable_node] = \
                    dict([(step, np.zeros(num_dims)) for step in times])

    for name_variable_node in names_variable_node:

        for step in range(len(times)):

            time = times[step]

            for dim_i in range(num_dims):

                name_variable = name_variable_node + '_' + names_variable_node[name_variable_node][dim_i]

                values = file_input.get_node_variable_values(
                    name_variable,
                    step + 1)

                for index_node, key_node in enumerate(domain.nodes):

                    node = domain.nodes[key_node]

                    try:
                        node.variables[name_variable_node][time][dim_i] = values[index_node]
                    except:
                        print key_node

#-------------------------------------------------------------------------------
# Store the element variable values
#-------------------------------------------------------------------------------

    element_start_id = 0

    names_variable_exodus = get_names_variable(file_input)

    for block_id in block_ids:

        num_elements_block = get_num_elements_block(file_input, block_id)

        domain.blocks[block_id] = lcm_postprocess.ObjBlock(
            num_elements = num_elements[block_id],
            num_points = num_points,
            num_nodes_per_elem = file_input.num_nodes_per_elem(block_id),
            name = file_input.get_elem_blk_name(block_id))
        # TODO: change line above to line below
        # num_points =  = num_points[block_id]

        block = domain.blocks[block_id]

        block.map_element_ids = dict()

        for index_element in range(num_elements[block_id]):

            # Get the global element number
            key_element = element_ids[index_element + element_start_id]

            # Map global element id to blockwise index
            block.map_element_ids[key_element] = index_element

            block.elements[key_element] = lcm_postprocess.ObjElement()

            element = block.elements[key_element]

            connectivity, num_elements_block, num_nodes_element = file_input.get_elem_connectivity(block_id)

            connectivity_array = np.reshape([x - 1 for x in connectivity], (num_elements_block, num_nodes_element))

            for index_node in range(block.num_nodes_per_elem):

                node_id = node_ids[connectivity_array[index_element, index_node]]

                element.nodes[node_id] = domain.nodes[node_id]

            for index_point in range(num_points):

                element.points[index_point] = lcm_postprocess.ObjPoint()

        for index_point in range(block.num_points):

            key_weight = 'Weights_' + str(index_point + 1)

            index_weight = names_variable_exodus.index(key_weight) + 1

            values_weights = get_element_variable_values(
                file_input, block_id, num_elements_block, index_weight, 1)

            for key_element in block.elements:

                block.elements[key_element].points[index_point].weight = \
                    values_weights[block.map_element_ids[key_element]]

        for key_element in block.elements:

            element = block.elements[key_element]

            element.volume = np.sum([element.points[x].weight for x in element.points])

        block.volume = np.sum([block.elements[x].volume for x in block.elements])

        element_start_id += block.num_elements

    domain.volume = np.sum([domain.blocks[x].volume for x in domain.blocks])

    #
    # Set the values of the variables in the domain object
    #
    for key_variable in names_variable_element:

        # print 'Reading variable:'
        # print '  ', key_variable

        indices_variable = names_variable_element[key_variable]

        if len(indices_variable) == num_points * num_dims**2:

            _set_values_tensor(
                file_input,
                key_variable,
                indices_variable,
                domain)

        # elif len(indices_variable) == num_slip_systems:

        #     _set_values_vector(
        #         file_input,
        #         key_variable,
        #         indices_variable,
        #         domain)

        elif len(indices_variable) == num_points:

            _set_values_scalar(
                file_input,
                key_variable,
                indices_variable,
                domain)

    with lcm_postprocess.stdout_redirected():
        file_input.close()

    return domain

# end def read_file_output_exodus(file_input, **kwargs):




# Create tensor-valued field in the domain object
# @profile
def _set_values_tensor(file_input, name_variable, indices_variable, domain):

    num_dims = domain.num_dims

    times = file_input.get_times()

    domain.variables[name_variable] = \
        dict([(step, np.zeros((num_dims, num_dims))) for step in times])

    names_variable_exodus = get_names_variable(file_input)

    # Note: exodus function get_element_variable_values returns values by block,    
    # so outer loop over block

    for key_block, block in domain.blocks.items():

        num_elements_block = get_num_elements_block(file_input, key_block)

        # block = domain.blocks[key_block]

        map_element_ids = block.map_element_ids

        variables_block = block.variables

        keys_variable = []

        for dim_i in range(num_dims):

            for dim_j in range(num_dims):

                index_dim = num_dims * dim_i + dim_j

                for index_point in range(block.num_points):

                    index_variable = index_point * num_dims**2 + index_dim

                    keys_variable.append(name_variable + '_' + indices_variable[index_variable])

        block.variables[name_variable] = \
            dict([(step, np.zeros((num_dims, num_dims))) for step in times])

        for element in block.elements.values():

            # element = block.elements[key_element]

            element.variables[name_variable] = \
                dict([(step, np.zeros((num_dims, num_dims))) for step in times])

            for point in element.points.values():

                # point = element.points[key_point]

                point.variables[name_variable] = \
                    dict([(step, np.zeros((num_dims, num_dims))) for step in times])
                
        for step in range(len(times)):

            time = times[step]

            for dim_i in range(num_dims):

                for dim_j in range(num_dims):

                    index_dim = num_dims * dim_i + dim_j

                    for index_point in range(block.num_points):

                        name_variable_exodus = keys_variable[index_dim * block.num_points + index_point]

                        index_variable = names_variable_exodus.index(name_variable_exodus) + 1

                        # values_block = file_input.get_element_variable_values(
                        #     key_block,
                        #     name_variable_exodus,
                        #     step + 1)

                        values_block = get_element_variable_values(
                            file_input,
                            key_block,
                            num_elements_block,
                            index_variable,
                            step + 1)

                        for key_element, element in block.elements.items():

                            # element = block.elements[key_element]

                            # point = element.points[index_point]

                            element.points[index_point].variables[name_variable][time][dim_i, dim_j] = \
                                values_block[map_element_ids[key_element]]

                    for element in block.elements.values():

                        # element = block.elements[key_element]

                        for point in element.points.values():

                            # point = element.points[key_point]

                            element.variables[name_variable][time][dim_i, dim_j] += \
                                point.variables[name_variable][time][dim_i, dim_j] * \
                                point.weight

                        element.variables[name_variable][time][dim_i, dim_j] /= element.volume

                        block.variables[name_variable][time][dim_i, dim_j] += \
                            element.variables[name_variable][time][dim_i, dim_j] * element.volume / block.volume

                    domain.variables[name_variable][time][dim_i, dim_j] += \
                        block.variables[name_variable][time][dim_i, dim_j] * \
                        block.volume / domain.volume

# end def _set_values_tensor(file_input, name_variable, domain):





# Create scalar-valued field in the domain object
def _set_values_scalar(file_input, name_variable, indices_variable, domain):

    times = file_input.get_times()

    domain.variables[name_variable] = dict([(step, 0.0) for step in times])

    names_variable_exodus = get_names_variable(file_input)

    # Note: exodus function get_element_variable_values returns values by block,    
    # so outer loop over block

    for key_block in domain.blocks:

        num_elements_block = get_num_elements_block(file_input, key_block)

        block = domain.blocks[key_block]

        keys_variable = []

        for index_point in range(block.num_points):

            keys_variable.append(name_variable + '_' + indices_variable[index_point])

        block.variables[name_variable] = dict([(step, 0.0) for step in times])

        for key_element in block.elements:

            element = block.elements[key_element]

            element.variables[name_variable] = dict([(step, 0.0) for step in times])

            for key_point in element.points:

                point = element.points[key_point]

                point.variables[name_variable] = dict([(step, 0.0) for step in times])
                
        for step in range(len(times)):

            time = times[step]

            for index_point in range(block.num_points):

                # values_block = file_input.get_element_variable_values(
                #     key_block, 
                #     key_variable, 
                #     step + 1)

                name_variable_exodus = keys_variable[index_point]

                index_variable = names_variable_exodus.index(name_variable_exodus) + 1

                values_block = get_element_variable_values(
                    file_input, 
                    key_block, 
                    num_elements_block,
                    index_variable,
                    step + 1)

                for key_element in block.elements:

                    element = block.elements[key_element]

                    point = element.points[index_point]

                    point.variables[name_variable][time] = values_block[block.map_element_ids[key_element]]

            for key_element in block.elements:

                element = block.elements[key_element]

                for key_point in element.points:

                    point = element.points[key_point]

                    element.variables[name_variable][time] += \
                        point.variables[name_variable][time] * point.weight / element.volume

                block.variables[name_variable][time] += element.variables[name_variable][time] * \
                    element.volume / block.volume

            domain.variables[name_variable][time] += block.variables[name_variable][time] * \
                block.volume / domain.volume

# end def _set_values_scalar(file_input, name_variable, domain):



if __name__ == '__main__':

    import sys
    import cPickle as pickle

    try:
        name_file_input = sys.argv[1]
    except:
        raise

    name_file_base = name_file_input.split('.')[0]
    name_file_extension = name_file_input.split('.')[-1]

    if len(sys.argv) == 2:

        domain = read_file_output_exodus(filename = sys.argv[1])

    elif len(sys.argv) > 2:

        domain = read_file_output_exodus(filename = sys.argv[1], names_variable = sys.argv[2:])

    file_pickling = open(name_file_base + '_Domain.pickle', 'wb')
    pickle.dump(domain, file_pickling, pickle.HIGHEST_PROTOCOL)
    file_pickling.close()

# end if __name__ == '__main__':
