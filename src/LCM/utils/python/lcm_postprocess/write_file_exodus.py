#!/usr/bin/python

import contextlib
import cStringIO
from exodus import exodus
import os
import sys
from _core import stdout_redirected

# Write postprocessed per-element data to an Exodus file
def write_file_exodus(domain = None, name_file_input = None, name_file_output = None):

    times = domain.times
    num_dims = domain.num_dims

    if os.path.isfile(name_file_output):
        cmd_line = "rm %s" % name_file_output
        os.system(cmd_line)

    with stdout_redirected():
        file_input = exodus(name_file_input)
        file_output = file_input.copy(name_file_output)

    # write times to file_output
    for step in range(len(times)):
       file_output.put_time(step + 1, times[step])

    #
    # write out displacement vector
    #
    file_output.set_node_variable_number(file_input.get_node_variable_number())

    count = 0
    for key, value in domain.names_variable_node.items():
        
        for suffix in value:

            count += 1
            name_variable = key + '_' + suffix
            file_output.put_node_variable_name(name_variable, count)

            for step in range(len(times)):

                file_output.put_node_variable_values(
                    name_variable,
                    step + 1,
                    file_input.get_node_variable_values(name_variable, step + 1))

    #
    # create variables in output file
    #

    names_var_output_tensor = ['Cauchy_Stress', 'F', 'Log_Strain']
    names_var_output_scalar = ['Mises_Stress', 'eqps', 'Misorientation', 'Strain_Energy']

    names_var_avail_tensor = [name for name in names_var_output_tensor if name in domain.variables]
    names_var_avail_scalar = [name for name in names_var_output_scalar if name in domain.variables]

    num_vars_tensor = len(names_var_avail_tensor)
    num_vars_scalar = len(names_var_avail_scalar)

    file_output.set_element_variable_number(num_dims**2 * len(names_var_avail_tensor) + len(names_var_avail_scalar))

    for index, name in enumerate(names_var_avail_tensor):

        for dim_i in range(num_dims):

            for dim_j in range(num_dims):

                name_indexed = name + '_' + str(dim_i + 1) + str(dim_j + 1)

                file_output.put_element_variable_name(
                    name_indexed, 
                    index * num_dims**2 + dim_i * num_dims + dim_j + 1)

                for key_block in domain.blocks:

                    block = domain.blocks[key_block]

                    for step in range(len(times)):

                        file_output.put_element_variable_values(
                            key_block,
                            name_indexed,
                            step + 1,
                            [block.elements[key_element].variables[name][times[step]][dim_i][dim_j] for key_element in block.elements])

    for index, name in enumerate(names_var_avail_scalar):

        file_output.put_element_variable_name(
            name, 
            num_vars_tensor + index + 2)

        for key_block in domain.blocks:

            block = domain.blocks[key_block]

            for step in range(len(times)):

                file_output.put_element_variable_values(
                    key_block,
                    name,
                    step + 1,
                    [block.elements[key_element].variables[name][times[step]] for key_element in block.elements])

    with stdout_redirected():
        file_output.close()

# end def write_exodus_file(name_file_output):



if __name__ == '__main__':

    import sys
    import cPickle as pickle

    try:
        name_file_input = sys.argv[1]
    except:
        raise

    name_file_base = name_file_input.split('.')[0]
    name_file_extension = name_file_input.split('.')[-1]

    try:
        file_pickling = open(name_file_base + '_Domain.pickle', 'rb')
        domain = pickle.load(file_pickling)
        file_pickling.close()
    except:
        raise

    if len(sys.argv) == 2:

        write_file_exodus(
            domain = domain,
            name_file_input = name_file_input,
            name_file_output = name_file_base + '_Postprocess.' + name_file_extension)

    elif len(sys.argv) == 3:

        write_file_exodus(
            domain = domain,
            name_file_input = name_file_input,
            name_file_output = sys.argv[2])

# end if __name__ == '__main__':
