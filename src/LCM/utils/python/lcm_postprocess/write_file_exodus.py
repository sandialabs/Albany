#!/usr/bin/python

import contextlib
import cStringIO
from exodus import exodus
import os
import sys


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()
    yield
    sys.stdout = save_stdout


# Write postprocessed per-element data to an Exodus file
def write_file_exodus(domain = None, name_file_input = None, name_file_output = None):

    times = domain.times
    num_dims = domain.num_dims

    if os.path.isfile(name_file_output):
        cmd_line = "rm %s" % name_file_output
        os.system(cmd_line)

    with nostdout():
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
    file_output.set_element_variable_number(3 * num_dims**2 + 3)

    for dim_i in range(num_dims):

        for dim_j in range(num_dims):

            name_stress = 'Cauchy_Stress_' + str(dim_i + 1) + str(dim_j + 1)

            file_output.put_element_variable_name(
                name_stress, 
                dim_i * num_dims + dim_j + 1)

            name_def_grad = 'F_' + str(dim_i + 1) + str(dim_j + 1)

            file_output.put_element_variable_name(
                name_def_grad, 
                num_dims**2 + dim_i * num_dims + dim_j + 1)

            name_strain = 'Log_Strain_' + str(dim_i + 1) + str(dim_j + 1)

            file_output.put_element_variable_name(
                name_strain, 
                2 * num_dims**2 + dim_i * num_dims + dim_j + 1)

            for key_block in domain.blocks:

                block = domain.blocks[key_block]

                for step in range(len(times)):

                    file_output.put_element_variable_values(
                        key_block,
                        name_stress,
                        step + 1,
                        [block.elements[key_element].variables['Cauchy_Stress'][times[step]][dim_i][dim_j] for key_element in block.elements])

                    file_output.put_element_variable_values(
                        key_block,
                        name_def_grad,
                        step + 1,
                        [block.elements[key_element].variables['F'][times[step]][dim_i][dim_j] for key_element in block.elements])

                    file_output.put_element_variable_values(
                        key_block,
                        name_strain,
                        step + 1,
                        [block.elements[key_element].variables['Log_Strain'][times[step]][dim_i][dim_j] for key_element in block.elements])

    file_output.put_element_variable_name(
        'Mises_Stress', 
        3 * num_dims**2 + 1)

    for key_block in domain.blocks:

        block = domain.blocks[key_block]

        for step in range(len(times)):

            file_output.put_element_variable_values(
                key_block,
                'Mises_Stress',
                step + 1,
                [block.elements[key_element].variables['Mises_Stress'][times[step]] for key_element in block.elements])

    file_output.put_element_variable_name(
        'eqps', 
        3 * num_dims**2 + 2)

    for key_block in domain.blocks:

        block = domain.blocks[key_block]

        for step in range(len(times)):

            file_output.put_element_variable_values(
                key_block,
                'eqps',
                step + 1,
                [block.elements[key_element].variables['eqps'][times[step]] for key_element in block.elements])

    file_output.put_element_variable_name(
        'Misorientation', 
        3 * num_dims**2 + 3)

    for key_block in domain.blocks:

        block = domain.blocks[key_block]

        for step in range(len(times)):

            file_output.put_element_variable_values(
                key_block,
                'Misorientation',
                step + 1,
                [block.elements[key_element].variables['Misorientation'][times[step]] for key_element in block.elements])

            
    with nostdout():
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
