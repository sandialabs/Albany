#!/usr/bin/python

import cPickle as pickle

# Write select data to text file
def write_file_data(domain = None, name_file_input = None, name_file_output = None, precision = 8):

    '''
    Writes simulation data to text file
    '''

    if name_file_input != None:
        domain = pickle.load(open(name_file_input, 'rb'))

    if name_file_output == None:
        name_file_output = 'data.out'

    file = open(name_file_output, 'w')

    str_format = '%.'+str(precision)+'e'

    file.write('Deformation Gradient\n')
    for step in sorted(domain.variables['F']):
        domain.variables['F'][step].tofile(file, sep = ' ', format = str_format)
        file.write('\n')

    file.write('Cauchy Stress\n')
    for step in sorted(domain.variables['Cauchy_Stress']):
        domain.variables['Cauchy_Stress'][step].tofile(file, sep = ' ', format = str_format)
        file.write('\n')

    file.close()

# end def write_file_data(...):


#
# If run as 'python -m lcm_postprocess.write_file_data <name_file_input>'
#
if __name__ == '__main__':
    
    import os
    import sys

    try:

        name_file_input = sys.argv[1]

    except IndexError:

        raise IndexError('Name of input file required')

    if os.path.exists(name_file_input) == False:

        raise IOError('File does not exist')

    write_file_data(name_file_input = name_file_input)