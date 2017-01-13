#!/usr/bin/python
'''
read_file_input_material.py
'''

import xml.etree.ElementTree as et
from _core import ObjDomain
# import lcm_postprocess
import numpy as np

# Read the materials input file
def read_file_input_material(
    filename = None,
    domain = ObjDomain(),
    names_variable = ['orientations','num_slip_systems']):
    
    tree = et.parse(filename)
    root = tree.getroot()

    lookup_id_name = dict()

    for key_block in domain.blocks:

        lookup_id_name[domain.blocks[key_block].name] = key_block

    lookup_id_material = dict()

    for child in root:

        if child.attrib['name'] == 'ElementBlocks':

            names_material = dict()

            for block in child:

                name_block = block.attrib['name']

                if name_block in lookup_id_name:

                    block_id = lookup_id_name[name_block]

                    for parameter in block:

                        if parameter.attrib['name'] == 'material':

                            domain.blocks[block_id].material.name = parameter.attrib['value']

                            lookup_id_material[domain.blocks[block_id].material.name] = block_id

    for name in names_variable:

        if name == 'orientations':

            for child in root:

                if child.attrib['name'] == 'Materials':

                    for material in child:

                        if material.attrib['name'] in lookup_id_material:

                            block_id = lookup_id_material[material.attrib['name']]

                            for plist in material:

                                if plist.attrib['name'] == 'Crystal Elasticity':

                                    orientation = np.zeros((3,3))

                                    for parameter in plist:

                                        name_paramter = parameter.attrib['name']

                                        if name_paramter.split()[0] == 'Basis':

                                            orientation[int(name_paramter.split()[2])-1][:] = \
                                                np.fromstring(
                                                    parameter.attrib['value'].translate(None,'{}'),
                                                    sep = ',')
                            try:
                                domain.blocks[block_id].material.orientation = orientation
                            except:
                                domain.blocks[block_id].material.orientation = np.eye(domain.num_dims)

        if name == 'num_slip_systems':

            for child in root:

                if child.attrib['name'] == 'Materials':

                    for material in child:

                        if material.attrib['name'] in lookup_id_material:

                            block_id = lookup_id_material[material.attrib['name']]

                            for parameter in material:

                                if parameter.attrib['name'] == 'Number of Slip Systems':

                                    domain.blocks[block_id].material.num_slip_systems = int(parameter.attrib['value'])

# end def read_file_input_material(name_file_input):



if __name__ == '__main__':

    import re
    import sys
    import cPickle as pickle

    try:
        name_file_input = sys.argv[1]
    except:
        raise

    match = re.search('(.+)_Material.xml', name_file_input)
    if type(match) != type(None):
        name_file_base = match.group(1)

    try:
        file_pickling = open(name_file_base + '_Domain.pickle', 'rb')
        domain = pickle.load(file_pickling)
        file_pickling.close()
    except:
        raise

    if len(sys.argv) == 2:

        read_file_input_material(filename = name_file_input, domain = domain)

    elif len(sys.argv) > 2:

        read_file_input_material(filename = name_file_input, domain = domain, names_variable = sys.argv[2:])

    file_pickling = open(name_file_base + '_Domain.pickle', 'wb')
    pickle.dump(domain, file_pickling, pickle.HIGHEST_PROTOCOL)
    file_pickling.close()

# end if __name__ == '__main__':