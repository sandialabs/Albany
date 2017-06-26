#!/usr/bin/python
'''
read_file_input_material.py
'''

import xml.etree.ElementTree as et
from _core import ObjDomain
# import lcm_postprocess
import numpy as np
import os

# Convert dict to etree for yaml support
def dict_to_etree(d):
    def _to_etree(d, root):
        # print d
        if not d:
            pass
        elif isinstance(d, basestring):
            root.text = d
        elif isinstance(d, dict):
            for k,v in d.items():
                assert isinstance(k, basestring)
                if k.startswith('#'):
                    assert k == '#text' and isinstance(v, basestring)
                    root.text = v
                elif k.startswith('@'):
                    assert isinstance(v, basestring)
                    root.set(k[1:], v)
                elif isinstance(v, list):
                    for e in v:
                        _to_etree(e, et.SubElement(root, k))
                else:
                    _to_etree(v, et.SubElement(root, k))
        else:
            root.text = str(d)
        # else: assert d == 'invalid type', (type(d), d)
    assert isinstance(d, dict) and len(d) == 1
    tag, body = next(iter(d.items()))
    node = et.Element(tag)
    _to_etree(body, node)
    return node

# Read the materials input file
def read_file_input_material(
    name_file = None,
    extension = None,
    domain = ObjDomain(),
    names_variable = ['orientations','num_slip_systems']):

    lookup_id_name = dict()

    for key_block in domain.blocks:

        lookup_id_name[domain.blocks[key_block].name] = key_block

    lookup_id_material = dict()

    names_material = dict()

    if extension == None:

        name_file, extension = os.path.splitext(name_file)

    if extension == '.yaml':

        try:
            import yaml
        except:
            raise Exception('PyYaml not installed')

        file = open(name_file + extension)
        dict_yaml = yaml.safe_load(file)['ANONYMOUS']
        # element_tree = dict_to_etree(dict_yaml)
        # tree = et.ElementTree(element = element_tree)

        for name_block, dict_block in dict_yaml['ElementBlocks'].items():

            if name_block in lookup_id_name:

                id_block = lookup_id_name[name_block]

                domain.blocks[id_block].material.name = dict_block['material']

                lookup_id_material[domain.blocks[id_block].material.name] = id_block

        for name in names_variable:

            if name == 'orientations':

                for name_material, plist in dict_yaml['Materials'].items():

                    if name_material in lookup_id_material:

                        id_block = lookup_id_material[name_material]

                        orientation = np.eye(domain.num_dims)

                        for name_parameter, value_parameter in plist['Crystal Elasticity'].items():

                            if name_parameter.split()[0] == 'Basis':

                                orientation[int(name_parameter.split()[2])-1][:] = \
                                    np.array(value_parameter)

                        domain.blocks[id_block].material.orientation = orientation

            if name == 'num_slip_systems':

                for name_material, plist in dict_yaml['Materials']:

                    if name_material in lookup_id_material:

                        id_block = lookup_id_material[name_material]

                        domain.blocks[id_block].material.num_slip_systems = int(plist['Number of Slip Systems'])

    elif extension == '.xml':
    
        tree = et.parse(name_file + extension)

        root = tree.getroot()

        for child in root:

            if child.attrib['name'] == 'ElementBlocks':

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

    name_file, extension = os.path.splitext(name_file_input)

    match = re.search('(.+)_Material', name_file)
    if type(match) != type(None):
        name_file_base = match.group(1)

    try:
        file_pickling = open(name_file_base + '_Domain.pickle', 'rb')
        domain = pickle.load(file_pickling)
        file_pickling.close()
    except:
        raise

    if len(sys.argv) == 2:

        read_file_input_material(name_file = name_file, extension = extension, domain = domain)

    elif len(sys.argv) > 2:

        read_file_input_material(name_file = name_file, extension = extension, domain = domain, names_variable = sys.argv[2:])

    file_pickling = open(name_file_base + '_Domain.pickle', 'wb')
    pickle.dump(domain, file_pickling, pickle.HIGHEST_PROTOCOL)
    file_pickling.close()

# end if __name__ == '__main__':