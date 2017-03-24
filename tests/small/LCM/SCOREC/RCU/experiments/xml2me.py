#!/usr/bin/python
# Read an Albany xml file and output the equivalent me2xml .cpp file.

import sys, re, glob
import xml.etree.ElementTree as et

def translate (root, fd, indent):
    if root.tag == 'ParameterList':
        fd.write(' '*indent + '{ pl a')
        if len(root.attrib) > 0:
            fd.write('("{0}")'.format(root.attrib['name']))
        fd.write(';\n')
        for child in root:
            translate(child, fd, indent+2)
        fd.write(' '*indent + '}')
        if len(root.attrib) > 0:
            fd.write(' // {0}'.format(root.attrib['name']))
        fd.write('\n')
    elif root.tag == 'Parameter':
        fd.write(' '*indent)
        if root.attrib['type'] == 'int':
            fd.write('p("{0}", {1});\n'.format(
                root.attrib['name'], root.attrib['value']))
        elif root.attrib['type'] == 'bool':
            try:
                value = str(bool(int(root.attrib['value']))).lower()
            except:
                value = root.attrib['value'].lower()
            fd.write('p("{0}", {1});\n'.format(root.attrib['name'], value))
        elif root.attrib['type'] == 'double':
            value = float(root.attrib['value'])
            if value == round(value):
                a2 = '{1:1.1f}'
            else:
                a2 = '{1}'
            fd.write(('p("{0}", ' + a2 + ');\n').format(
                root.attrib['name'], value))
        elif root.attrib['type'] == 'string':
            fd.write('p("{0}", "{1}");\n'.format(
                root.attrib['name'], root.attrib['value']))
        else:
            fd.write('p("{0}", "{1}", "{2}");\n'.format(
                root.attrib['name'], root.attrib['type'], root.attrib['value']))
         
def xml2me (fn):
    print fn
    fd = open(fn + '.cpp', 'a') # Append to be safe.
    fd.write('//bld g++ -g -I ~/code/util {0}.cpp\n\n'.format(fn) +
             '#include "me2xml.hpp"\n\n' +
             'void xml () {\n')
    root = et.parse(fn + '.xml').getroot()
    translate(root, fd, 2)
    fd.write('}\n');
    fd.close()

if __name__ == '__main__':
    for a in sys.argv[1:]: xml2me(a)
