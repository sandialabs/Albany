#!/usr/bin/python
'''
LCM_postprocess.py input.e output.e

creates usable output from LCM calculations
'''

import sys
import os
from exodus import exodus
from exodus import copy_mesh
import numpy
import matplotlib.pyplot as plt

inFileName = sys.argv[1]
outFileName = sys.argv[2]

inFile = exodus(inFileName,"r")
if os.path.isfile(outFileName):
  cmdLine = "rm %s" % outFileName
  os.system(cmdLine)
outFile = copy_mesh(inFileName, outFileName)

# get number of dimensions
num_dims = inFile.num_dimensions()

print "Dimensions?"
print num_dims

# get times
times = inFile.get_times()
print "Print the times"
for time in times:
  print time

# get list of nodal variables
node_var_names= inFile.get_node_variable_names()
print "Printing the names of the nodal variables"
for name in node_var_names:
    print name

# get list of element variables
elem_var_names= inFile.get_element_variable_names()
print "Printing the names of the element variables"
for name in elem_var_names:
    print name

# create any global variables


# figure out number of integration points
#print dir(elem_var_names[0])
# check that "Weights" exist as an element variable

num_points = 0
for name in elem_var_names:
  if (name.startswith("Weights_")):
      num_points += 1

if (num_points == 0):
  raise Exception("The weights field is not available...try again.")
      
print "Number of Integration points?"
print num_points

# write times to outFile
for step in range(len(times)):
  outFile.put_time(step+1,times[step])

# write out displacement vector
dx = []
dy = []
dz = []

outFile.set_node_variable_number(int(3))
outFile.put_node_variable_name('displacement_x', 1)
outFile.put_node_variable_name('displacement_y', 2)
outFile.put_node_variable_name('displacement_z', 3)

for step in range(len(times)):
  dx = inFile.get_node_variable_values('disp_x',step+1)
  dy = inFile.get_node_variable_values('disp_y',step+1)
  dz = inFile.get_node_variable_values('disp_z',step+1)
  outFile.put_node_variable_values('displacement_x',step+1,dx)
  outFile.put_node_variable_values('displacement_y',step+1,dy)
  outFile.put_node_variable_values('displacement_z',step+1,dz)
#

# 

#print dir(inFile)

outFile.close()
