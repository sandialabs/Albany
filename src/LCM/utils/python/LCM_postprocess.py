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


# set i/o units ------------------------------------------------------------------------------------

inFile = exodus(inFileName,"r")
if os.path.isfile(outFileName):
  cmdLine = "rm %s" % outFileName
  os.system(cmdLine)
outFile = copy_mesh(inFileName, outFileName)


# get number of dimensions -------------------------------------------------------------------------
num_dims = inFile.num_dimensions()

print "Dimensions"
print num_dims


# get times ----------------------------------------------------------------------------------------
times = inFile.get_times()
print "Print the times"
print times[0], times[-1]


# get list of nodal variables ----------------------------------------------------------------------
node_var_names= inFile.get_node_variable_names()
print "Printing the names of the nodal variables"
print node_var_names


# get list of element variables --------------------------------------------------------------------
elem_var_names = inFile.get_element_variable_names()
num_variables_unique = 0
names_variable_unique = []
print "Printing the names of the element variables"
for name in elem_var_names:
    new_name = True
    for name_unique in names_variable_unique:
      if (name.startswith(name_unique+"_")):
        new_name = False
    if (new_name == True):
      indices = [int(s) for s in name if s.isdigit()]
      names_variable_unique.append(name[0:name.find(str(indices[0]))-1])
print names_variable_unique


# get number of element blocks and bloc ids --------------------------------------------------------
block_ids = inFile.get_elem_blk_ids()
num_blocks = inFile.num_blks()




# create any global variables ----------------------------------------------------------------------


# figure out number of integration points ----------------------------------------------------------
num_points = 0
for name in elem_var_names:
  if (name.startswith("Weights_")):
      num_points += 1

# check that "Weights" exist as an element variable ------------------------------------------------
if (num_points == 0):
  raise Exception("The weights field is not available...try again.")
      
print "Number of Integration points"
print num_points

# write times to outFile ---------------------------------------------------------------------------
for step in range(len(times)):
  outFile.put_time(step+1,times[step])

# write out displacement vector --------------------------------------------------------------------
dx = []
dy = []
dz = []

outFile.set_node_variable_number(int(3))
outFile.put_node_variable_name('displacement_x', 1)
outFile.put_node_variable_name('displacement_y', 2)
outFile.put_node_variable_name('displacement_z', 3)

for step in range(len(times)):

  dx = inFile.get_node_variable_values('displacement_x',step+1)
  dy = inFile.get_node_variable_values('displacement_y',step+1)
  dz = inFile.get_node_variable_values('displacement_z',step+1)

  outFile.put_node_variable_values('displacement_x',step+1,dx)
  outFile.put_node_variable_values('displacement_y',step+1,dy)
  outFile.put_node_variable_values('displacement_z',step+1,dz)


# outFile.close()
# sys.exit()

outFile.set_element_variable_number(int(1))
outFile.put_element_variable_name('stress_cauchy_11', 1)

for name in names_variable_unique:
  if (name == 'Cauchy_Stress'):
    for step in range(len(times)):
      for block in block_ids:
        stress_cauchy_11 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Cauchy_Stress_01',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        for point in range(1,num_points):

          stress_cauchy_11 = [x+y*z for (x,y,z) in 
            zip(
              stress_cauchy_11,
              inFile.get_element_variable_values(block,'Cauchy_Stress_' + str(point*num_dims**2+1),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]
        
        outFile.put_element_variable_values(block,'stress_cauchy_11',step+1,stress_cauchy_11)

# 

#print dir(inFile)

outFile.close()
