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


# create variables in output file ------------------------------------------------------------------

outFile.set_element_variable_number(int(24))
outFile.put_element_variable_name('stress_cauchy_11', 1)
outFile.put_element_variable_name('stress_cauchy_12', 2)
outFile.put_element_variable_name('stress_cauchy_13', 3)
outFile.put_element_variable_name('stress_cauchy_22', 4)
outFile.put_element_variable_name('stress_cauchy_23', 5)
outFile.put_element_variable_name('stress_cauchy_33', 6)
outFile.put_element_variable_name('F_11', 7)
outFile.put_element_variable_name('F_12', 8)
outFile.put_element_variable_name('F_13', 9)
outFile.put_element_variable_name('F_21', 10)
outFile.put_element_variable_name('F_22', 11)
outFile.put_element_variable_name('F_23', 12)
outFile.put_element_variable_name('F_31', 13)
outFile.put_element_variable_name('F_32', 14)
outFile.put_element_variable_name('F_33', 15)
outFile.put_element_variable_name('Fp_11', 16)
outFile.put_element_variable_name('Fp_12', 17)
outFile.put_element_variable_name('Fp_13', 18)
outFile.put_element_variable_name('Fp_21', 19)
outFile.put_element_variable_name('Fp_22', 20)
outFile.put_element_variable_name('Fp_23', 21)
outFile.put_element_variable_name('Fp_31', 22)
outFile.put_element_variable_name('Fp_32', 23)
outFile.put_element_variable_name('Fp_33', 24)



# ==================================================================================================
# search through variables, find desired quantities, and output them -------------------------------
# ==================================================================================================

for name in names_variable_unique:


# --------------------------------------------------------------------------------------------------
# handle the cauchy stress -------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

  if (name == 'Cauchy_Stress'):
    for step in range(len(times)):
      for block in block_ids:

# ----- get the weighted contributions of the cauchy stresses --------------------------------------

        stress_cauchy_11 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Cauchy_Stress_01',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        stress_cauchy_12 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Cauchy_Stress_02',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        stress_cauchy_13 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Cauchy_Stress_03',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        stress_cauchy_22 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Cauchy_Stress_05',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        stress_cauchy_23 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Cauchy_Stress_06',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        stress_cauchy_33 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Cauchy_Stress_09',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        for point in range(1,num_points):

# ----- average the cauchy stress over the block ---------------------------------------------------

          stress_cauchy_11 = [x+y*z for (x,y,z) in 
            zip(
              stress_cauchy_11,
              inFile.get_element_variable_values(block,'Cauchy_Stress_' + str(point*num_dims**2+1),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          stress_cauchy_12 = [x+y*z for (x,y,z) in 
            zip(
              stress_cauchy_12,
              inFile.get_element_variable_values(block,'Cauchy_Stress_' + str(point*num_dims**2+2),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          stress_cauchy_13 = [x+y*z for (x,y,z) in 
            zip(
              stress_cauchy_13,
              inFile.get_element_variable_values(block,'Cauchy_Stress_' + str(point*num_dims**2+3),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          stress_cauchy_22 = [x+y*z for (x,y,z) in 
            zip(
              stress_cauchy_22,
              inFile.get_element_variable_values(block,'Cauchy_Stress_' + str(point*num_dims**2+5),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          stress_cauchy_23 = [x+y*z for (x,y,z) in 
            zip(
              stress_cauchy_23,
              inFile.get_element_variable_values(block,'Cauchy_Stress_' + str(point*num_dims**2+6),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          stress_cauchy_33 = [x+y*z for (x,y,z) in 
            zip(
              stress_cauchy_33,
              inFile.get_element_variable_values(block,'Cauchy_Stress_' + str(point*num_dims**2+9),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

# ----- output the volume-averaged quantity --------------------------------------------------------
        
        outFile.put_element_variable_values(block,'stress_cauchy_11',step+1,stress_cauchy_11)
        
        outFile.put_element_variable_values(block,'stress_cauchy_12',step+1,stress_cauchy_12)
        
        outFile.put_element_variable_values(block,'stress_cauchy_13',step+1,stress_cauchy_13)
        
        outFile.put_element_variable_values(block,'stress_cauchy_22',step+1,stress_cauchy_22)
        
        outFile.put_element_variable_values(block,'stress_cauchy_23',step+1,stress_cauchy_23)
        
        outFile.put_element_variable_values(block,'stress_cauchy_33',step+1,stress_cauchy_33)


# --------------------------------------------------------------------------------------------------
# handle the deformation gradient ------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

  if (name == 'F'):
    for step in range(len(times)):
      for block in block_ids:

# ----- get the weighted contributions of the deformation gradient ---------------------------------

        F_11 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'F_01',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        F_12 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'F_02',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        F_13 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'F_03',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        F_21 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'F_04',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        F_22 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'F_05',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        F_23 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'F_06',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        F_31 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'F_07',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        F_32 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'F_08',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        F_33 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'F_09',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        for point in range(1,num_points):

# ----- average the deformation gradient over the block --------------------------------------------

          F_11 = [x+y*z for (x,y,z) in 
            zip(
              F_11,
              inFile.get_element_variable_values(block,'F_' + str(point*num_dims**2+1),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          F_12 = [x+y*z for (x,y,z) in 
            zip(
              F_12,
              inFile.get_element_variable_values(block,'F_' + str(point*num_dims**2+2),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          F_13 = [x+y*z for (x,y,z) in 
            zip(
              F_13,
              inFile.get_element_variable_values(block,'F_' + str(point*num_dims**2+3),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          F_21 = [x+y*z for (x,y,z) in 
            zip(
              F_21,
              inFile.get_element_variable_values(block,'F_' + str(point*num_dims**2+4),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          F_22 = [x+y*z for (x,y,z) in 
            zip(
              F_22,
              inFile.get_element_variable_values(block,'F_' + str(point*num_dims**2+5),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          F_23 = [x+y*z for (x,y,z) in 
            zip(
              F_23,
              inFile.get_element_variable_values(block,'F_' + str(point*num_dims**2+6),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          F_31 = [x+y*z for (x,y,z) in 
            zip(
              F_31,
              inFile.get_element_variable_values(block,'F_' + str(point*num_dims**2+7),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          F_32 = [x+y*z for (x,y,z) in 
            zip(
              F_32,
              inFile.get_element_variable_values(block,'F_' + str(point*num_dims**2+8),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          F_33 = [x+y*z for (x,y,z) in 
            zip(
              F_33,
              inFile.get_element_variable_values(block,'F_' + str(point*num_dims**2+9),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

# ----- output the volume-averaged quantity --------------------------------------------------------
        
        outFile.put_element_variable_values(block,'F_11',step+1,F_11)
        
        outFile.put_element_variable_values(block,'F_12',step+1,F_12)
        
        outFile.put_element_variable_values(block,'F_13',step+1,F_13)
        
        outFile.put_element_variable_values(block,'F_21',step+1,F_21)
        
        outFile.put_element_variable_values(block,'F_22',step+1,F_22)
        
        outFile.put_element_variable_values(block,'F_23',step+1,F_23)
        
        outFile.put_element_variable_values(block,'F_31',step+1,F_31)
        
        outFile.put_element_variable_values(block,'F_32',step+1,F_32)
        
        outFile.put_element_variable_values(block,'F_33',step+1,F_33)


# --------------------------------------------------------------------------------------------------
# handle the plastic deformation gradient ----------------------------------------------------------
# --------------------------------------------------------------------------------------------------

  if (name == 'Fp'):
    for step in range(len(times)):
      for block in block_ids:

# ----- get the weighted contributions of the plastic deformation gradient -------------------------

        Fp_11 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Fp_01',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        Fp_12 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Fp_02',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        Fp_13 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Fp_03',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        Fp_21 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Fp_04',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        Fp_22 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Fp_05',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        Fp_23 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Fp_06',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        Fp_31 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Fp_07',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        Fp_32 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Fp_08',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        Fp_33 = [x*y for (x,y) in 
          zip(inFile.get_element_variable_values(block,'Fp_09',step+1), 
            inFile.get_element_variable_values(block,'Weights_1',step+1))]

        for point in range(1,num_points):

# ----- average the plastic deformation gradient over the block ------------------------------------

          Fp_11 = [x+y*z for (x,y,z) in 
            zip(
              Fp_11,
              inFile.get_element_variable_values(block,'Fp_' + str(point*num_dims**2+1),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          Fp_12 = [x+y*z for (x,y,z) in 
            zip(
              Fp_12,
              inFile.get_element_variable_values(block,'Fp_' + str(point*num_dims**2+2),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          Fp_13 = [x+y*z for (x,y,z) in 
            zip(
              Fp_13,
              inFile.get_element_variable_values(block,'Fp_' + str(point*num_dims**2+3),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          Fp_21 = [x+y*z for (x,y,z) in 
            zip(
              Fp_21,
              inFile.get_element_variable_values(block,'Fp_' + str(point*num_dims**2+4),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          Fp_22 = [x+y*z for (x,y,z) in 
            zip(
              Fp_22,
              inFile.get_element_variable_values(block,'Fp_' + str(point*num_dims**2+5),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          Fp_23 = [x+y*z for (x,y,z) in 
            zip(
              Fp_23,
              inFile.get_element_variable_values(block,'Fp_' + str(point*num_dims**2+6),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          Fp_31 = [x+y*z for (x,y,z) in 
            zip(
              Fp_31,
              inFile.get_element_variable_values(block,'Fp_' + str(point*num_dims**2+7),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          Fp_32 = [x+y*z for (x,y,z) in 
            zip(
              Fp_32,
              inFile.get_element_variable_values(block,'Fp_' + str(point*num_dims**2+8),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

          Fp_33 = [x+y*z for (x,y,z) in 
            zip(
              Fp_33,
              inFile.get_element_variable_values(block,'Fp_' + str(point*num_dims**2+9),step+1), 
              inFile.get_element_variable_values(block,'Weights_'+str(point),step+1))]

# ----- output the volume-averaged quantity --------------------------------------------------------
        
        outFile.put_element_variable_values(block,'Fp_11',step+1,Fp_11)
        
        outFile.put_element_variable_values(block,'Fp_12',step+1,Fp_12)
        
        outFile.put_element_variable_values(block,'Fp_13',step+1,Fp_13)
        
        outFile.put_element_variable_values(block,'Fp_21',step+1,Fp_21)
        
        outFile.put_element_variable_values(block,'Fp_22',step+1,Fp_22)
        
        outFile.put_element_variable_values(block,'Fp_23',step+1,Fp_23)
        
        outFile.put_element_variable_values(block,'Fp_31',step+1,Fp_31)
        
        outFile.put_element_variable_values(block,'Fp_32',step+1,Fp_32)
        
        outFile.put_element_variable_values(block,'Fp_33',step+1,Fp_33)



outFile.close()
