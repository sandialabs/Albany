
#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0

######################
# Test 1 
######################
print "test - Parallel Cubes 8 proc DBC"
name = "Parallel_Cubes_8_DBC"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

#specify tolerance to determine test failure / passing
tolerance = 1.0e-8; 
#meanvalue = 0.000594484007237; #meanvalue for 10 LOCA steps 
meanvalue = 0.000118897637152;

# run AlbanyT 
command = ["mpirun", "-np", "8", "AlbanyT", "cubes.yaml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

for line in open(log_file_name):
  if "Main_Solve: MeanValue of final solution" in line:
    s = line
    s = line[40:]
    d = float(s)
    print d
    if (d > meanvalue + tolerance or d < meanvalue - tolerance):
      result = result+1 

if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

with open(log_file_name, 'r') as log_file:
    print log_file.read() 

######################
# Test 2 
######################
print "test - Parallel Cubes 8 proc SDBC"
name = "Parallel_Cubes_8_SDBC"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

#specify tolerance to determine test failure / passing
tolerance = 1.0e-8; 
#meanvalue = 0.000594484007237; #meanvalue for 10 LOCA steps 
meanvalue = 0.000118897650651;

# run AlbanyT 
command = ["mpirun", "-np", "8", "AlbanyT", "cubes.yaml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

for line in open(log_file_name):
  if "Main_Solve: MeanValue of final solution" in line:
    s = line
    s = line[40:]
    d = float(s)
    print d
    if (d > meanvalue + tolerance or d < meanvalue - tolerance):
      result = result+1 

if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

with open(log_file_name, 'r') as log_file:
    print log_file.read() 


sys.exit(result)
