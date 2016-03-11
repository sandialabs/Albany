
#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0

######################
# Test 1 
######################
print "test 1 - DTKInterp Notched Cyl"
name = "DTKInterpNotchedCyl"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

#specify tolerance to determine test failure / passing
tolerance = 1.0e-16; 
relerr = 1.13987e-13;

# run DTK_Interp_and_Error 
command = ["mpirun", "-np", "4", "DTK_Interp_and_Error", "--xml-in-file=input_disp.xml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

for line in open(log_file_name):
  if " Dof = 2, |e|_2 / |f|_2 (rel error):" in line:
    s = line
    s = line[42:]
    d = float(s)
    print d
    if (d > relerr + tolerance or d < relerr - tolerance):
      result = result+1 

if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

sys.exit(result)
