
#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0

######################
# Test 1 
######################
print "test 1 - Schwarz Alternating"
name = "cuboid"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

#specify tolerance to determine test failure / passing
tol = 1.0e-06
total_iters = 22
abs_error = 3.11567926649862737e-15
rel_error = 5.26057123994272394e-16

# run Albany
command = ["./AlbanyT", "cuboids.yaml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

for line in open(log_file_name):
  if "Total iterations" in line:
    s = line
    s = line[20:]
    n = int(s)
    print ("Number of Schwarz iterations: ", n)
    print ("Expected                    : ", total_iters)
    if (n != total_iters):
      result = result+1 

  if "Last absolute error" in line:
    s = line
    s = line[20:]
    ae = float(s)
    f = ae / abs_error
    print ("Last absolute error: ", ae)
    print ("Expected           : ", abs_error)
    if (f < 1.0 - tol or 1.0 + tol < f):
      result = result+1 

  if "Last relative error" in line:
    s = line
    s = line[20:]
    re = float(s)
    f = re / rel_error
    print ("Last relative error: ", re)
    print ("Expected           : ", rel_error)
    if (f < 1.0 - tol or 1.0 + tol < f):
      result = result+1 

if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

sys.exit(result)
