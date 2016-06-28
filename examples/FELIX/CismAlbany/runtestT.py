
#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0

######################
# Test 1 
######################
print "test 1 - CismAlbany"
name = "CismAlbany"
log_file_name = 'mismatches'
logfile = open(log_file_name, 'r')

#specify tolerance to determine test failure / passing
tolerance = 1.0e-9; 
mismatch = [9.8825e-06, 4.9986e-06, 4.9991e-06, 0.039632, 0.40334, 0.26009, 0.020565];
result = 0
i = 0
for line in open(log_file_name):
  s = line
  d = float(s)
  m = d - mismatch[i]
  print 'mismatch', m
  i = i + 1
  if (m > tolerance or m < -tolerance):
    result = result+1 

if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

sys.exit(result)
