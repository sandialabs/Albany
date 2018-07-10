
#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0

name = "Parallel_Cubes_4_SDBC"
log_file_name = name + ".Log"

with open(log_file_name, 'r') as log_file:
    print log_file.read()

#specify tolerance to determine test failure / passing
tolerance = 1.0e-8;
#meanvalue = 0.000594484007237; #meanvalue for 10 LOCA steps
meanvalue = 0.000118897650651;

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

