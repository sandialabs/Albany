#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0

name = "DTKInterpNotchedCyl"
log_file_name = name + ".log"

with open(log_file_name, 'r') as log_file:
    print log_file.read()

#specify tolerance to determine test failure / passing
tolerance = 1.0e-16;
relerr = 1.13987e-13;

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
