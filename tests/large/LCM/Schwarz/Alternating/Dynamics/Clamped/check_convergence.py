#! /usr/bin/env python
import sys
import os
import re

from subprocess import Popen

name = "clamped_parallel"
log_file_name = name + ".log"
result = 0

with open(log_file_name, 'r') as log_file:
    print log_file.read()

converged = False

for line in open(log_file_name):
  if "Schwarz Alternating Method converged: YES" in line:
    converged = True

for line in open(log_file_name):
  if "Schwarz Alternating Method converged: NO" in line:
    converged = False

if converged == False:
  result = result + 1

with open(log_file_name, 'r') as log_file:
    print log_file.read()

if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name


sys.exit(result)
