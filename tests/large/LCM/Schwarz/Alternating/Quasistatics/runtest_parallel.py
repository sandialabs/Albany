#! /usr/bin/env python
import sys
import os
import re

from subprocess import Popen

name = "cuboid_parallel"
log_file_name = name + ".log"
result = 0

print "test 1 - Schwarz Alternating Parallel"

if os.path.exists(log_file_name):
    os.remove(log_file_name)
    
logfile = open(log_file_name, 'w')

# run Albany
command = ["mpirun", "-np", "2", "AlbanyT", "cuboids.yaml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()

if return_code != 0:
    result = return_code

converged = False

for line in open(log_file_name):
  if "Schwarz Alternating Method converged: YES" in line:
    converged = True

if converged == False:    
  result = result + 1

if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name

with open(log_file_name, 'r') as log_file:
    print log_file.read() 


sys.exit(result)
