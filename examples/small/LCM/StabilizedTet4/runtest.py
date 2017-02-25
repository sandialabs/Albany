#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0

######################
# Test 1 - Standard Galerkin
######################
print "test 1 - Galerkin FEM"
name = "galerkin"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

# run Albany 
command = ["./Albany", name + ".xml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

# run exodiff
command = ["./exodiff", "-stat", "-f", \
           name + ".exodiff", \
           name + ".gold.e", \
           name + ".e"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

######################
# Test 2 - Stabilized Galerkin
######################
print "test 2 - Stabilized Galerkin FEM"
name = "stabilized"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

# run Albany 
command = ["./Albany", name + ".xml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

# run exodiff
command = ["./exodiff", "-stat", "-f", \
           name + ".exodiff", \
           name + ".gold.e", \
           name + ".e"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

sys.exit(result)
