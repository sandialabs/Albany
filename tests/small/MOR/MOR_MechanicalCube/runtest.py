#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0

######################
# Test 1 - FOM
######################
print "test 1 - FOM"
name = "FOM"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

# run Albany 
command = ["./Albany", name + ".yaml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code


# run exodiff
command = ["./exodiff", "-stat", "-f", \
           name + ".exodiff", \
           name + ".gold.exo", \
           name + ".exo"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()

with open(log_file_name, 'r') as log_file:
    print log_file.read() 

if return_code != 0:
    result = return_code
    
if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

######################
# Test 2 - RBgen
######################
print "test 1 - RBgen"
name = "RB"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

# run Albany 
command = ["./AlbanyRBGen", "RBgen.xml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code


# run exodiff
command = ["./exodiff", "-stat", "-f", \
           name + ".exodiff", \
           name + ".gold.exo", \
           name + ".exo"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()

with open(log_file_name, 'r') as log_file:
    print log_file.read() 

if return_code != 0:
    result = return_code
if result != 0:
    print "result is %s" % result
    print "RBgen test has failed"
    sys.exit(result)
    
    
######################
# Test 3 - PGROM
######################
print "test 1 - PGROM"
name = "PGROM"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

# run Albany 
command = ["./Albany", name + ".yaml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code


# run exodiff
command = ["./exodiff", "-stat", "-f", \
           name + ".exodiff", \
           name + ".gold.exo", \
           name + ".exo"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()

with open(log_file_name, 'r') as log_file:
    print log_file.read()       

if return_code != 0:
    result = return_code
if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)
    
sys.exit(result)
