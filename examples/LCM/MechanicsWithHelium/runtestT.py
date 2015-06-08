#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0

######################
# Test 1 - Mechanics
######################
print "test 1 - just Mechanics"
name = "Mechanics"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

# run AlbanyT
command = ["./AlbanyT", name + ".xml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

# run exodiff"
command = ["./exodiff", "-file", \
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
# Test 2 - Mechanics And Hydrogen
######################
print "test 2 - Mechanics and Hydrogen"
name = "MechanicsAndHydrogen"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

# run AlbanyT 
command = ["./AlbanyT", name + ".xml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

# run exodiff
command = ["./exodiff", "-file", \
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
# Test 3 - Mechanics And Hydrogen version 2
######################
print "test 3 - Mechanics and Hydrogen version 2"
name = "MechanicsAndHydrogenV2"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

# run AlbanyT 
command = ["./AlbanyT", name + ".xml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

# run exodiff
command = ["./exodiff", "-file", \
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
# Test 4 - Mechanics And Helium
######################
print "test 4 - Mechanics and Helium"
name = "MechanicsAndHelium"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

# run AlbanyT 
command = ["./AlbanyT", name + ".xml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

# run exodiff
command = ["./exodiff", "-file", \
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
# Test 5 - Mechanics And Helium version 2
######################
print "test 5 - Mechanics and Helium version 2"
name = "MechanicsAndHeliumV2"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

# run AlbanyT 
command = ["./AlbanyT", name + ".xml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

# run exodiff
command = ["./exodiff", "-file", \
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
