#! /usr/bin/env python
import sys
import os
import re

from subprocess import Popen

#
# Test 1 DBC
#
name = "cube_dbc"
log_file_name = name + ".log"
result = 0

print "test 1 - DBC"

if os.path.exists(log_file_name):
    os.remove(log_file_name)

logfile = open(log_file_name, 'w')

# run Albany
command = ["mpirun", "-np", "4",  "AlbanyT", "finer_cube_dbc.yaml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()

if return_code != 0:
    result = return_code

#specify tolerance to determine test failure / passing
tolerance = 1.0e-09;
meanvalue = 1.149197794509e-01;

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

with open(log_file_name, 'r') as log_file:
    print log_file.read()

#
# Test 2 DBC file
#
name = "cube_dbc_file"
log_file_name = name + ".log"

print "test 2 - DBC file"

if os.path.exists(log_file_name):
    os.remove(log_file_name)

logfile = open(log_file_name, 'w')

# run Albany
command = ["mpirun", "-np", "4",  "AlbanyT", "finer_cube_dbc_file.yaml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()

if return_code != 0:
    result = return_code

#specify tolerance to determine test failure / passing
tolerance = 1.0e-09;
meanvalue = 1.149197794509e-01;

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

with open(log_file_name, 'r') as log_file:
    print log_file.read()

#
# Test 3 SDBC
#
name = "cube_sdbc"
log_file_name = name + ".log"

print "test 3 - Strong DBC"

if os.path.exists(log_file_name):
    os.remove(log_file_name)

logfile = open(log_file_name, 'w')

# run Albany
command = ["mpirun", "-np", "4",  "AlbanyT", "finer_cube_sdbc.yaml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()

if return_code != 0:
    result = return_code

#specify tolerance to determine test failure / passing
tolerance = 1.0e-09;
meanvalue = 1.149197794509e-01;

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

with open(log_file_name, 'r') as log_file:
    print log_file.read()

#
# Test 4 SDBC file
#
name = "cube_sdbc_file"
log_file_name = name + ".log"

print "test 4 - Strong DBC file"

if os.path.exists(log_file_name):
    os.remove(log_file_name)

logfile = open(log_file_name, 'w')

# run Albany
command = ["mpirun", "-np", "4",  "AlbanyT", "finer_cube_sdbc_file.yaml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()

if return_code != 0:
    result = return_code

#specify tolerance to determine test failure / passing
tolerance = 1.0e-09;
meanvalue = 1.149197794509e-01;

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

with open(log_file_name, 'r') as log_file:
    print log_file.read()

sys.exit(result)
