
#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0

######################
# Test 1 
######################
print "test 1 - yaml input format"
name = "cuboid"

#specify tolerance to determine test failure / passing
tolerance = 1.0e-9; 
meanvalue = 0.281586446117;

# single domain run using XML
command = ["./AlbanyT", "cuboid.xml"]
log_file_name = name + ".xml.log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

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

# single domain run using Yaml
command = ["./AlbanyT", "cuboid.yaml"]
log_file_name = name + ".yaml.log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

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
    
name = "cuboids"

# two domain run using XML
command = ["./AlbanyT", "cuboids.xml"]
log_file_name = name + ".xml.log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

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

# two domain run using YAML
command = ["./AlbanyT", "cuboids.yaml"]
log_file_name = name + ".yaml.log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

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

sys.exit(result)
