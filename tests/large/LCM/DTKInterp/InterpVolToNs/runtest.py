
#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0

######################
# Test 1 
######################
print "test 1 - DTKInterp Volume to NS Notched Cyl"
name = "DTKInterpVolumeToNsNotchedCyl"
log_file_name = name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')


# run DTK_Interp_and_Error 
command = ["mpirun", "-np", "4", "DTK_Interp_Volume_to_NS", "--xml-in-file=input_schwarz.yaml"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

sys.exit(result)
