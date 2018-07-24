
#! /usr/bin/env python

import sys
import os
import re
import subprocess
from subprocess import Popen

result = 0

######################
# Test 1 
######################
print "test 1 - CismAlbany"
name = "CismAlbany"
if os.path.exists('out'):
    os.remove('out')
if os.path.exists('coords_mismatch'):
    os.remove('coords_mismatch')
if os.path.exists('sh_mismatch'):
    os.remove('sh_mismatch')
if os.path.exists('thk_mismatch'):
    os.remove('thk_mismatch')
if os.path.exists('temp_mismatch'):
    os.remove('temp_mismatch')
if os.path.exists('uvel_mismatch'):
    os.remove('uvel_mismatch')
if os.path.exists('vvel_mismatch'):
    os.remove('vvel_mismatch')
if os.path.exists('beta_mismatch'):
    os.remove('beta_mismatch')
if os.path.exists('mismatches'):
    os.remove('mismatches')
subprocess.call("bash run_test.sh >& out",shell=True); 
subprocess.call("bash process_output_test.sh",shell=True)

with open('out', 'r') as log_file:
    print log_file.read() 


result = 0

log_file_name = 'mismatches'
log_file_exists = True

if os.path.isfile(log_file_name) == False:
    result = 100
    log_file_exists = False      

if log_file_exists == True: 
    logfile = open(log_file_name, 'r')
    
    length_mismatches = 0 

    i = 0
    with open(log_file_name) as f:
      for i, l in enumerate(f):
        pass
    length_mismatches = i + 1

    if length_mismatches != 7:
      result = 100
    else: 
      #specify tolerance to determine test failure / passing
      tolerance = 1.0e-9; 
      #mismatch = [9.8825e-06, 4.9986e-06, 4.9991e-06, 0.039632, 0.40334, 0.26009, 0.020565];
      mismatch = [1.1039e-05, 4.9986e-06, 4.9991e-06, 0.039632, 0.40334, 0.26009, 0.020565];

      i = 0
      for line in open(log_file_name):
        s = line
        d = float(s)
        m = d - mismatch[i]
        print 'mismatch', m
        i = i + 1
      if (m > tolerance or m < -tolerance):
        result = result+1 

      with open(log_file_name, 'r') as log_file:
        print log_file.read() 


if result != 0:
    print "result is %s" % result
    print "%s test has failed" % name
    sys.exit(result)

sys.exit(result)


