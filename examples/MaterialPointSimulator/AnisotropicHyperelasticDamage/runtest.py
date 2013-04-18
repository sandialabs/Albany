#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

result = 0

base_name = "AHD"

log_file_name = base_name + ".log"
if os.path.exists(log_file_name):
    os.remove(log_file_name)
logfile = open(log_file_name, 'w')

# run the point simulator
command = ["./MPS", "--input=\""+base_name+".xml\""]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

# run exodiff
command = ["./exodiff", "-stat", "-f", "Composite.exodiff", "Composite.gold.exo", "Composite.exo"]
p = Popen(command, stdout=logfile, stderr=logfile)
return_code = p.wait()
if return_code != 0:
    result = return_code

sys.exit(result)
