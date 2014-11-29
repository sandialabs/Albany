##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
#! /usr/bin/env python

import sys
import os
import string
from subprocess import Popen

def runtest(albany_command, xml_file_name):

    result = 0
    
    log_file_name = xml_file_name[:-4] + ".log"
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    logfile = open(log_file_name, 'w')

    command = string.splitfields(albany_command, ";")
    command.append(xml_file_name)

    # run Albany    
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    if return_code != 0:
        result = return_code

    # run exodiff
#    command = ["./exodiff", "-stat", "-f", \
#                   base_name+".exodiff", \
#                   base_name+".gold.exo", \
#                   base_name+".exo"]
#    p = Popen(command, stdout=logfile, stderr=logfile)
#    return_code = p.wait()
#    if return_code != 0:
#        result = return_code

    logfile.close()
        
    return result

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print "\nError in run_exodiff_test.py: expected two argument, got", len(sys.argv)-1, ".\n"
        sys.exit(1)

    executable = sys.argv[1]
    xml_file_name = sys.argv[2]

    result = runtest(executable, xml_file_name)

    sys.exit(result)
