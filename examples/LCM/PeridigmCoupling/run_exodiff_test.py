##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
#! /usr/bin/env python

import sys
import os
import string
import glob
from subprocess import Popen

def runtest(albany_command, xml_file_name):

    result = 0

    base_name = xml_file_name[:-4]

    # parse the Albany command and append the xml file name
    command = string.splitfields(albany_command, ";")
    command.append(xml_file_name)

    # open log file    
    log_file_name = base_name + ".log"
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    logfile = open(log_file_name, 'w')

    # remove old output files, if any
    files_to_remove = glob.glob(base_name+".e")
    for file in os.listdir(os.getcwd()):
      if file in files_to_remove:
        os.remove(file)

    # run Albany    
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    if return_code != 0:
        result = return_code

    # run exodiff
    command = ["./exodiff", "-stat", "-f", \
                   base_name+".exodiff", \
                   base_name+".gold.e", \
                   base_name+".e"]
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    if return_code != 0:
        result = return_code

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
