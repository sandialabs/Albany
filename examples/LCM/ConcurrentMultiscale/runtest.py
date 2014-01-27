##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
#! /usr/bin/env python

import sys
import os
import re
from subprocess import Popen

def runtest( base_name ):
    result = 0
    
    log_file_name = base_name + ".log"
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    logfile = open(log_file_name, 'w')

    # run Albany
    command = ["./Albany", base_name+".xml\""]
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    if return_code != 0:
        result = return_code

    # run exodiff
    command = ["./exodiff", "-stat", "-f", \
                   base_name+".exodiff", \
                   base_name+".gold.exo", \
                   base_name+".exo"]
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    if return_code != 0:
        result = return_code
        
    #return sys.exit(result)
    return result
