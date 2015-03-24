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
import re
from subprocess import Popen

def runtest(albany_command, xml_file_name):

    # When run as part of ctest, the albany_command string is a set of commands
    # separated by semicolons.  This will not be the case if this script
    # is run outside ctest.

    result = 0
    base_name = xml_file_name[:-4]
    
    # parse the Albany command and append the xml file name
    command = string.splitfields(albany_command, ";")
    command.append(xml_file_name)

    # determine the number of processors
    num_processors = 1
    if "-np" in command:
        index = command.index("-np")
        num_processors = int(command[index+1])

    # open the log file
    log_file_name = base_name + "_np" + str(num_processors) + ".log"
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    logfile = open(log_file_name, 'w')

    logfile.write("\nrun_exodiff_test.py command: " + str(command) + "\n\n")
    logfile.flush()

    # remove old output files, if any
    exodus_files = glob.glob(base_name + "*.e") + glob.glob(base_name + "*.e.*")
    files_to_remove = []
    for file_name in exodus_files:
        if "gold" not in file_name:
            files_to_remove.append(file_name)
    for file in os.listdir(os.getcwd()):
      if file in files_to_remove:
        os.remove(file)

    # run Albany    
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    if return_code != 0:
        result = return_code

    # determine if a separate Peridigm output file exists
    has_peridigm_output = False
    file_names = glob.glob("*.e")
    for name in file_names:
        if "_PeridigmResults" in name:
            has_peridigm_output = True

    exodus_extension = "e"
    if num_processors > 1:
        exodus_extension = "np" + str(num_processors) + ".e" 

    # run epu
    if num_processors > 1:
        command = ["./epu", "-output_extension", exodus_extension, "-p", str(num_processors), base_name]
        p = Popen(command, stdout=logfile, stderr=logfile)
        return_code = p.wait()
        if return_code != 0:
            result = return_code
        if has_peridigm_output:
            command = ["./epu", "-output_extension", exodus_extension, "-p", str(num_processors), base_name+"_PeridigmResults"]
            p = Popen(command, stdout=logfile, stderr=logfile)
            return_code = p.wait()
            if return_code != 0:
                result = return_code

    exodus_extension = "." + exodus_extension
            
    # run exodiff on Albany output file
    command = ["./exodiff", "-stat", "-f", \
               base_name+".exodiff", \
               base_name+".gold.e", \
               base_name+exodus_extension]
    p = Popen(command, stdout=logfile, stderr=logfile)
    return_code = p.wait()
    if return_code != 0:
        result = return_code

    # run exodiff on Peridigm output file, if any
    if has_peridigm_output:
        command = ["./exodiff", "-stat", "-f", \
                   base_name+"_PeridigmResults.exodiff", \
                   base_name+"_PeridigmResults.gold.e", \
                   base_name+"_PeridigmResults"+exodus_extension]
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
