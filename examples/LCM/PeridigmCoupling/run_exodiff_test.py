##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
#! /usr/bin/env python

import sys
import os
import string
import glob
import re
from subprocess import Popen, PIPE

def runtest(albany_command, xml_file_name, final_step_only):

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

    exodus_extension = ".e"
    if num_processors > 1:
        exodus_extension = ".np" + str(num_processors) + ".e"

    final_step_extension = ""
    if final_step_only:
        final_step_extension = ".final_step"

    # remove old output files, if any
    files_to_remove = []
    if num_processors == 1:
        files_to_remove.append(base_name + ".e")
        if final_step_only:
            files_to_remove.append(base_name + final_step_extension + ".e")
    else:
        for i in range(num_processors):
            files_to_remove.append(base_name + ".e." + str(num_processors) + "." + str(i))
        files_to_remove.append(base_name + exodus_extension)
        if final_step_only:
            files_to_remove.append(base_name + final_step_extension + exodus_extension)
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

    # run epu
    if num_processors > 1:
        command = ["./epu", "-output_extension", exodus_extension[1:], "-p", str(num_processors), base_name]
        p = Popen(command, stdout=logfile, stderr=logfile)
        return_code = p.wait()
        if return_code != 0:
            result = return_code
        if has_peridigm_output:
            command = ["./epu", "-output_extension", exodus_extension[1:], "-p", str(num_processors), base_name+"_PeridigmResults"]
            p = Popen(command, stdout=logfile, stderr=logfile)
            return_code = p.wait()
            if return_code != 0:
                result = return_code

    if final_step_only:
        command = ["./algebra", \
               base_name + exodus_extension, \
               base_name + final_step_extension + exodus_extension]
        p = Popen(command, stdout=logfile, stderr=logfile, stdin=PIPE)
        stdout_data = p.communicate(input='tmin 100000000 \n save all \n exit')[0]
        return_code = p.wait()
        if return_code != 0:
            result = return_code

    # run exodiff on Albany output file
    command = ["./exodiff", "-stat", "-f", \
               base_name + final_step_extension + ".exodiff", \
               base_name + final_step_extension + ".gold.e", \
               base_name + final_step_extension + exodus_extension]
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

    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print "\nError in run_exodiff_test.py: expected two or three argument, got", len(sys.argv)-1, ".\n"
        sys.exit(1)

    executable = sys.argv[1]
    xml_file_name = sys.argv[2]

    final_step_only = False
    if "final_step_only" in sys.argv[-1]:
        final_step_only = True

    result = runtest(executable, xml_file_name, final_step_only)

    sys.exit(result)
