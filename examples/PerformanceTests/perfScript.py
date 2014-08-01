#! /usr/bin/env python
# usage:  python this-script -machine machineName -executable executableName
#  errors will be in:  perfTest.log
#
# Thanks to Dave Littlewood for creating this script  10/2013.

import sys
import os
import glob
import string
from subprocess import Popen, PIPE

base_name = "perfTest"

def read_line(file):
    """Scans the input file and ignores lines starting with a '#' or '\n'."""
    
    buff = file.readline()
    if len(buff) == 0: return None
    while buff[0] == '#' or buff[0] == '\n':
        buff = file.readline()
        if len(buff) == 0: return None
    return buff

if __name__ == "__main__":

    result = 0

    # open log file
    log_file_name = base_name + ".log"
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    logfile = open(log_file_name, 'w')

    # log file will be dumped if verbose option is given
    verbose = False
    if "-verbose" in sys.argv:
        verbose = True

    machine_name = "None"
    if "-machine" in sys.argv:
        machine_name = sys.argv[sys.argv.index("-machine") + 1]
        logfile.write("\n**** machine name = " + machine_name + "\n")
    else:
        logfile.write("\n**** Error, machine name argument required (-machine my_machine_name)\n")
        result = 1

    # gold standard performance data for this machine
    perf_gold_file = open("data.perf")
    buff = read_line(perf_gold_file)
    gold_perf_data = []
    while buff != None:
        vals = string.splitfields(buff)
        if machine_name in vals:
            gold_perf_data = vals
        buff = read_line(perf_gold_file)
    if gold_perf_data == []:
        logfile.write("\n**** Error, reference (gold) performance data not found for machine " + machine_name + "\n")
        result = 1
    gold_num_proc = int(gold_perf_data[1])
    gold_wallclock_time = float(gold_perf_data[2])
    gold_wallclock_time_tolerance = float(gold_perf_data[3])

    path_name = "None"
    if "-executable" in sys.argv:
        path_name = sys.argv[sys.argv.index("-executable") + 1]
    else:
        result = 1

    executable_name = path_name + "/" + gold_perf_data[4]
    input_file_name = gold_perf_data[5]
    logfile.write("\n**** Executable name = " + executable_name + "\n")
    logfile.write("\n**** Input file name = " + input_file_name + "\n")

    if gold_num_proc == 1:
    # run Albany: serial for now
        command = [executable_name, input_file_name]    
    else:
        command = ["mpirun", "-np", gold_perf_data[1], executable_name, input_file_name]    

    p = Popen(command, stdout=PIPE)
    return_code = p.wait()
    if return_code != 0:
        result = return_code
    out, err = p.communicate()
    if out != None:
        logfile.write(out)
    if err != None:
        logfile.write(err)
    logfile.flush()
        
    # compare performance statistics against gold statistics

    # performance data for current run
    stdout_vals = string.splitfields(out)
    wallclock_time_index = stdout_vals.index("Time***")
    wallclock_time = float(stdout_vals[wallclock_time_index+1])


    if(wallclock_time > gold_wallclock_time + gold_wallclock_time_tolerance):
        result = 1
        logfile.write("\n**** PERFORMANCE TEST FAILED:  wallclock time exceeded benchmark value plus tolerance.")
    elif(wallclock_time < gold_wallclock_time - gold_wallclock_time_tolerance):
        result = 1
        logfile.write("\n**** PERFORMANCE TEST FAILED:  wallclock time was LESS than benchmark value minus tolerance (code is running TOO FAST!).")
    else:
        logfile.write("\n**** PERFORMANCE TEST PASSED:  wallclock time was within tolerance.")
    logfile.write("\n****                           wallclock time  = " +  str(wallclock_time))
    logfile.write("\n****                           benchmark value = " +  str(gold_wallclock_time))
    logfile.write("\n****                           tolerance       = " +  str(gold_wallclock_time_tolerance) +"\n")
    logfile.flush()
          
    # compare output against gold file only if the gold file is present
    logfile.close()

    # dump the output if the user requested verbose
    if verbose == True:
        os.system("cat " + log_file_name)

    sys.exit(result)
