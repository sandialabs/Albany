#!/usr/bin/python
'''
read_file_log.py
'''

import numpy as np
import re
import lcm_postprocess

# Read the console output file
def read_file_log(filename = None):

    #
    # Read the log file
    #
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()

    #
    # Find start of continuation steps
    #
    lines_step_start = [line for line in lines if line.lower().find('start of continuation step') != -1]
    lines_step_end = [line for line in lines if line.lower().find('end of continuation step') != -1]

    times = []

    run = lcm_postprocess.ObjRun()

    index_step_start = -1

    indices_step_start = {}

    for line in lines:

        #
        # Find the start of the step
        #
        match = re.search('Start of Continuation Step\s+([0-9]+)', line)

        if type(match) != type(None):

            step_number = int(match.group(1))

            match = re.search('(\d+\.\d+e[\+,-]\d+) from (\d+\.\d+e[\+,-]\d+)', line)

            key_times = (0.0, 0.0)

            if type(match) != type(None):

                key_times = (float(match.group(2)), float(match.group(1)))

            run.steps[key_times] = lcm_postprocess.ObjStep(step_number = step_number)

            step = run.steps[key_times]

            step.size_step = key_times[1] - key_times[0]

            continue

        #
        # Find the end of the step
        #
        match = re.search('Step Converged in\s+([0-9]+)', line)

        if type(match) != type(None):

            step.num_iters_nonlinear = int(match.group(1))

            continue

        match = re.search('End of Continuation Step\s+([0-9]+)', line)

        if type(match) != type(None):

            step.status_convergence = 1

            continue

        match = re.search('([0-9]+)\s+experienced a convergence failure', line)

        if type(match) != type(None):

            step.status_convergence = -1

            continue

        #
        # Extract lines that have nonlinear iteration information
        #
        match = re.search('Nonlinear Solver Step\s+([0-9]+)', line)

        if type(match) != type(None):

            key_iter_nonlinear = int(match.group(1))

            step.iters_nonlinear[key_iter_nonlinear] = lcm_postprocess.ObjIterNonlinear()

            step.num_iters_nonlinear = key_iter_nonlinear

            iter_nonlinear = step.iters_nonlinear[key_iter_nonlinear]

            continue

        match = re.search('\|\|F\|\|\s+=\s+(\d+\.\d+e[\+,-]\d+)', line)

        if type(match) != type(None):

            iter_nonlinear.norm_residual = float(match.group(1))

            match = re.search('dx\s+=\s+(\d+\.\d+e[\+,-]\d+)', line)

            iter_nonlinear.norm_increment = float(match.group(1))

            match = re.search('\((\w+)\!\)', line)

            if type(match) != type(None):

                if match.group(1) == 'Failed':

                    iter_nonlinear.status_convergence = -1

                elif match.group(1) == 'Converged':

                    iter_nonlinear.status_convergence = 1

            continue

        #
        # Extract lines that have linear iteration information
        #
        match = re.search('Iter\s+(\d)', line)

        if type(match) != type(None):

            key_iter_linear = int(match.group(1))

            iter_nonlinear.iters_linear[key_iter_linear] = lcm_postprocess.ObjIterLinear()

            iter_linear = iter_nonlinear.iters_linear[key_iter_linear]

            match = re.search(':\s+(\d+\.\d+e[\+,-]\d+)', line)

            try:
                iter_linear.norm_residual = float(match.group(1))
            except:
                print '***WARNING***'
                print 'Exception while reading linear iteration line:'
                print line
                pass

            iter_nonlinear.num_iters_linear = key_iter_linear

            continue

        #
        # Extract lines that have timing information
        #
        match = re.search('TimeMonitor results over\s+([0-9]+) processor', line)

        if type(match) != type(None):

            run.num_processors = int(match.group(1))

            if run.num_processors > 1:

                index_split = 2

            else:

                index_split = 0

            continue

        match = re.search('\*\*\*Total Time\*\*\*\s+(.+)', line)

        if type(match) != type(None):

            run.time_compute = float(match.group(1).split()[index_split])

        match = re.search('ConstitutiveModelInterface\S+\s+(.+)', line)

        if type(match) != type(None):

            run.time_constitutive = float(match.group(1).split()[index_split])

            continue

        match = re.search('total solve time\s+(.+)', line)

        if type(match) != type(None):

            run.time_linsolve = float(match.group(1).split()[index_split])

            continue


    #
    # Compile step and run data
    #
    for key_step in run.steps:

        step = run.steps[key_step]

        step.num_iters_linear = np.sum([step.iters_nonlinear[x].num_iters_linear for x in step.iters_nonlinear])

    run.num_iters_nonlinear = np.sum([run.steps[x].num_iters_nonlinear for x in run.steps])

    run.num_iters_linear = np.sum([run.steps[x].num_iters_linear for x in run.steps])

    run.num_steps = len(run.steps)

    return run

# end def read_file_log(filename):



if __name__ == '__main__':

    import re
    import sys
    import cPickle as pickle

    try:
        name_file_input = sys.argv[1]
    except:
        raise

    match = re.search('(.+)_Log.out', name_file_input)
    if type(match) != type(None):
        parts_path = match.group(1).split('/')
        name_file_base = parts_path[-1]
        match = re.search('(.*)' + name_file_base, name_file_input)
        path_file = match.group(1)

    if len(sys.argv) == 2:

        run = read_file_log(filename = name_file_input)

    file_pickling = open(name_file_base + '_Run.pickle', 'wb')
    pickle.dump(run, file_pickling, pickle.HIGHEST_PROTOCOL)
    file_pickling.close()

# end if __name__ == '__main__':
