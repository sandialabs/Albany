#!/usr/bin/python
'''
LCM_postprocess.py input.e

creates usable output from LCM calculations
'''

#
# Imported modules
#
import contextlib
import cPickle as pickle
import cStringIO
import matplotlib.pyplot as plt
from matplotlib import rcParams
from multiprocessing import Pool
import numpy as np
from operator import itemgetter
import os
from scipy.linalg import *
import sys
import time

from lcm_postprocess.read_file_output_exodus import read_file_output_exodus
from lcm_postprocess.read_file_log import read_file_log
from lcm_postprocess.read_file_input_material import read_file_input_material
from lcm_postprocess.derive_value_variable import derive_value_variable
from lcm_postprocess.write_file_exodus import write_file_exodus
from lcm_postprocess.plot_data_run import plot_data_run
from lcm_postprocess.plot_inverse_pole_figure import plot_inverse_pole_figure
from lcm_postprocess.plot_data_stress import plot_data_stress




#
# Local classes
#
class Timer:  

    def __enter__(self):
        self.start = time.clock()
        self.now = self.start
        self.last = self.now
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

    def check(self):
        self.now = time.clock()
        self.step = self.now - self.last
        self.interval = self.now - self.start
        self.last = self.now




#
# Context manager for silencing output
#
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()
    yield
    sys.stdout = save_stdout



#
# Local functions
#











  





 









# Write select data to text file
def write_data(domain, name_file_data, precision = 8):

    file = open(name_file_data, 'w')

    str_format = '%.'+str(precision)+'e'

    file.write('Deformation Gradient\n')
    for step in domain.F:
        domain.F[step].tofile(file, sep = ' ', format = str_format)
        file.write('\n')

    file.write('Cauchy Stress\n')
    for step in domain.Cauchy_Stress:
        domain.Cauchy_Stress[step].tofile(file, sep = ' ', format = str_format)
        file.write('\n')

    file.close()

# end def write_data(data, name_file_data)




























#
# Main function to postprocess simulation data
#
def postprocess(
    name_file_output_exodus,
    plotting = False,
    pickling = True,
    verbosity = 1,
    **kwargs):



    print ''



    #
    # Set i/o units 
    #
    name_file_base = name_file_output_exodus.split('.')[0]
    name_file_extension = name_file_output_exodus.split('.')[-1]




    #
    # Create the simulation time step structure
    #
    with Timer() as timer:

        if verbosity > 0:
            print 'Creating simulation time step object...'

        try:

            run = read_file_log(name_file_base + '_Log.out')

        except IOError:

            print '    No log file found.'

    if verbosity > 0:
        print '    Elapsed time: ' + str(timer.interval) + 's\n'




    #
    # Get values of whole domain variables
    #
    with Timer() as timer:

        if verbosity > 0:
            print 'Reading exodus output file...'

        domain = read_file_output_exodus(filename = name_file_output_exodus)

    if verbosity > 0:
        print '    Elapsed time: ' + str(timer.interval) + 's\n'





    #
    # Get material data
    #
    with Timer() as timer:

        if verbosity > 0:
            print 'Retrieving material data...'

        read_file_input_material(
            name_file_base + '_Material.xml', 
            domain = domain,
            names_variable = ['orientations'])

    if verbosity > 0:
        print '    Elapsed time: ' + str(timer.interval) + 's\n'



    
    #
    # Compute derived variable values
    #
    with Timer() as timer:

        if verbosity > 0:
            print 'Deriving output data...'
    
        derive_value_variable(domain)

    if verbosity > 0:
        print '    Elapsed time: ' + str(timer.interval) + 's\n'



    #
    # Write data to exodus output file
    #
    with Timer() as timer:

        if verbosity > 0:
            print 'Writing data to exodus file...'

        if 'name_file_output' in kwargs:
            name_file_postprocess = kwargs['name_file_output']
        else:
            name_file_postprocess = name_file_base + '_Postprocess.' + name_file_extension
        
        write_file_exodus(domain, name_file_output_exodus, name_file_postprocess)

    if verbosity > 0:
        print '    Elapsed time: ' + str(timer.interval) + 's\n'




    #
    # Plot the convergence data
    #
    with Timer() as timer:

        if verbosity > 0:
            print 'Plotting convergence data...'

        if plotting == True:

            plot_data_run(run = run)

    if verbosity > 0:
        print '    Elapsed time: ' + str(timer.interval) + 's\n'




    #
    # Plot the inverse pole figures
    #
    with Timer() as timer:

        if verbosity > 0:
            print 'Plotting inverse pole figures...'

        if plotting == True:
            
            for step in [domain.times[0], domain.times[-1]]:

                plot_inverse_pole_figure(domain = domain, time = step)

    if verbosity > 0:
        print '    Elapsed time: ' + str(timer.interval) + 's\n'



    #
    # Plot stress-strain data
    #
    with Timer() as timer:

        if verbosity > 0:
            print 'Plotting stress-strain data...'

        if plotting == True:
            plot_data_stress(domain = domain)

    if verbosity > 0:
        print '    Elapsed time: ' + str(timer.interval) + 's\n'

    if pickling == True:

        file_pickling = open(name_file_base + '_Domain.pickle', 'wb')
        pickle.dump(domain, file_pickling, pickle.HIGHEST_PROTOCOL)
        file_pickling.close()

        file_pickling = open(name_file_base + '_Run.pickle', 'wb')
        pickle.dump(run, file_pickling, pickle.HIGHEST_PROTOCOL)
        file_pickling.close()



    # Return topology and data
    return domain, run

# end def postprocess(name_file_output_exodus, name_file_output):




#
# If run as 'python -m LCM_postprocess <filename>'
#
if __name__ == '__main__':

    try:

        name_file_output_exodus = sys.argv[1]

    except IndexError:

        raise IndexError('Name of input file required')

    if os.path.exists(name_file_output_exodus) == False:

        raise IOError('File does not exist')


    if len(sys.argv) == 3:

        postprocess(name_file_output_exodus, plotting = sys.argv[2])

    else:

        postprocess(name_file_output_exodus, plotting = True)
