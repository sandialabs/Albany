#!/usr/bin/python
'''
LCM_postprocess.py input.e

creates usable output from LCM calculations
'''

#
# Imported modules
#
import cPickle as pickle
from _core import Timer
from read_file_log import read_file_log
from read_file_output_exodus import read_file_output_exodus
from read_file_input_material import read_file_input_material
from derive_value_variable import derive_value_variable
from write_file_exodus import write_file_exodus
from plot_data_run import plot_data_run
from plot_inverse_pole_figure import plot_inverse_pole_figure
from plot_data_stress import plot_data_stress
from write_file_data import write_file_data


#
# Main function to postprocess simulation data
#
def postprocess(
    name_file_output_exodus,
    plotting = False,
    pickling = True,
    write_data = True,
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
    # Plot the simulation results
    #
    if plotting == True:
    
        #
        # Plot the convergence data
        #
        with Timer() as timer:

            if verbosity > 0:
                print 'Plotting convergence data...'

            plot_data_run(run = run)

        if verbosity > 0:
            print '    Elapsed time: ' + str(timer.interval) + 's\n'

        #
        # Plot the inverse pole figures
        #
        with Timer() as timer:

            if verbosity > 0:
                print 'Plotting inverse pole figures...'
            
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

            plot_data_stress(domain = domain)

        if verbosity > 0:
            print '    Elapsed time: ' + str(timer.interval) + 's\n'



    #
    # Serialize data objects
    #
    if pickling == True:

        with Timer() as timer:

            if verbosity > 0:
                print 'Writing serialized object files...'

            file_pickling = open(name_file_base + '_Domain.pickle', 'wb')
            pickle.dump(domain, file_pickling, pickle.HIGHEST_PROTOCOL)
            file_pickling.close()

            file_pickling = open(name_file_base + '_Run.pickle', 'wb')
            pickle.dump(run, file_pickling, pickle.HIGHEST_PROTOCOL)
            file_pickling.close()

        if verbosity > 0:
            print '    Elapsed time: ' + str(timer.interval) + 's\n'


    #
    # Write data to text file
    #
    if write_data == True:

        with Timer() as timer:

            if verbosity > 0:
                print 'Writing data to text file...'

            write_file_data(domain = domain, name_file_output = name_file_base + '_Data.out')

        if verbosity > 0:
            print '    Elapsed time: ' + str(timer.interval) + 's\n'



    # Return topology and data
    return domain, run

# end def postprocess(name_file_output_exodus, name_file_output):




#
# If run as 'python -m LCM_postprocess <filename>'
#
if __name__ == '__main__':

    import os
    import sys

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
