#!/usr/bin/python
'''
LCM_postprocess.py input.e

creates usable output from LCM calculations
'''

#
# Imported modules
#
import argparse
import cPickle as pickle
from lcm_postprocess._core import Timer
from lcm_postprocess.read_file_log import read_file_log
from lcm_postprocess.read_file_output_exodus import read_file_output_exodus
from lcm_postprocess.read_file_input_material import read_file_input_material
from lcm_postprocess.derive_value_variable import derive_value_variable
from lcm_postprocess.write_file_exodus import write_file_exodus
from lcm_postprocess.plot_data_run import plot_data_run
from lcm_postprocess.plot_inverse_pole_figure import plot_inverse_pole_figure
from lcm_postprocess.plot_data_stress import plot_data_stress
from lcm_postprocess.write_file_data import write_file_data
import os
import sys


#
# Main function to postprocess simulation data
#
# @profile
def postprocess(
    name_file_output_exodus,
    compute_run = True,
    plotting = False,
    pickling = True,
    write_data = True,
    verbosity = 1,
    fmt = 'pdf',
    **kwargs):

    if verbosity > 0:
        print ''


    domain = None
    run = None

    #
    # Set i/o units 
    #
    name_file_base = name_file_output_exodus.split('.')[0]
    name_file_extension = name_file_output_exodus.split('.')[-1]




    with Timer() as timer:

        #
        # Create the simulation time step structure
        #
        if compute_run == True:
            exist_file_log = True
            if verbosity > 0:
                print 'Creating simulation time step object...'

            try:

                run = read_file_log(name_file_base + '_Log.out')

                if verbosity > 0:
                    timer.print_time()

            except IOError:

                exist_file_log = False
                print '    **No log file found.**\n'




        #
        # Get values of whole domain variables
        #
        if verbosity > 0:
            print 'Reading exodus output file...'

        domain = read_file_output_exodus(filename = name_file_output_exodus)

        if verbosity > 0:
            timer.print_time()





        #
        # Get material data
        #
        if verbosity > 0:
            print 'Retrieving material data...'

        if os.path.isfile(name_file_base + '_Material.xml'):
            name_file_material = name_file_base + '_Material.xml'
        elif os.path.isfile(name_file_base + '_Material.yaml'):
            name_file_material = name_file_base + '_Material.yaml'
        else:
            raise Exception('Material file not found.')

        read_file_input_material(
            name_file = name_file_material, 
            domain = domain,
            names_variable = ['orientations'])

        if verbosity > 0:
            timer.print_time()



    
        #
        # Compute derived variable values
        #
        if verbosity > 0:
            print 'Deriving output data...'
    
        derive_value_variable(domain)

        if verbosity > 0:
            timer.print_time()



        #
        # Write data to exodus output file
        #
        if verbosity > 0:
            print 'Writing data to exodus file...'

        if 'name_file_output' in kwargs:
            name_file_postprocess = kwargs['name_file_output']
        else:
            name_file_postprocess = name_file_base + '_Postprocess.' + name_file_extension
        
        write_file_exodus(domain, name_file_output_exodus, name_file_postprocess)

        if verbosity > 0:
            timer.print_time()




        #
        # Plot the simulation results
        #
        if plotting == True:
        
            #
            # Plot the convergence data
            #
            if run is not None:

                if verbosity > 0:
                    print 'Plotting convergence data...'

                plot_data_run(run = run)

                if verbosity > 0:
                    timer.print_time()

            else:

                print '**Not plotting run data.**\n'

            #
            # Plot the inverse pole figures
            #
            if verbosity > 0:
                print 'Plotting inverse pole figures...'
            
            for step in [domain.times[0], domain.times[-1]]:

                plot_inverse_pole_figure(domain = domain, time = step, fmt = fmt)

            if verbosity > 0:
                timer.print_time()

            #
            # Plot stress-strain data
            #
            if verbosity > 0:
                print 'Plotting stress-strain data...'

            plot_data_stress(domain = domain, truncate_legend = True)

            if verbosity > 0:
                timer.print_time()



        #
        # Serialize data objects
        #
        if pickling == True:

            if verbosity > 0:
                print 'Writing serialized object files...'

            file_pickling = open(name_file_base + '_Domain.pickle', 'wb')
            pickle.dump(domain, file_pickling, pickle.HIGHEST_PROTOCOL)
            file_pickling.close()

            if run is not None:

                file_pickling = open(name_file_base + '_Run.pickle', 'wb')
                pickle.dump(run, file_pickling, pickle.HIGHEST_PROTOCOL)
                file_pickling.close()

            if verbosity > 0:
                timer.print_time()


        #
        # Write data to text file
        #
        if write_data == True:

            if verbosity > 0:
                print 'Writing data to text file...'

            write_file_data(domain = domain, name_file_output = name_file_base + '_Data.out')

            if verbosity > 0:
                timer.print_time()



    # Return topology and data
    return domain, run

# end def postprocess(name_file_output_exodus, name_file_output):




#
# If run as 'python -m LCM_postprocess <filename>'
#
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-e',
        '--exodus_file',
        default = None,
        required = True,
        help = 'Name of Exodus file.')

    parser.add_argument(
        '--plot',
        default = True,
        action = 'store_true',
        help = 'Plot results.')

    parser.add_argument(
        '--no-plot',
        dest = 'plot',
        action = 'store_false',
        help = 'Plot results.')

    parser.add_argument(
        '-f',
        '--format',
        default = 'pdf',
        help = 'Specify image file format.')

    parser.add_argument(
        '--compute_run',
        default = True,
        dest = 'compute_run',
        action = 'store_true',
        help = 'Image file format.')

    parser.add_argument(
        '--no-compute_run',
        dest = 'compute_run',
        action = 'store_false',
        help = 'Image file format.')

    args_dict = parser.parse_args().__dict__

    print args_dict['exodus_file']

    if os.path.exists(args_dict['exodus_file']) == False:

        raise IOError('File does not exist')

    postprocess(
        name_file_output_exodus = args_dict['exodus_file'],
        compute_run = args_dict['compute_run'],
        plotting = args_dict['plot'],
        fmt = args_dict['format'])
