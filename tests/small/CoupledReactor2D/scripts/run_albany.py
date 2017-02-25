#!/usr/bin/env python

import re
import os
from optparse import OptionParser
from albany_stats import *

def parse_albany_args(parser = OptionParser()):
    parser.add_option('--sg', action="store_true", 
                      dest="sg", default=False,
                      help='Run AlbanySG')
    parser.add_option('--ni', action="store_true", 
                      dest="ni", default=False,
                      help='Use NI solver method')
    parser.add_option('--mpni', action="store_true", 
                      dest="mpni", default=False,
                      help='Use MPNI solver method')
    parser.add_option('--sg_kl', action="store_true", 
                      dest="sg_kl", default=False,
                      help='Use KL SG solver method')
    parser.add_option('--dakota', action="store_true", 
                      dest="dakota", default=False,
                      help='Run AlbanyDakota')
    parser.add_option('--num_proc', default=1, 
                      help='Number of processors')
    parser.add_option('--num_spatial_proc', default=-1, 
                      help='Number of spatial processors')
    parser.add_option('--num_mesh', default=32,
                      help='Number of elements in each direction')
    parser.add_option('--num_kl', default=5,
                      help='Number of KL terms')
    parser.add_option('--num_jac_kl', default=6,
                      help='Number of KL terms for reduced Jacobian method')
    parser.add_option('--dim', default=3,
                      help='Stochastic dimension (>= --num_kl)')
    parser.add_option('--order', default=4,
                      help='Polynomial order')
    parser.add_option('--input', default='input.xml.in',
                      help='Albany input filename')
    parser.add_option('--input_heat', default='input_heat.xml.in',
                      help='Albany heat input filename')
    parser.add_option('--input_neut', default='input_neut.xml.in',
                      help='Albany neutronics input filename')
    parser.add_option('--input_ig', default=None,
                      help='Initial guess input filename for SG')
    parser.add_option('--dakota_input', default='dakota.in.in',
                      help='Dakota input filename')
    parser.add_option('--output', default='albany.out',
                      help='Output filename')
    parser.add_option('--dakota_output', default='dakota.out',
                      help='Dakota output filename')
    parser.add_option('--albany', default='./', 
                      help='Albany path')
    parser.add_option('--mpirun', default='mpirun -np', 
                      help='How to run mpirun')
    parser.add_option('-f', '--force', action="store_true", 
                      dest="force", default=False,
                      help='Force run even if output file exists')
    (options, args) = parser.parse_args()

    return options

def get_input_file_name(input_fname):
    return input_fname[:input_fname.rfind('.in')]

def write_input_file(input_fname, output_fname, echo_file, options):    
    in_file = open(input_fname, 'r')
    out_file = open(output_fname, 'w')
    
    lines = in_file.readlines()
    for line in lines:
        line = re.sub('@num_mesh@', str(options.num_mesh), line)
        line = re.sub('@num_kl@', str(options.num_kl), line)
        if options.sg:
            line = re.sub('@num_spatial_proc@', str(options.num_spatial_proc), 
                          line)
            line = re.sub('@dim@', str(options.dim), line)
            line = re.sub('@order@', str(options.order), line)
            line = re.sub('@num_jac_kl@', str(options.num_jac_kl), line)
        out_file.write(line)
        if options.run:
            echo_file.write(line)

def write_dakota_input_file(input_fname, output_fname, echo_file, options):    
    in_file = open(input_fname, 'r')
    out_file = open(output_fname, 'w')
    dim = int(options.dim)

    uuv_lower_bounds = '-1.0 '*dim
    uuv_upper_bounds = ' 1.0 '*dim
    uuv_descriptor = ' \'a\' '*dim
    nuv_means = '0.0 '*dim
    nuv_std_deviations = '1.0 '*dim
    nuv_lower_bounds = '-1.0 '*dim
    nuv_upper_bounds = ' 1.0 '*dim
    nuv_descriptor = ' \'a\' '*dim
    
    lines = in_file.readlines()
    for line in lines:
        line = re.sub('@dim@', str(options.dim), line)
        line = re.sub('@order@', str(options.order), line)
        line = re.sub('@uuv_lower_bounds@', uuv_lower_bounds, line)
        line = re.sub('@uuv_upper_bounds@', uuv_upper_bounds, line)
        line = re.sub('@uuv_descriptor@', uuv_descriptor, line)
        line = re.sub('@nuv_means@', nuv_means, line)
        line = re.sub('@nuv_std_deviations@', nuv_std_deviations, line)
        line = re.sub('@nuv_lower_bounds@', nuv_lower_bounds, line)
        line = re.sub('@nuv_upper_bounds@', nuv_upper_bounds, line)
        line = re.sub('@nuv_descriptor@', nuv_descriptor, line)
        line = re.sub('@num_spatial_proc@', str(options.num_spatial_proc), 
                      line)
        out_file.write(line)
        if options.run:
            echo_file.write(line)

def run_albany(options):

    # Determine if we should actually run anything or not
    options.run = False
    path = os.path.join(os.curdir, options.output)
    if not os.path.exists(path) or options.force:
        options.run = True

    # Open output file
    echo_file = None
    if options.run:
        echo_file = open(options.output, 'w')

    # Write input files
    input_ig_fname = ''
    input_fname = ''
    if options.sg and options.input_ig != None:
        input_ig_fname = get_input_file_name(options.input_ig)
        if options.run:
            echo_file.write('Initial guess input file:\n')
        write_input_file(options.input_ig, input_ig_fname, echo_file, options)

    input_fname = get_input_file_name(options.input)
    if options.run:
        echo_file.write('Input file:')
    write_input_file(options.input, input_fname, echo_file, options)

    input_fname_heat = get_input_file_name(options.input_heat)
    if options.run:
        echo_file.write('Heat input file:')
    write_input_file(options.input_heat, input_fname_heat, echo_file, options)

    input_fname_neut = get_input_file_name(options.input_neut)
    if options.run:
        echo_file.write('Neut Input file:')
    write_input_file(options.input_neut, input_fname_neut, echo_file, options)

    if options.dakota:
        dakota_input_fname = get_input_file_name(options.dakota_input)
        if options.run:
            echo_file.write('Dakota input file:')
        write_dakota_input_file(options.dakota_input, dakota_input_fname, 
                                echo_file, options)

    # Build up command line
    cmd = options.mpirun + ' ' + str(options.num_proc) + ' '
    cmd = cmd + options.albany + 'Albany'
    if options.sg:
        cmd = cmd + 'SG'
    elif options.dakota:
        cmd = cmd + 'Dakota'
    cmd = cmd + 'Coupled'
    cmd = cmd + ' '
    if options.input_ig != None:
        cmd = cmd + input_ig_fname + ' '
    cmd = cmd + input_fname + ' >> ' + options.output + ' 2>&1'

    # Print and run command
    if options.run:
        print cmd
        echo_file.write(cmd + '\n')
        echo_file.close()
        os.system(cmd)

    # Rename dakota output file
    if options.run and options.dakota and options.dakota_output != 'dakota.out':
        os.system('mv -f dakota.out ' + options.dakota_output)

    # Get stats
    if options.sg:
        if options.ni:
            stats = AlbanySGNIStats()
        elif options.mpni:
            stats = AlbanyMPNIStats()
        else:
            stats = AlbanySGStats()
    elif options.dakota:
        stats = AlbanyDakotaStats()
    else:
        stats = AlbanyStats()
    
    if options.dakota:
        stats.add_stats_from_file(options.output, options.dakota_output)
    else:
        stats.add_stats_from_file(options.output)

    print 'Stats for file:  ' + options.output
    print stats

    return stats

def main():

    # Parse options
    options = parse_albany_args()

    # Run Albany
    run_albany(options)

if __name__ == "__main__":
    main()

