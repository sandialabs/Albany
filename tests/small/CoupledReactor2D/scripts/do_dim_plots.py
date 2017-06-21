#!/usr/bin/env python

import sys
sys.path.append('..')

from albany_sweeps import *
from run_albany import parse_albany_args
from matplotlib.pyplot import *
from optparse import OptionParser

# Get command line arguments
parser = OptionParser()
parser.add_option("-s", "--save", dest="filename", default=None,
                  help="save figure to FILE", metavar="FILE")
options = parse_albany_args(parser)
options.mpirun = 'mpirun -np'
options.albany = '/home/etphipp/Albany/build/opt_mpi/src/'

options.dakota = False
options.sg = True
options.ni = False
options.mpni = False
options.input_det = 'input_det.xml.in'
options.input_sg = 'inputSG.xml.in'
options.force = False
options.tag = ''
n_range = [64]
M_range = range(1,8)
N_range_sg = [3]

all_stats_sg = dim_sweep(n_range, M_range, N_range_sg, options)

options.dakota = True
options.sg = False
options.input_dakota = 'input.xml.in'
options.force = False
N_range_col = [3]

all_stats_col = dim_sweep(n_range, M_range, N_range_col, options)

params = { 'font.size': 16,
           'font.weight': 'bold',
           'lines.linewidth': 2,
           'axes.linewidth': 2,
           'xtick.major.size': 8,
           'xtick.major.width': 2,
           'ytick.major.size': 8,
           'ytick.minor.size': 4
}
rcParams.update(params)

figure(1)
clf()
hold(True)
for i in range(len(N_range_sg)):
    N_sg = N_range_sg[i]
    N_col = N_range_col[i]
    subplot(len(N_range_sg), 1, i+1)
    for j in range(len(n_range)):
        n = n_range[j]
        semilogy(M_range, all_stats_sg[j][i].solve, '-*',
                 M_range, all_stats_col[j][i].solve, '-*')
        legend(('Galerkin', 'Non-intrusive'),loc='upper left')
        xlabel('Stochastic Dimension M')
        ylabel('Scaled Run Time')
        a = gca()
        # xticks = a.get_xticklines()
        # for tick in xticks:
        #     tick.linewidth = 20
hold(False)
#suptitle('Linear Problem')
draw()

if options.filename != None:
    savefig(options.filename)
