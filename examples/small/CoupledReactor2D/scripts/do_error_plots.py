#!/usr/bin/env python

import sys
sys.path.append('..')

from albany_sweeps import *
from run_albany import parse_albany_args
from matplotlib.pyplot import *
from matplotlib.backends.backend_pdf import PdfPages
from optparse import OptionParser

# Get command line arguments
parser = OptionParser()
parser.add_option("-s", "--save", dest="filename", default=None,
                  help="save figure to FILE", metavar="FILE")
options = parse_albany_args(parser)
options.mpirun = 'mpirun -np'
options.albany = '/Users/etphipp/development/Albany/build/opt_mpi/src/'
options.num_proc = 1

options.dakota = False
options.sg = True
options.ni = True
options.mpni = False
options.input_det = 'input_network_coupled.xml.in'
options.input_heat_det = 'input_heat.xml.in'
options.input_neut_det = 'input_neutronics.xml.in'
n_range = [32]
M_range = [2, 4]
N_range = range(1,6)

options.tag = 'red_'
options.force = False
options.input_sg = 'input_network_coupled_sg_red.xml.in'
options.input_heat_sg = 'input_heat_sg.xml.in'
options.input_neut_sg = 'input_neutronics_sg.xml.in'
all_stats_red = error_sweep(n_range, M_range, N_range, options)

options.tag = 'ni_'
options.force = False
options.input_sg = 'input_network_coupled_sg_ni.xml.in'
options.input_heat_sg = 'input_heat_sg.xml.in'
options.input_neut_sg = 'input_neutronics_sg.xml.in'
all_stats_ni = error_sweep(n_range, M_range, N_range, options)

for i in range(0,len(n_range)):
  for j in range(0,len(M_range)):
    stats = all_stats_ni[i][j]
    l = len(stats.mean)
    all_stats_red[i][j].compute_errors(stats.mean[l-1], stats.var[l-1])

params = { 'font.size': 14,
           'font.weight': 'bold',
           'legend.fontsize' : 12, 
           'lines.linewidth': 2,
           'axes.linewidth': 2,
           'axes.fontweight': 'bold',
           'xtick.major.size': 8,
           'xtick.major.width': 2,
           'ytick.major.size': 8,
           'ytick.minor.size': 4
}
rcParams.update(params)

error_both_fig_1 = figure(1)
clf()
hold(True)
n = 32
n_idx = n_range.index(n)
var_names = [ 'Temp', 'Flux' ]
m = len(var_names)
idx = 1
for i in range(len(M_range)):
    M = M_range[i]
    for j in range(m):
        subplot(len(M_range), m, idx)
        l = len(all_stats_red[n_idx][i].mean)
        ve_red = []
        for k in range(l):
            ve_red.append(all_stats_red[n_idx][i].var_err[k][j])
        ve_ni = []
        for k in range(l-1):
            ve_ni.append(all_stats_ni[n_idx][i].var_err[k][j])
        loglog(all_stats_red[n_idx][i].total[0:l], ve_red, '-*',
               all_stats_ni[n_idx][i].total[0:l-1], ve_ni, '-*',)
        legend((var_names[j] + ' Reduced M = ' + str(M), 
                var_names[j] + ' Non-intrusive M = ' + str(M)))
        xlabel('scaled time', weight='bold')
        ylabel('variance error', weight='bold')
        idx = idx + 1
    #ylim(0.1, 1e3)
    #xlim(1e-6, 1)
hold(False)
#suptitle('Error')
draw()

if options.filename != None:
    pp = PdfPages(options.filename)
    pp.savefig(error_both_fig_1)
    pp.close()
