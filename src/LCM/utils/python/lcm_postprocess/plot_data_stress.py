#!/usr/bin/python

import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import rcParams
import numpy as np


# Plot ij-components of Cauchy stress vs ij-components of Logarithmic strain 
# @profile
def plot_data_stress(domain = None, filename = None, truncate_legend = False):

    if filename != None:
        domain = pickle.load(open(filename, 'rb'))

    num_dims = domain.num_dims
    times = domain.times
    num_blocks = len(domain.blocks)

#    rc_params.update({'figure.autolayout': True})

    string_legend = ['Domain']
    if truncate_legend is True:
        num_blocks_plot = 13
    else:
        num_blocks_plot = num_blocks    

    string_legend.extend(['Block ' + str(key_block) for key_block in sorted(list(domain.blocks))[:num_blocks_plot]])
    if truncate_legend is True:
        string_legend.extend(['...', 'Block ' + str(sorted(list(domain.blocks))[-1])])

    # plt.rc('text', usetex=True)
    plt.rc('font', family = 'serif', size = 22)

    fig = plt.figure()

    for dim_i in range(num_dims):

        for dim_j in range(num_dims):

            fig.clf()
            ax = fig.add_subplot(111)
            ax.hold(True)

            ax.plot(
                [int(domain.variables['Log_Strain'][key_step][(dim_i, dim_j)]*1e8)/1e8 for key_step in times],
                [int(domain.variables['Cauchy_Stress'][key_step][(dim_i, dim_j)]*1e8)/1e8 for key_step in times],
                marker = 'o')

            for key_block in domain.blocks:

                block = domain.blocks[key_block]

                ax.plot(
                    [int(block.variables['Log_Strain'][key_step][(dim_i, dim_j)]*1e8)/1e8 for key_step in times],
                    [int(block.variables['Cauchy_Stress'][key_step][(dim_i, dim_j)]*1e8)/1e8 for key_step in times],
                    linestyle = ':')

            plt.xlabel('Logarithmic Strain $\epsilon_{'+ str(dim_i + 1) + str(dim_j + 1) +'}$')
            plt.ylabel('Cauchy Stress $\sigma_{'+ str(dim_i + 1) + str(dim_j + 1) +'}$ (MPa)')

            # ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            plt.locator_params(axis='x',nbins=4)

            legend = plt.legend(
                string_legend,
                bbox_to_anchor = (1.05, 1), 
                loc = 2, 
                borderaxespad = 0.,
                fontsize = 15,
                ncol = np.max([1, int(len(string_legend) / 15.)]))

            plt.savefig(
                'stress_strain_'+ str(dim_i + 1) + str(dim_j + 1) +'.pdf',
                additional_artists = [legend],
                bbox_inches='tight')

    plt.close(fig)

# end def plot_data_stress(domain):



if __name__ == '__main__':

    import sys

    try:
        name_file_input = sys.argv[1]
    except:
        raise

    try:
        truncate_legend = sys.argv[2]
    except:
        truncate_legend = False
        pass

    plot_data_stress(
        filename = name_file_input,
        truncate_legend = truncate_legend)

# end if __name__ == '__main__':