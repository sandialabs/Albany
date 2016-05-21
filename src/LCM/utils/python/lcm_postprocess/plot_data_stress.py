#!/usr/bin/python

import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams


# Plot ij-components of Cauchy stress vs ij-components of Logarithmic strain 
def plot_data_stress(domain = None, filename = None):

    if filename != None:
        domain = pickle.load(open(filename, 'rb'))

    num_dims = domain.num_dims
    times = domain.times

#    rc_params.update({'figure.autolayout': True})

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=22)

    fig = plt.figure()

    for dim_i in range(num_dims):

        for dim_j in range(num_dims):

            fig.clf()
            plt.hold(True)

            plt.plot(
                [domain.variables['Log_Strain'][key_step][(dim_i, dim_j)] for key_step in times],
                [domain.variables['Cauchy_Stress'][key_step][(dim_i, dim_j)] for key_step in times],
                marker = 'o')

            str_legend = ['Domain']

            for key_block in domain.blocks:

                block = domain.blocks[key_block]

                plt.plot(
                    [block.variables['Log_Strain'][key_step][(dim_i, dim_j)] for key_step in times],
                    [block.variables['Cauchy_Stress'][key_step][(dim_i, dim_j)] for key_step in times],
                    linestyle = ':')

                str_legend.append('Block ' + str(key_block))


            plt.xlabel('Logarithmic Strain $\epsilon_{'+ str(dim_i + 1) + str(dim_j + 1) +'}$')
            plt.ylabel('Cauchy Stress $\sigma_{'+ str(dim_i + 1) + str(dim_j + 1) +'}$ (MPa)')

            legend = plt.legend(
                str_legend,
                bbox_to_anchor = (1.05, 1), 
                loc = 2, 
                borderaxespad = 0.,
                fontsize = 15)

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

    plot_data_stress(filename = name_file_input)

# end if __name__ == '__main__':