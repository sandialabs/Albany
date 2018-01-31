#!/usr/bin/python

import argparse
import cPickle as pickle
from cycler import cycler
import matplotlib.ticker as mtick
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np


# Plot ij-components of Cauchy stress vs ij-components of Logarithmic strain 
# @profile
def plot_data_stress(
    domain = None,
    filename = None,
    truncate_legend = False,
    name_domain = 'Domain',
    names_block = None,
    monochrome = False,
    **kwargs):

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

    if type(names_block) is list:
        if len(names_block) != num_blocks:
            raise Exception('Wrong number of block names')
    else:
        if names_block == None:
            name_base = 'Block'
        elif type(names_block) is str:
            name_base = names_block
        names_block = [name_base + ' ' + str(key_block) for key_block in sorted(list(domain.blocks))]
    
    if truncate_legend ==True:
        string_legend.extend(names_block[:num_blocks_plot] + ['...'] + [names_block[-1]])
    else:
        string_legend.extend(names_block)

    # rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 22
    if monochrome == True:
        rcParams['axes.prop_cycle'] = cycler(color=['0.75'])

    fig = Figure()
    canvas = FigureCanvas(fig)

    alpha_block = kwargs.get('alpha_block', 0.2)
    linewidth_block = kwargs.get('linewidth_block', 0.5)
    linewidth_domain = kwargs.get('linewidth_domain', 4)
    color_domain = kwargs.get('color_domain', 'k')

    for dim_i in range(num_dims):

        for dim_j in range(num_dims):

            fig.clf()
            ax = fig.add_subplot(111)
            ax.hold(True)

            handles = []

            for key_block in domain.blocks:

                block = domain.blocks[key_block]

                handles.append(
                    ax.plot(
                        [int(block.variables['Log_Strain'][key_step][(dim_i, dim_j)]*1e8)/1e8 for key_step in times],
                        [int(block.variables['Cauchy_Stress'][key_step][(dim_i, dim_j)]*1e8)/1e8 for key_step in times],
                        linewidth = linewidth_block,
                        alpha = alpha_block)[0])

            handles.insert(0,
                ax.plot(
                    [int(domain.variables['Log_Strain'][key_step][(dim_i, dim_j)]*1e8)/1e8 for key_step in times],
                    [int(domain.variables['Cauchy_Stress'][key_step][(dim_i, dim_j)]*1e8)/1e8 for key_step in times],
                    linewidth = linewidth_domain,
                    color = color_domain)[0])

            ax.set_xlabel('Logarithmic Strain $\epsilon_{'+ str(dim_i + 1) + str(dim_j + 1) +'}$')
            ax.set_ylabel('Cauchy Stress $\sigma_{'+ str(dim_i + 1) + str(dim_j + 1) +'}$ (MPa)')

            ax.set_xlim([kwargs.get('x_min',ax.get_xlim()[0]), kwargs.get('x_max',ax.get_xlim()[1])])
            ax.set_ylim([kwargs.get('y_min',ax.get_ylim()[0]), kwargs.get('y_max',ax.get_ylim()[1])])

            # ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            ax.locator_params(axis = 'x', nbins = kwargs.get('nbins_x', 4))
            ax.locator_params(axis = 'y', nbins = kwargs.get('nbins_y', 5))

            if kwargs.get('print_legend', True):
                legend = ax.legend(
                    handles,
                    string_legend,
                    bbox_to_anchor = (1.05, 1), 
                    loc = 2, 
                    borderaxespad = 0.,
                    fontsize = 15)#,
                    # ncol = np.max([1, int(len(string_legend) / 15.)]))

                canvas.print_figure(
                    'stress_strain_'+ str(dim_i + 1) + str(dim_j + 1) +'.pdf',
                    bbox_extra_artists = [legend],
                    bbox_inches = 'tight')
            else:

                canvas.print_figure(
                    'stress_strain_'+ str(dim_i + 1) + str(dim_j + 1) +'.pdf',
                    bbox_inches = 'tight')

# end def plot_data_stress(domain):



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--name_file_input',
        default = None,
        required = True,
        help = 'Specify domain file.')

    parser.add_argument(
        '-n', '--name_base', default = 'Block', help = 'Specify legend entry base string.')

    parser.add_argument(
        '-t', 
        '--truncate_legend',
        default = False,
        action = 'store_true',
        help = 'Specify file format for figure')

    parser.add_argument(
        '-m', 
        '--monochrome',
        default = False,
        action = 'store_true',
        help = 'Plot block results in gray')

    args_dict = parser.parse_args().__dict__

    plot_data_stress(
        filename = args_dict['name_file_input'],
        truncate_legend = args_dict['truncate_legend'],
        names_block = args_dict['name_base'],
        monochrome = args_dict['monochrome'])

# end if __name__ == '__main__':
