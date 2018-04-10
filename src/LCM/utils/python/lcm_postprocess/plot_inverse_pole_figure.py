#!/usr/bin/python

import argparse
from matplotlib import rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# Plot inverse pole figures for simulations with defined local orientations
def plot_inverse_pole_figure(**kwargs):

    fmt = kwargs.get('fmt', 'pdf')
    axes = kwargs.get('axes', 'xyz')
    ids_axis = {'x':1, 'y':2, 'z':3}
    captions = kwargs.get('captions', True)

    #
    # Read data 
    #
    name_file_input = kwargs.get('name_file_input', None)

    if name_file_input != None:

        name_file_input = kwargs['name_file_input']
        name_file_base = name_file_input.split('.')[0]
        orientations = np.loadtxt(name_file_input)

    else:

        domain = kwargs.get('domain', None)

        if 'domain' != None:

            time = kwargs.get('time', None)

            name_file_base = str(time)

            list_orientations = []

            for key_block in domain.blocks:
                block = domain.blocks[key_block]
                for key_element in block.elements:
                    element = block.elements[key_element]
                    list_orientations.append(element.variables['orientation'][time].flatten())
            orientations = np.array(list_orientations)

        else:

            raise Type_error("Need either 'name_file_input' or 'domain' keyword args")

    #
    # Create figure axis arrays
    #
    num_pts = 100
    XX = np.zeros(num_pts + 2)
    YY = np.zeros(num_pts + 2)

    for x in range(1, num_pts + 1):

        HKL = np.array([x / float(num_pts), 1., 1.])
        HKL /= np.linalg.norm(HKL)        
        XX[x] = HKL[1] / (1. + HKL[2])
        YY[x] = HKL[0] / (1. + HKL[2])

    #
    # Compute IPF quantities
    #
    X = {}
    Y = {}
    for axis in axes:
        indices = [x + 3 * (ids_axis[axis] - 1) for x in range(3)]
        D = np.array([x / np.linalg.norm(x) for x in abs(orientations[:, indices])])

        #
        # Convert orientations to ipf space 
        #
        X[axis] = np.zeros(len(orientations))
        Y[axis] = np.zeros(len(orientations))
        for x in range(len(orientations)):
            A = sorted(D[x,:])    
            X[axis][x] = A[1] / (1. + A[2])
            Y[axis][x] = A[0] / (1. + A[2])
                                                                            
    #
    # Create figures  
    #
    # rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 22

    n_axes = len(axes)
    ids_axis = {'x':1, 'y':2, 'z':3}

    fig = Figure()
    fig.set_size_inches(5 * n_axes, 4)

    subplots = {x:int('1' + str(n_axes) + str(ids_axis[x])) for x in axes}

    ax = {}
    colors = {'x':'r', 'y':'b', 'z':'g'}
    for axis in axes:
        ax[axis] = fig.add_subplot(subplots[axis])
        ax[axis].plot(X[axis], Y[axis], colors[axis] + 'o')
        ax[axis].plot(XX, YY, 'k', linewidth = 1)
        ax[axis].set_aspect('equal', adjustable = 'box')
        ax[axis].axis('off')
        if captions == True:
            ax[axis].text(0.2, -0.05, 'IPF_' + axis.upper(), fontsize = 16)
        ax[axis].text(-0.03, -0.03, r'$[001]$', fontsize = 15)
        ax[axis].text(0.38, -0.03, r'$[011]$', fontsize = 15)
        ax[axis].text(0.34, 0.375, r'$[\bar111]$', fontsize = 15)

    extra_artists = []
    if captions == True:
        title = fig.suptitle('Inverse pole figures', fontsize = 14, fontweight = 'bold')
        extra_artists.append(title)
    canvas = FigureCanvas(fig)

    canvas.print_figure(
        name_file_base + '_IPF.' + fmt,
        bbox_extra_artists = extra_artists,
        bbox_inches = 'tight')

    return X,Y

# end plot_inverse_pole_figure(**kwargs):



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--name_file_input', default = None, help = 'Specify rotations file.')
    parser.add_argument('-f', '--format', default = 'pdf', help = 'Specify file format for figure')
    parser.add_argument('-a', '--axes', default = 'xyz', help = 'Specify IPF axes')
    parser.add_argument('-c', '--captions', default = True, help = 'Print captions [True/False]')

    args_dict = parser.parse_args().__dict__

    name_file_input = args_dict['name_file_input']
    # fmt = args.format
    # axes = args.axes

    name_file_base = name_file_input.split('.')[0]
    name_file_extension = name_file_input.split('.')[-1]

    if name_file_extension == 'pickle':
    	import cPickle as pickle
        file_pickling = open(name_file_input, 'rb')
        domain = pickle.load(file_pickling)
        file_pickling.close()
        args_dict['domain'] = domain
        for time in [0,-1]:
            args_dict['time'] = domain.times[time]
            plot_inverse_pole_figure(**args_dict)
    else:
    	X,Y = plot_inverse_pole_figure(**args_dict)

# end if __name__ == '__main__':
