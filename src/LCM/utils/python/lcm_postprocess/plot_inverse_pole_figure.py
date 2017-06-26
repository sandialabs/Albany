#!/usr/bin/python

from matplotlib import rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# Plot inverse pole figures for simulations with defined local orientations
def plot_inverse_pole_figure(**kwargs):

    fmt = kwargs.get('fmt', 'pdf')

    #
    # Read data 
    #
    if 'name_file_input' in kwargs:

        name_file_input = kwargs['name_file_input']
        name_file_base = name_file_input.split('.')[0]
        orientations = np.loadtxt(name_file_input)

    elif 'domain' in kwargs:

        domain = kwargs['domain']

        if 'time' in kwargs:
            time = kwargs['time']
        else:
            time = 0.0
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
    # Create axes 
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
    RD = np.array([x / np.linalg.norm(x) for x in abs(orientations[:, 0:3])])
    TD = np.array([x / np.linalg.norm(x) for x in abs(orientations[:, 3:6])])
    ND = np.array([x / np.linalg.norm(x) for x in abs(orientations[:, 6:9])])

    #
    # Convert orientations to ipf space 
    #
    X_RD = np.zeros(len(orientations))
    Y_RD = np.zeros(len(orientations))
    for x in range(len(orientations)):
        A = sorted(RD[x,:])    
        X_RD[x] = A[1] / (1. + A[2])
        Y_RD[x] = A[0] / (1. + A[2])

    X_TD = np.zeros(len(orientations))
    Y_TD = np.zeros(len(orientations))
    for x in range(len(orientations)):
        A = sorted(TD[x,:])    
        X_TD[x] = A[1] / (1. + A[2])
        Y_TD[x] = A[0] / (1. + A[2])
        
    X_ND = np.zeros(len(orientations))
    Y_ND = np.zeros(len(orientations))
    for x in range(len(orientations)):
        A = sorted(ND[x,:])    
        X_ND[x] = A[1] / (1. + A[2])
        Y_ND[x] = A[0] / (1. + A[2])
                                                                            
    #
    # Create figures  
    #
    # rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 22

    fig = Figure()
    fig.set_size_inches(15,4)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.plot(X_RD, Y_RD, 'ro')
    ax1.plot(XX, YY, 'k', linewidth = 1)
    ax1.set_aspect('equal', adjustable = 'box')
    ax1.axis('off')
    ax1.text(0.2, -0.05, 'RD', fontsize = 16)
    ax1.text(-0.03, -0.03, r'$[001]$', fontsize = 15)
    ax1.text(0.38, -0.03, r'$[011]$', fontsize = 15)
    ax1.text(0.34, 0.375, r'$[\bar111]$', fontsize = 15)

    ax2.plot(X_TD, Y_TD, 'bo')
    ax2.plot(XX, YY, 'k', linewidth = 1)
    ax2.set_aspect('equal', adjustable = 'box')
    ax2.axis('off')
    ax2.text(0.2, -0.05, 'TD', fontsize = 16)
    ax2.text(-0.03, -0.03, r'$[001]$', fontsize = 15)
    ax2.text(0.38, -0.03, r'$[011]$', fontsize = 15)
    ax2.text(0.34, 0.375, r'$[\bar111]$', fontsize = 15)

    ax3.plot(X_ND, Y_ND, 'go')
    ax3.plot(XX, YY, 'k', linewidth = 1)
    ax3.set_aspect('equal', adjustable = 'box')
    ax3.axis('off')
    ax3.text(0.2, -0.05, 'ND', fontsize = 16)
    ax3.text(-0.03, -0.03, r'$[001]$', fontsize = 15)
    ax3.text(0.38, -0.03, r'$[011]$', fontsize = 15)
    ax3.text(0.34, 0.375, r'$[\bar111]$', fontsize = 15)

    title = fig.suptitle('Inverse pole figures', fontsize = 14, fontweight = 'bold')
    canvas = FigureCanvas(fig)

    canvas.print_figure(
        name_file_base + '_IPF.' + fmt,
        bbox_extra_artists = [title],
        bbox_inches = 'tight')

# end plot_inverse_pole_figure(**kwargs):



if __name__ == '__main__':

    import sys

    try:
        name_file_input = sys.argv[1]
    except:
        raise

    if 3 == len(sys.argv):
        fmt = sys.argv[2]
    else:
        fmt = 'pdf'

    name_file_base = name_file_input.split('.')[0]
    name_file_extension = name_file_input.split('.')[-1]

    if name_file_extension == 'pickle':
    	import cPickle as pickle
        file_pickling = open(name_file_input, 'rb')
        domain = pickle.load(file_pickling)
        file_pickling.close()
        plot_inverse_pole_figure(domain = domain, time = domain.times[0], fmt = fmt)
        plot_inverse_pole_figure(domain = domain, time = domain.times[-1], fmt = fmt)
    else:
    	plot_inverse_pole_figure(name_file_input = name_file_input, fmt = fmt)

# end if __name__ == '__main__':
