#!/usr/bin/python

import cPickle as pickle
import matplotlib.pyplot as plt
# from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from operator import itemgetter

def set_num_ticks(axis, num_xticks = 5, num_yticks = 5, integer = (False, False)):
    xticks = [x for x in np.linspace(axis.get_xlim()[0], axis.get_xlim()[1], num_xticks)]
    yticks = [y for y in np.linspace(axis.get_ylim()[0], axis.get_ylim()[1], num_yticks)]
    if integer[0] is True:
        xticks = [np.floor(x) for x in xticks]
    if integer[1] is True:
        yticks = [np.floor(y) for y in yticks]
    axis.set_xticks(xticks)
    axis.set_yticks(yticks)

# Plot the time stepping and convergence data
# @profile
def plot_data_run(run = None, filename = None):

    if filename != None:
        run = pickle.load(open(filename, 'rb'))

    steps = sorted(run.steps.values(), key = lambda step: step.step_number)

    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=22)
    # rcParams['text.latex.preamble'] = [r'\usepackage{boldtensors}']

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    #
    # Plot the number of nonlinear iterations vs step #
    #
    step_numbers = [step.step_number for step in steps if step.status_convergence == 1]
    num_iters_nonlinear = [step.num_iters_nonlinear for step in steps if step.status_convergence == 1]

    ax.bar(
        step_numbers,
        num_iters_nonlinear)

    ax.set_xlabel('Continuation Step')
    ax.set_ylabel('Nonlinear Iterations')

    set_num_ticks(ax, integer = (True, True))

    canvas.print_figure('nonlinear_iterations.pdf', bbox_inches = 'tight')


    #
    # Plot the step size vs iteration #
    #
    ax.clear()

    sizes_step = [step.size_step for step in steps if step.status_convergence == 1]

    p1 = ax.plot(
        step_numbers[1:],
        sizes_step[1:],
        color = 'blue',
        marker = '.')

    ax.hold(True)

    step_numbers = [step.step_number for step in steps if step.status_convergence == -1]
    sizes_step = [step.size_step for step in steps if step.status_convergence == -1]

    if len(step_numbers) > 0:

        p2 = ax.plot(
            step_numbers[1:],
            sizes_step[1:],
            color='red',
            linestyle = 'none',
            marker = 'o')

        legend = ax.legend(
            ['Converged', 'Failed'],
            bbox_to_anchor = (1.05, 1), 
            loc = 2, 
            borderaxespad = 0.,
            fontsize = 15)

        extra_artists = [legend]

    else:

        extra_artists = []

    ax.set_yscale('log')
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('Step Size (s)')

    canvas.print_figure(
        'step_size.pdf',
        bbox_extra_artists = extra_artists,
        bbox_inches = 'tight')


    #
    # Plot the increment norm vs iteration #
    #
    ax.clear()
    ax.hold(True)
    
    for step in steps:

        if step.status_convergence == 1:

            points_plot = range(step.num_iters_nonlinear + 1)
            values_plot = [step.iters_nonlinear[x].norm_increment for x in sorted(step.iters_nonlinear)]

            ax.plot(
                points_plot[1:],
                values_plot[1:])

    try:
        ax.set_yscale('log')
    except ValueError:
        ax.set_yscale('linear')
        pass

    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Increment Norm $\left\| \Delta u^{(n)} \right\|$')

    string_legend = ['Step '+str(i+1) for i in range(run.num_steps)]

    legend = ax.legend(
        string_legend,
        bbox_to_anchor = (1.05, 1), 
        loc = 2, 
        borderaxespad = 0.,
        fontsize = 15)

    canvas.print_figure(
        'norm_increment.pdf',
        bbox_extra_artists = [legend],
        bbox_inches = 'tight')


    #
    # Plot the residual norm vs iteration #
    #
    ax.clear()
    ax.hold(True)

    for step in steps:

        if step.status_convergence == 1:

            points_plot = range(step.num_iters_nonlinear + 1)
            values_plot = [step.iters_nonlinear[x].norm_residual for x in step.iters_nonlinear]

            ax.plot(
                points_plot[1:],
                values_plot[1:])

    try:
        ax.set_yscale('log')
    except ValueError:
        ax.set_yscale('linear')
        pass
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Residual Norm $\left\| F^{(n)} \right\|$')

    legend = ax.legend(
        string_legend,
        bbox_to_anchor = (1.05, 1), 
        loc = 2, 
        borderaxespad = 0.,
        fontsize = 15)

    canvas.print_figure(
        'norm_residual.pdf',
        bbox_extra_artists = [legend],
        bbox_inches = 'tight')


    #
    # Plot the increment norm convergence
    #
    ax.clear()
    ax.hold(True)
    
    for step in steps:

        if step.status_convergence == 1:

            values_plot = [step.iters_nonlinear[x].norm_increment for x in step.iters_nonlinear]

            ax.plot(
                values_plot[1:-1],
                values_plot[2:])

    try:
        ax.set_xscale('log')
        ax.set_yscale('log')
    except:
        ax.set_xscale('linear')
        ax.set_yscale('linear')
    ax.set_xlabel(r'Increment Norm $\left\| \Delta u^{(n)} \right\|$')
    ax.set_ylabel(r'Increment Norm $\left\| \Delta u^{(n+1)} \right\|$')

    legend = ax.legend(
        string_legend,
        bbox_to_anchor = (1.05, 1), 
        loc = 2, 
        borderaxespad = 0.,
        fontsize = 15)

    canvas.print_figure(
        'norm_increment_convergence.pdf',
        bbox_extra_artists = [legend],
        bbox_inches = 'tight')


    #
    # Plot the residual norm convergence
    #
    ax.clear()
    ax.hold(True)
    
    for step in steps:

        if step.status_convergence == 1:

            values_plot = [step.iters_nonlinear[x].norm_residual for x in step.iters_nonlinear]

            ax.plot(
                values_plot[1:-1],
                values_plot[2:])

    try:
        ax.set_xscale('log')
        ax.set_yscale('log')
    except:
        ax.set_xscale('linear')
        ax.set_yscale('linear')
    ax.set_ylabel(r'Residual Norm $\left\| F^{(n)} \right\|$')
    ax.set_ylabel(r'Residual Norm $\left\| F^{(n+1)} \right\|$')

    legend = ax.legend(
        string_legend,
        bbox_to_anchor = (1.05, 1), 
        loc = 2, 
        borderaxespad = 0.,
        fontsize = 15)

    canvas.print_figure(
        'norm_residual_convergence.pdf',
        bbox_extra_artists = [legend],
        bbox_inches = 'tight')

# end def plot_data_run(run = None, filename = None):



if __name__ == '__main__':

    import sys

    try:
        name_file_input = sys.argv[1]
    except:
        raise

    plot_data_run(filename = name_file_input)

# end if __name__ == '__main__':
