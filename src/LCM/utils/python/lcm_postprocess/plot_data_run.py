#!/usr/bin/python

import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
from operator import itemgetter

# Plot the time stepping and convergence data
def plot_data_run(run = None, filename = None):

    if filename != None:
        run = pickle.load(open(filename, 'rb'))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=22)
    rcParams['text.latex.preamble'] = [r'\usepackage{boldtensors}']

    fig = plt.figure()

    #
    # Plot the number of nonlinear iterations vs step #
    #
    step_numbers = [run.steps[x].step_number for x in run.steps if run.steps[x].status_convergence == 1]
    num_iters_nonlinear = [run.steps[x].num_iters_nonlinear for x in run.steps if run.steps[x].status_convergence == 1]

    plt.bar(
        step_numbers,
        num_iters_nonlinear)

    plt.savefig(
        'nonlinear_iterations.pdf',
        bbox_inches = 'tight')

    #
    # Plot the step size vs iteration #
    #
    fig.clf()

    step_numbers = [run.steps[x].step_number for x in run.steps if run.steps[x].status_convergence == 1]
    sizes_step = [run.steps[x].size_step for x in run.steps if run.steps[x].status_convergence == 1]
    data = sorted(zip(step_numbers, sizes_step), key = itemgetter(0))

    p1 = plt.plot(
        [x[0] for x in data[1:]],
        [x[1] for x in data[1:]],
        color = 'blue',
        marker = '.')

    plt.hold(True)

    step_numbers = [run.steps[x].step_number for x in run.steps if run.steps[x].status_convergence == -1]
    sizes_step = [run.steps[x].size_step for x in run.steps if run.steps[x].status_convergence == -1]
    data = sorted(zip(step_numbers, sizes_step), key = itemgetter(0))

    if len(data) > 0:

        p2 = plt.plot(
            [x[0] for x in data],
            [x[1] for x in data],
            color='red',
            linestyle = 'none',
            marker = 'o')

        legend = plt.legend(
            ['Converged', 'Failed'],
            bbox_to_anchor = (1.05, 1), 
            loc = 2, 
            borderaxespad = 0.,
            fontsize = 15)

        additional_artists = [legend]

    else:

        additional_artists = []

    plt.yscale('log')
    plt.xlabel('Time Step')
    plt.ylabel('Step Size (s)')

    plt.savefig(
        'step_size.pdf',
        additional_artists = additional_artists,
        bbox_inches = 'tight')


    #
    # Plot the increment norm vs iteration #
    #
    fig.clf()
    plt.hold(True)
    
    for key_step in run.steps:

        if run.steps[key_step].status_convergence == 1:
        
            step = run.steps[key_step]

            points_plot = range(step.num_iters_nonlinear + 1)
            values_plot = [step.iters_nonlinear[x].norm_increment for x in step.iters_nonlinear]

            plt.plot(
                points_plot[1:],
                values_plot[1:])

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'Increment Norm $\left\| \Delta u^{(n)} \right\|$')

    string_legend = ['Step '+str(i+1) for i in range(run.num_steps)]

    legend = plt.legend(
        string_legend,
        bbox_to_anchor = (1.05, 1), 
        loc = 2, 
        borderaxespad = 0.,
        fontsize = 15)

    plt.savefig(
        'norm_increment.pdf',
        additional_artists = [legend],
        bbox_inches = 'tight')


    #
    # Plot the residual norm vs iteration #
    #
    fig.clf()
    plt.hold(True)

    for key_step in run.steps:

        if run.steps[key_step].status_convergence == 1:
        
            step = run.steps[key_step]

            points_plot = range(step.num_iters_nonlinear + 1)
            values_plot = [step.iters_nonlinear[x].norm_residual for x in step.iters_nonlinear]

            plt.plot(
                points_plot[1:],
                values_plot[1:])

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'Residual Norm $\left\| F^{(n)} \right\|$')

    legend = plt.legend(
        string_legend,
        bbox_to_anchor = (1.05, 1), 
        loc = 2, 
        borderaxespad = 0.,
        fontsize = 15)

    plt.savefig(
        'norm_residual.pdf',
        additional_artists = [legend],
        bbox_inches = 'tight')


    #
    # Plot the increment norm convergence
    #
    fig.clf()
    plt.hold(True)
    
    for key_step in run.steps:

        if run.steps[key_step].status_convergence == 1:
        
            step = run.steps[key_step]

            values_plot = [step.iters_nonlinear[x].norm_increment for x in step.iters_nonlinear]

            plt.plot(
                values_plot[1:-1],
                values_plot[2:])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Increment Norm $\left\| \Delta u^{(n)} \right\|$')
    plt.ylabel(r'Increment Norm $\left\| \Delta u^{(n+1)} \right\|$')

    legend = plt.legend(
        string_legend,
        bbox_to_anchor = (1.05, 1), 
        loc = 2, 
        borderaxespad = 0.,
        fontsize = 15)

    plt.savefig(
        'norm_increment_convergence.pdf',
        additional_artists = [legend],
        bbox_inches = 'tight')


    #
    # Plot the residual norm convergence
    #
    fig.clf()
    plt.hold(True)
    
    for key_step in run.steps:

        if run.steps[key_step].status_convergence == 1:
        
            step = run.steps[key_step]

            values_plot = [step.iters_nonlinear[x].norm_residual for x in step.iters_nonlinear]

            plt.plot(
                values_plot[1:-1],
                values_plot[2:])

    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'Residual Norm $\left\| F^{(n)} \right\|$')
    plt.ylabel(r'Residual Norm $\left\| F^{(n+1)} \right\|$')

    legend = plt.legend(
        string_legend,
        bbox_to_anchor = (1.05, 1), 
        loc = 2, 
        borderaxespad = 0.,
        fontsize = 15)

    plt.savefig(
        'norm_residual_convergence.pdf',
        additional_artists = [legend],
        bbox_inches = 'tight')

    plt.close(fig)

# end def plot_data_run(run = None, filename = None):



if __name__ == '__main__':

    import sys

    try:
        name_file_input = sys.argv[1]
    except:
        raise

    plot_data_run(filename = name_file_input)

# end if __name__ == '__main__':
