#!/usr/bin/python

from ._core import InputError
import cPickle as pickle
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator
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



def extract_steps(**kwargs):

    filename = kwargs.get('filename')
    run = kwargs.get('run')
    steps = kwargs.get('steps')

    if steps is None and run is None and filename is None:
        raise InputError('filename or run or steps must be specified')

    if filename != None:
        run = pickle.load(open(filename, 'rb'))

    if run != None:
        steps = sorted(run.steps.values(), key = lambda step: step.step_number)

    return steps



def setup_plot(**kwargs):

    # rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 22
    # rcParams['axes.titlesize'] = 18
    # rcParams['text.latex.preamble'] = [r'\usepackage{boldtensors}']



# Plot the time stepping and convergence data
# @profile
def plot_data_run(steps = None, title = None, **kwargs):

    steps = extract_steps(**kwargs)
    setup_plot(**kwargs)

    plot_nonlinear_iterations(steps = steps, title = title)
    plot_step_size(steps = steps, title = title)
    plot_norm_increment(steps = steps, title = title)
    plot_norm_residual(steps = steps, title = title)
    plot_convergence_increment(steps = steps, title = title)
    plot_convergence_residual(steps = steps, title = title)

# end def plot_data_run(run = None, filename = None):



#
# Plot the number of nonlinear iterations vs step #
#
def plot_nonlinear_iterations(**kwargs):

    steps = extract_steps(**kwargs)
    
    fig = Figure()
    ax = fig.add_subplot(111)
    title = fig.suptitle(kwargs.get('title'), fontsize = 14, fontweight = 'bold')
    canvas = FigureCanvas(fig)

    step_numbers = [
        step.step_number for step in steps if step.status_convergence == 1]

    num_iters_nonlinear = [
        step.num_iters_nonlinear for step in steps if step.status_convergence == 1]

    ax.bar(
        step_numbers,
        num_iters_nonlinear)

    ax.set_xlabel('Continuation Step')
    ax.set_ylabel('Nonlinear Iterations')

    set_num_ticks(ax, integer = (True, True))

    canvas.print_figure(
        'nonlinear_iterations.pdf',
        bbox_extra_artists = [title],
        bbox_inches = 'tight')



#
# Plot the step size vs iteration #
#
def plot_step_size(**kwargs):

    steps = extract_steps(**kwargs)
    
    fig = Figure()
    ax = fig.add_subplot(111)
    title = fig.suptitle(kwargs.get('title'), fontsize = 14, fontweight = 'bold')
    canvas = FigureCanvas(fig)

    step_numbers = [
        step.step_number for step in steps if step.status_convergence == 1]

    if steps is None:
        if run is None:
            raise InputError('run or steps must be specified')
        steps = extract_steps(run = run)

    ax.hold(False)

    sizes_step = [
        step.size_step for step in steps if step.status_convergence == 1]

    p1 = ax.plot(
        step_numbers[1:],
        sizes_step[1:],
        color = 'blue',
        marker = '.')

    ax.hold(True)

    step_numbers = [
        step.step_number for step in steps if step.status_convergence == -1]

    sizes_step = [
        step.size_step for step in steps if step.status_convergence == -1]

    extra_artists = [title]

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

        extra_artists.append(legend)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

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
def plot_norm_increment(**kwargs):

    steps = extract_steps(**kwargs)
    
    fig = Figure()
    ax = fig.add_subplot(111)
    title = fig.suptitle(kwargs.get('title'), fontsize = 14, fontweight = 'bold')
    canvas = FigureCanvas(fig)

    ax.clear()
    fig.suptitle(title, fontsize=14, fontweight='bold')
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

    string_legend = [str(i + 1) for i in range(len(steps))]

    legend = ax.legend(
        string_legend,
        bbox_to_anchor = (1.05, 1), 
        loc = 2, 
        borderaxespad = 0.,
        fontsize = 15,
        ncol = np.max([1, int(len(string_legend) / 15.)]),
        title = 'Step')

    canvas.print_figure(
        'norm_increment.pdf',
        bbox_extra_artists = [title, legend],
        bbox_inches = 'tight')



#
# Plot the residual norm vs iteration #
#
def plot_norm_residual(**kwargs):

    steps = extract_steps(**kwargs)
    
    fig = Figure()
    ax = fig.add_subplot(111)
    title = fig.suptitle(kwargs.get('title'), fontsize = 14, fontweight = 'bold')
    canvas = FigureCanvas(fig)

    ax.clear()
    fig.suptitle(title, fontsize=14, fontweight='bold')
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

    string_legend = [str(i + 1) for i in range(len(steps))]

    legend = ax.legend(
        string_legend,
        bbox_to_anchor = (1.05, 1), 
        loc = 2, 
        borderaxespad = 0.,
        fontsize = 15,
        ncol = np.max([1, int(len(string_legend) / 15.)]),
        title = 'Step')

    canvas.print_figure(
        'norm_residual.pdf',
        bbox_extra_artists = [title, legend],
        bbox_inches = 'tight')



#
# Plot the increment norm convergence
#
def plot_convergence_increment(**kwargs):

    steps = extract_steps(**kwargs)
    
    fig = Figure()
    ax = fig.add_subplot(111)
    title = fig.suptitle(kwargs.get('title'), fontsize = 14, fontweight = 'bold')
    canvas = FigureCanvas(fig)

    ax.clear()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    ax.hold(True)
    
    for step in steps:

        if step.status_convergence == 1:

            values_plot = [
                step.iters_nonlinear[x].norm_increment for x in step.iters_nonlinear]

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

    string_legend = [str(i + 1) for i in range(len(steps))]

    legend = ax.legend(
        string_legend,
        bbox_to_anchor = (1.05, 1), 
        loc = 2, 
        borderaxespad = 0.,
        fontsize = 15,
        ncol = np.max([1, int(len(string_legend) / 15.)]),
        title = 'Step')

    canvas.print_figure(
        'norm_increment_convergence.pdf',
        bbox_extra_artists = [title, legend],
        bbox_inches = 'tight')



#
# Plot the residual norm convergence
#
def plot_convergence_residual(**kwargs):

    steps = extract_steps(**kwargs)
    
    fig = Figure()
    ax = fig.add_subplot(111)
    title = fig.suptitle(kwargs.get('title'), fontsize = 14, fontweight = 'bold')
    canvas = FigureCanvas(fig)

    ax.clear()
    fig.suptitle(title, fontsize=14, fontweight='bold')
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

    string_legend = [str(i + 1) for i in range(len(steps))]

    legend = ax.legend(
        string_legend,
        bbox_to_anchor = (1.05, 1), 
        loc = 2, 
        borderaxespad = 0.,
        fontsize = 15,
        ncol = np.max([1, int(len(string_legend) / 15.)]),
        title = 'Step')

    canvas.print_figure(
        'norm_residual_convergence.pdf',
        bbox_extra_artists = [title, legend],
        bbox_inches = 'tight')



if __name__ == '__main__':

    import sys

    try:
        name_file_input = sys.argv[1]
    except:
        raise

    try:
        title = sys.argv[2]
    except:
        title = ''.join(name_file_input.split('/')[-1].split('.')[:-1])
        pass

    try:
        function = locals()[sys.argv[3]]
    except:
        function = plot_data_run
        pass

    function(filename = name_file_input, title = title)

# end if __name__ == '__main__':
