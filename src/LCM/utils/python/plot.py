from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot(data=None, **kwargs):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    if type(data) is np.ndarray:
        data_x = [datum[0] for datum in data]
        data_y = [datum[1] for datum in data]

    fig = plt.figure()
    plt.plot(data_x, data_y)

#    ax.plot(data_x, data_y)

#    ax.legend(['s_11'], loc = 'lower right')

    if 'xlabel' in kwargs:
        ax.set_xlabel(xlabel)
    if 'ylabel' in kwargs:
        ax.set_ylabel(ylabel)

    plt.show()

if '__main__' == __name__:
    try:
        data = np.loadtxt(sys.argv[1])
    except:
        raise
    
    plot(data)
