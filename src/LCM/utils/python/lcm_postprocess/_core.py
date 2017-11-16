#!/usr/bin/python

import contextlib
import os
import time

#
# Local functions
#
@contextlib.contextmanager
def nostdout():

    import cStringIO
    import sys

    save_stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()
    yield
    sys.stdout = save_stdout

@contextlib.contextmanager
def stdout_redirected(to = os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''

    import cStringIO
    import sys

    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to = file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to = old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

#
# Local classes
#
class InputError(Exception):

    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)



def read_input(arglist, delimiter = '='):

    argdict = {}

    for arg in arglist:

        key, value = arg.split(delimiter)

        argdict[key] = value

    return argdict

#
# Class for timing
#
class Timer:  

    def __enter__(self):
        self.start = time.clock()
        self.now = self.start
        self.last = self.now
        return self

    def __exit__(self, *args):
        self.check()
        self.end = self.now

    def check(self):
        self.now = time.clock()
        self.step = self.now - self.last
        self.interval = self.now - self.start
        self.last = self.now
        return self.step

    def print_time(self):
        self.check()
        hours = int(self.step / 3600.)
        minutes = int((self.step - 3600. * hours) / 60.)
        seconds = self.step - 3600. * hours - 60. * minutes
        time_step = '{:02d}h {:02d}m {:05.2f}s'.format(hours, minutes, seconds)
        hours = int(self.interval / 3600.)
        minutes = int((self.interval - 3600. * hours) / 60.)
        seconds = self.interval - 3600. * hours - 60. * minutes
        time_run = '{:02d}h {:02d}m {:05.2f}s'.format(hours, minutes, seconds)
        print '    Elapsed time: ' + time_step, '    Running time: ' + time_run + '\n'

#
# Class for common properties of local objects
#
class ObjLocal(object):

    def __repr__(self):

        str_repr = self.__class__.__name__ + '('

        items = self.__dict__.items()

        str_repr += items[0][0] + ' = ' + str(items[0][1])

        for item in items[1:]:

            if isinstance(item[1], (int, int, float, str)):
                str_item = str(item[1])
            else:
                str_item = item[1].__class__.__name__

            str_repr += ', ' + item[0] + ' = ' + str_item

        str_repr += ')'

        return str_repr

    def view(self):

        view_tree(obj_data = self)




#
# Topology data structure
#
class ObjDomain(ObjLocal):

    # @profile
    def __init__(self, **kwargs):

        self.blocks = dict()
        self.nodes = dict()
        self.variables = dict()
        self.names_variable_node = dict()
        self.names_variable_element = dict()

        for (key, value) in kwargs.items():
            setattr(self, key, value)
    #         self.variables.append(key)

    # def __setattr__(self, name, value, variable = False):

    #     super(objDomain).__setattr__(name, value)
    #     if variable == True:
    #         self.variables.append(name)


# Node
class ObjNode(ObjLocal):

    # @profile
    def __init__(self, **kwargs):

        self.variables = dict()

        for (key, value) in kwargs.items():
            setattr(self, key, value)


# Element block
class ObjBlock(ObjLocal):

    # @profile
    def __init__(self, **kwargs):

        self.elements = dict()
        self.material = ObjMaterial()
        self.variables = dict()

        for (key, value) in kwargs.items():
            setattr(self, key, value)


# Element
class ObjElement(ObjLocal):

    # @profile
    def __init__(self, **kwargs):

        self.points = dict()
        self.nodes = dict()
        self.variables = dict()

        for (key, value) in kwargs.items():
            setattr(self, key, value)


# Integration point
class ObjPoint(ObjLocal):

    # @profile
    def __init__(self, **kwargs):

        self.variables = dict()

        for (key, value) in kwargs.items():
            setattr(self, key, value)


# Material
class ObjMaterial(ObjLocal):

    def __init__(self, **kwargs):

        self.name = ''

        for (key, value) in kwargs.items():
            setattr(self, key, value)




#
# Simulation numerical information data structure
#
class ObjRun(ObjLocal):

    def __init__(self, **kwargs):

        self.steps = dict()
        self.num_iters_nonlinear = 0
        self.num_iters_linear = 0

        self.num_processors = 1

        self.time_compute = 0.0
        self.time_linsolve = 0.0
        self.time_constitutive = 0.0

        for (key, value) in kwargs.items():
            setattr(self, key, value)


# Time step
class ObjStep(ObjLocal):

    def __init__(self, **kwargs):

        self.iters_nonlinear = dict()
        self.num_iters_nonlinear = 0
        self.num_iters_linear = 0
        self.step_number = 0
        self.size_step = 0.0
        self.status_convergence = 0

        for (key, value) in kwargs.items():
            setattr(self, key, value)


# Nonlinear solver iteration
class ObjIterNonlinear(ObjLocal):

    def __init__(self, **kwargs):

        self.iters_linear = dict()
        self.status_convergence = 0
        self.num_iters_linear = 0
        self.norm_residual = 0.0
        self.norm_increment = 0.0

        for (key, value) in kwargs.items():
            setattr(self, key, value)


# Linear solver iteration
class ObjIterLinear(ObjLocal):

    def __init__(self, **kwargs):

        self.norm_residual = 0.0

        for (key, value) in kwargs.items():
            setattr(self, key, value)




# Populate a ttk.Treeview object for viewing data
def _populate_tree(tree, parent, dic):

    import xml.etree.ElementTree as et
    import numpy as np
    import uuid
    from lcm_postprocess import ObjLocal

    try:

        for key in sorted(dic):

            uid = uuid.uuid4()

            if isinstance(dic[key], dict):

                tree.insert(parent, 'end', uid, text = str(key) + '{}')
                _populate_tree(tree, uid, dic[key])

            elif isinstance(dic[key], et.Element):

                list_children = dic[key].__dict__['_children']
                if 'value' in dic[key].attrib:
                    value = dic[key].attrib['value']
                else:
                    value = ''
                tree.insert(
                    parent,
                    'end',
                    uid,
                    text = dic[key].attrib['name'],
                    value = value)
                _populate_tree(
                    tree,
                    uid,
                    dict([(i, x) for i, x in enumerate(list_children)]))

            elif isinstance(dic[key], (tuple, np.ndarray)):
                
                tree.insert(parent, 'end', uid, text = str(key) + '()')
                _populate_tree(
                    tree,
                    uid,
                    dict([(i, x) for i, x in enumerate(dic[key])]))

            elif isinstance(dic[key], list):
                
                tree.insert(parent, 'end', uid, text = str(key) + '[]')
                _populate_tree(
                    tree, 
                    uid,
                    dict([(i, x) for i, x in enumerate(dic[key])]))

            elif isinstance(dic[key], ObjLocal):

                tree.insert(parent, 'end', uid, text = str(key) + '<>')
                _populate_tree(
                    tree,
                    uid,
                    dic[key].__dict__)

            else:
                
                value = dic[key]
                if isinstance(value, str):
                    value = value.replace(' ', '_')
                tree.insert(parent, 0, uid, text = str(key), value = value)

    except KeyboardInterrupt:

        return

# end def populate_tree(tree, parent, dic):





# View simulation data as a ttk.Treeview object
def _view_tree(dict_data = None, obj_data = None, filename = None):

    import Tkinter as tk
    import ttk

    if filename != None:
        extension = filename.split('.')[-1]
        if extension == 'pickle':
            import cPickle as pickle
            file_pickling = open(filename, 'rb')
            obj_data = pickle.load(file_pickling)
            file_pickling.close()
        elif extension == 'xml':
            import xml.etree.ElementTree as et
            tree = et.parse(filename)
            obj_data = tree.getroot()

    if obj_data != None:
        dict_data = obj_data.__dict__

    # Setup the root UI
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.title("Object Viewer")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Setup the Frames
    tree_frame = ttk.Frame(root, padding="3")
    tree_frame.grid(row=0, column=0, sticky=tk.NSEW)

    # Setup the Tree
    tree = ttk.Treeview(tree_frame, columns=('Values'))
    tree.column('Values', width=100, anchor='center')
    tree.heading('Values', text='Values')
    _populate_tree(tree, '', dict_data)
    tree.pack(fill=tk.BOTH, expand=1)

    # Limit windows minimum dimensions
    root.update_idletasks()
    root.minsize(root.winfo_reqwidth(), root.winfo_reqheight())
    root.mainloop()

# end view_tree(dict_data = None, obj_data = None, filename = None):
