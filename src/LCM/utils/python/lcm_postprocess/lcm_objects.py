#!/usr/bin/python

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


# Element block
class ObjNode(ObjLocal):

    def __init__(self, **kwargs):

        self.variables = dict()

        for (key, value) in kwargs.items():
            setattr(self, key, value)


# Element block
class ObjBlock(ObjLocal):

    def __init__(self, **kwargs):

        self.elements = dict()
        self.material = ObjMaterial()
        self.variables = dict()

        for (key, value) in kwargs.items():
            setattr(self, key, value)


# Element
class ObjElement(ObjLocal):

    def __init__(self, **kwargs):

        self.points = dict()
        self.nodes = dict()
        self.variables = dict()

        for (key, value) in kwargs.items():
            setattr(self, key, value)


# Integration point
class ObjPoint(ObjLocal):
    
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
def populate_tree(tree, parent, dic):

    import xml.etree.ElementTree as et
    import numpy as np
    import uuid
    from lcm_postprocess.lcm_objects import ObjLocal

    try:

        for key in dic:

            uid = uuid.uuid4()

            if isinstance(dic[key], dict):

                tree.insert(parent, 'end', uid, text=str(key) + '{}')
                populate_tree(tree, uid, dic[key])

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
                populate_tree(tree, uid, dict([(i, x) for i, x in enumerate(list_children)]))

            elif isinstance(dic[key], (tuple, np.ndarray)):
                
                tree.insert(parent, 'end', uid, text=str(key) + '()')
                populate_tree(
                    tree,
                    uid,
                    dict([(i, x) for i, x in enumerate(dic[key])]))

            elif isinstance(dic[key], list):
                
                tree.insert(parent, 'end', uid, text=str(key) + '[]')
                populate_tree(
                    tree, 
                    uid,
                    dict([(i, x) for i, x in enumerate(dic[key])]))

            elif isinstance(dic[key], ObjLocal):

                tree.insert(parent, 'end', uid, text=str(key) + '<>')
                populate_tree(tree, uid, dic[key].__dict__)

            else:
                
                value = dic[key]
                if isinstance(value, str):
                    value = value.replace(' ', '_')
                tree.insert(parent, 0, uid, text=str(key), value=value)

    except KeyboardInterrupt:

        return

# end def populate_tree(tree, parent, dic):





# View simulation data as a ttk.Treeview object
def view_tree(dict_data = None, obj_data = None, filename = None):

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
    populate_tree(tree, '', dict_data)
    tree.pack(fill=tk.BOTH, expand=1)

    # Limit windows minimum dimensions
    root.update_idletasks()
    root.minsize(root.winfo_reqwidth(), root.winfo_reqheight())
    root.mainloop()

# end view_tree(dict_data = None, obj_data = None, filename = None):



if __name__ == '__main__':

    import sys

    try:
        name_file_input = sys.argv[1]
    except:
        raise

    view_tree(filename = name_file_input)

# end if __name__ == '__main__':