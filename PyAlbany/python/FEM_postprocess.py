import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import sys
import math

try:
    import exomerge
except:
    if sys.version_info.major == 2:
        import exomerge2 as exomerge
    if sys.version_info.major == 3:
        import exomerge3 as exomerge

def quads_to_tris(quads):
    # credits: https://stackoverflow.com/questions/52202014/how-can-i-plot-2d-fem-results-using-matplotlib
    tris = [[None for j in range(3)] for i in range(2*len(quads))]
    for i in range(len(quads)):
        j = 2*i
        n0 = quads[i][0]
        n1 = quads[i][1]
        n2 = quads[i][2]
        n3 = quads[i][3]
        tris[j][0] = n0
        tris[j][1] = n1
        tris[j][2] = n2
        tris[j + 1][0] = n2
        tris[j + 1][1] = n3
        tris[j + 1][2] = n0
    return np.array(tris)

# plots a finite element mesh
def plot_fem_mesh(nodes_x, nodes_y, elements):
    # credits: https://stackoverflow.com/questions/52202014/how-can-i-plot-2d-fem-results-using-matplotlib
    for element in elements:
        plt.fill(nodes_x[element], nodes_y[element], edgecolor='black', fill=False)


def readExodus(filename, solnames=[], nProcs=1, timesteps='last'):
    n_sol = len(solnames)
    if nProcs == 1:
        model = exomerge.import_model(filename, timesteps=timesteps)
        positions = np.array(model.nodes)
        x = np.ascontiguousarray(positions[:,0])
        y = np.ascontiguousarray(positions[:,1])
        for element_block_id in model.get_element_block_ids():
            connectivity = model.get_connectivity(element_block_id)
            nodes_per_element = model.get_nodes_per_element(element_block_id)
            element_count = int(len(connectivity) / nodes_per_element)
            elements = np.zeros((element_count, nodes_per_element), dtype=int)
            for element_index in range(element_count):        
                elements[element_index, :] = connectivity[element_index * nodes_per_element: (element_index + 1) * nodes_per_element]
    else:
        x = np.array([])
        y = np.array([])

        current_index = 0

        digits_nProcs = int(math.log10(nProcs))+1

        for i_proc in range(0, nProcs):
            if i_proc != 0:
                digits_i_proc = int(math.log10(i_proc))+1
            else:
                digits_i_proc = 1
            tmp_filename = filename+'.'+str(nProcs)+'.'
            for i_digit in range(digits_i_proc, digits_nProcs):
                tmp_filename += '0'
            tmp_filename += str(i_proc)
            model = exomerge.import_model(tmp_filename, timesteps=timesteps)
            positions = np.array(model.nodes)
            x = np.append(x, np.ascontiguousarray(positions[:,0]))
            y = np.append(y, np.ascontiguousarray(positions[:,1]))

            next_index = current_index + len(np.ascontiguousarray(positions[:,0]))
            for element_block_id in model.get_element_block_ids():
                connectivity = model.get_connectivity(element_block_id)
                nodes_per_element = model.get_nodes_per_element(element_block_id)
                element_count = int(len(connectivity) / nodes_per_element)
                current_elements = np.zeros((element_count, nodes_per_element), dtype=int)
                for element_index in range(element_count):        
                    current_elements[element_index, :] = connectivity[element_index * nodes_per_element: (element_index + 1) * nodes_per_element]
                current_elements += current_index
                if i_proc == 0:
                    elements = current_elements
                else:
                    elements = np.append(elements, current_elements, axis=0)
            current_index = next_index

    if n_sol == 0:
        return x, y, elements

    if n_sol != 0:
        sol = np.zeros((n_sol, len(x)))
        if nProcs == 1:
            for i in range(0, n_sol):
                sol[i,:] = np.ascontiguousarray(model.node_fields[solnames[i]])[0,:]
        else:
            current_index = 0
            for i_proc in range(0, nProcs):
                if i_proc != 0:
                    digits_i_proc = int(math.log10(i_proc))+1
                else:
                    digits_i_proc = 1
                tmp_filename = filename+'.'+str(nProcs)+'.'
                for i_digit in range(digits_i_proc, digits_nProcs):
                    tmp_filename += '0'
                tmp_filename += str(i_proc)
                model = exomerge.import_model(tmp_filename, timesteps=timesteps)
                current_length = len(np.ascontiguousarray(model.node_fields[solnames[0]])[0,:])
                for i in range(0, n_sol):
                    sol[i,current_index:(current_index+current_length)] = np.ascontiguousarray(model.node_fields[solnames[i]])[0,:]
                current_index += current_length

        if elements.shape[1] == 3:
            triangulation = tri.Triangulation(x, y, elements)
        if elements.shape[1] == 4:    
            triangulation = tri.Triangulation(x, y, quads_to_tris(elements))

        return x, y, sol, elements, triangulation

def tricontourf(x, y, z, elements, triangulation, output_file_name, figsize=(6, 4), zlabel='', dpi=800, show_mesh=True, cmap='coolwarm', nlevels=9):
    plt.figure(figsize=figsize)
    if show_mesh:
        plot_fem_mesh(x, y, elements)
    zmax = np.amax(z)
    zmin = np.amin(z)
    if zmax != zmin:
        levels = np.linspace(zmin, zmax, nlevels)
    else:
        levels = np.linspace(zmin-0.001*zmin, zmin+0.001*zmin, 3)
    plt.tricontourf(triangulation, z, cmap=cmap, levels=levels)
    cbar = plt.colorbar()
    plt.axis([np.amin(x), np.amax(x), np.amin(y), np.amax(y)])
    plt.gca().set_aspect('equal', adjustable='box')
    if zlabel != '':
        cbar.ax.set_ylabel(zlabel)
    plt.savefig(output_file_name, dpi=dpi, bbox_inches='tight',pad_inches = 0)
    plt.close()