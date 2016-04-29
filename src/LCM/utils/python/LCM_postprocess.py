#!/usr/bin/python
'''
LCM_postprocess.py input.e [output.e]

creates usable output from LCM calculations
'''

#
# Imported modules
#
import sys
import os
import xml.etree.ElementTree as et
from exodus import exodus
from exodus import copy_mesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.linalg import *



#
# Local classes
#
class objDomain(object):

    def __init__(self, **kwargs):

        self.blocks = dict()

        for (key, value) in kwargs.items():
            setattr(self, key, value)



class objBlock(object):

    def __init__(self, **kwargs):

        self.elements = dict()

        for (key, value) in kwargs.items():
            setattr(self, key, value)



class objElement(object):

    def __init__(self, **kwargs):

        self.points = dict()

        for (key, value) in kwargs.items():
            setattr(self, key, value)



class objPoint(object):

    def __init__(self, **kwargs):

        for (key, value) in kwargs.items():
            setattr(self, key, value)



#
# Local functions
#
def readXml(nameFileBase, **kwargs):
    
    tree = et.parse(nameFileBase+'_Material.xml')
    root = tree.getroot()

    for child in root:

        if child.attrib['name'] == 'ElementBlocks':

            namesMaterial = dict()

            for block in child:

                nameBlock = block.attrib['name']

                for parameter in block:

                    if parameter.attrib['name'] == 'material':

                        nameMaterial = parameter.attrib['value']

                namesMaterial[nameBlock] = nameMaterial

    for keyArg in kwargs:

        if keyArg == 'orientations':

            orientations = kwargs[keyArg]

            for child in root:

                if child.attrib['name'] == 'Materials':

                    for material in child:

                        nameMaterial = material.attrib['name']

                        for plist in material:

                            if plist.attrib['name'] == 'Crystal Elasticity':

                                orientation = np.zeros((3,3))

                                for parameter in plist:

                                    nameParamter = parameter.attrib['name']

                                    if nameParamter.split()[0] == 'Basis':

                                        orientation[int(nameParamter.split()[2])-1][:] = \
                                            np.fromstring(
                                                parameter.attrib['value'].translate(None,'{}'),
                                                sep = ',')

                        for key, value in namesMaterial.items():

                            if value == nameMaterial:

                                orientations[key] = orientation

# end def readXml(nameFileInput):




def setWeightsVolumes(fileInput, domain):

    for keyBlock in domain.blocks:

        for indexPoint in range(domain.blocks[keyBlock].num_points):

            keyWeight = 'Weights_' + str(indexPoint + 1)

            values_weights = fileInput.get_element_variable_values(
                keyBlock, 
                keyWeight, 
                1)

            for keyElement in domain.blocks[keyBlock].elements:

                domain.blocks[keyBlock].elements[keyElement].points[indexPoint].weight = values_weights[keyElement]

        for keyElement in domain.blocks[keyBlock].elements:

            domain.blocks[keyBlock].elements[keyElement].volume = \
                np.sum([domain.blocks[keyBlock].elements[keyElement].points[x].weight for x in
                    domain.blocks[keyBlock].elements[keyElement].points])

        domain.blocks[keyBlock].volume = np.sum([domain.blocks[keyBlock].elements[x].volume for x in
            domain.blocks[keyBlock].elements])

    domain.volume = np.sum([domain.blocks[x].volume for x in domain.blocks])

# end setWeightsVolumes(fileInput, domain):



def setValuesTensor(fileInput, nameVariable, domain):

    num_dims = domain.num_dims

    times = fileInput.get_times()

    setattr(
        domain, 
        nameVariable,
        dict([(step, np.zeros((num_dims, num_dims))) for step in times]))

    # Note: exodus function get_element_variable_values returns values by block,    
    # so outer loop over block

    for keyBlock in domain.blocks:

        block = domain.blocks[keyBlock]

        setattr(
            block, 
            nameVariable,
            dict([(step, np.zeros((num_dims, num_dims))) for step in times]))

        for keyElement in block.elements:

            element = block.elements[keyElement]

            setattr(
                element, 
                nameVariable,
                dict([(step, np.zeros((num_dims, num_dims))) for step in times]))

            for keyPoint in element.points:

                point = element.points[keyPoint]

                setattr(
                    point, 
                    nameVariable,
                    dict([(step, np.zeros((num_dims, num_dims))) for step in times]))
                
        for step in range(len(times)):

            for dim_i in range(num_dims):

                for dim_j in range(num_dims):

                    index_dim = num_dims * dim_i + dim_j + 1

                    for indexPoint in range(block.num_points):

                        indexVariable = indexPoint * num_dims**2 + index_dim

                        keyVariable = nameVariable + '_{:02d}'.format(indexVariable)

                        valuesBlock = fileInput.get_element_variable_values(
                            keyBlock, 
                            keyVariable, 
                            step + 1)

                        for keyElement in block.elements:

                            element = block.elements[keyElement]

                            point = element.points[indexPoint]

                            getattr(point, nameVariable)[times[step]][dim_i, dim_j] = valuesBlock[keyElement]


                    for keyElement in block.elements:

                        element = block.elements[keyElement]

                        for keyPoint in element.points:

                            point = element.points[keyPoint]

                            getattr(element, nameVariable)[times[step]][dim_i, dim_j] += \
                                getattr(point, nameVariable)[times[step]][dim_i, dim_j] * \
                                point.weight / element.volume

                        getattr(block, nameVariable)[times[step]][dim_i, dim_j] += \
                            getattr(element, nameVariable)[times[step]][dim_i, dim_j] * \
                            element.volume / block.volume

                    getattr(domain, nameVariable)[times[step]][dim_i, dim_j] += \
                        getattr(block, nameVariable)[times[step]][dim_i, dim_j] * \
                        block.volume / domain.volume

# end def setValuesTensor(fileInput, nameVariable, domain):  




def plotInversePoleFigure(**kwargs):

    #
    # Read data 
    #
    if 'nameFileInput' in kwargs:
        nameFileInput = kwargs['nameFileInput']
        nameFileBase = nameFileInput.split('.')[0]
        orientations = np.loadtxt(nameFileInput)
    elif 'orientations' in kwargs:
        orientations = kwargs['orientations']
        nameFileBase = ''
    else:
        raise TypeError('Need either nameFileInput or orientations keyword args')

    RD = abs(orientations[:, 0:3])
    TD = abs(orientations[:, 3:6])
    ND = abs(orientations[:, 6:9])

    #
    # Create axes 
    #
    HKL = np.ones(3)
    XX = np.zeros(103)
    YY = np.zeros(103)

    for x in range(101):

        HKL[0] = -1 / 100. * (1 + x) + 1 / 100.
        HKL[1] = 1
        HKL[2] = 1
        
        HKL /= np.linalg.norm(HKL)        
        XX[x] = HKL[1] / (1 + HKL[2])
        YY[x] = -HKL[0] / (1 + HKL[2])
        
    XX[102] = XX[0]
    YY[102] = 0

    #
    # Convert orientations to ipf space 
    #
    X_RD = np.zeros(len(orientations))
    Y_RD = np.zeros(len(orientations))
    for x in range(len(orientations)):
        A = sorted(RD[x,:])    
        X_RD[x] = A[1] / (1 + A[2])
        Y_RD[x] = A[0] / (1 + A[2])

    X_TD = np.zeros(len(orientations))
    Y_TD = np.zeros(len(orientations))
    for x in range(len(orientations)):
        A = sorted(TD[x,:])    
        X_TD[x] = A[1] / (1 + A[2])
        Y_TD[x] = A[0] / (1 + A[2])
        
    X_ND = np.zeros(len(orientations))
    Y_ND = np.zeros(len(orientations))
    for x in range(len(orientations)):
        A = sorted(ND[x,:])    
        X_ND[x] = A[1] / (1 + A[2])
        Y_ND[x] = A[0] / (1 + A[2])                        
                                                                            
    #
    # Create figures  
    #
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif', size = 22)

    fig = plt.figure(figsize = (15,4))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    fig.suptitle('Inverse pole figures', fontsize = 14, fontweight = 'bold')

    plt.subplot(1, 3, 1)
    plt.plot(X_RD, Y_RD, 'ro')
    plt.plot(XX, YY, 'k', linewidth = 1)
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.axis('off')
    ax1.text(0.2, -0.05, 'RD', fontsize = 16)
    ax1.text(-0.03, -0.03, r'$[001]$', fontsize = 15)
    ax1.text(0.38, -0.03, r'$[011]$', fontsize = 15)
    ax1.text(0.34, 0.375, r'$[\bar111]$', fontsize = 15)

    plt.subplot(1, 3, 2)
    plt.plot(X_TD, Y_TD, 'bo')
    plt.plot(XX, YY, 'k', linewidth = 1)
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.axis('off')
    ax2.text(0.2, -0.05, 'TD', fontsize = 16)
    ax2.text(-0.03, -0.03, r'$[001]$', fontsize = 15)
    ax2.text(0.38, -0.03, r'$[011]$', fontsize = 15)
    ax2.text(0.34, 0.375, r'$[\bar111]$', fontsize = 15)

    plt.subplot(1, 3, 3)
    plt.plot(X_ND, Y_ND, 'go')
    plt.plot(XX, YY, 'k', linewidth = 1)
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.axis('off')
    ax3.text(0.2, -0.05, 'ND', fontsize = 16)
    ax3.text(-0.03, -0.03, r'$[001]$', fontsize = 15)
    ax3.text(0.38, -0.03, r'$[011]$', fontsize = 15)
    ax3.text(0.34, 0.375, r'$[\bar111]$', fontsize = 15)

    plt.savefig(nameFileBase + '_IPF.pdf')

# end plotInversePoleFigure(**kwargs):



def writeData(data, nameFileData, precision = 8):

    file = open(nameFileData, 'w')

    for data_step in data:

        print data_step

        data_step.tofile(file, sep = ' ')
        # for datum in data_step:
        #     print datum, precision
        #     file.write('{:.{}e} '.format(datum, precision))
        file.write('\n')

    file.close()

# end def writeData(data, nameFileData)




def postprocess(nameFileInput, **kwargs):

    debug = 1

    #
    # Set i/o units 
    #
    nameFileBase = nameFileInput.split('.')[0]
    nameFileExtension = nameFileInput.split('.')[-1]

    #
    # Read input file
    #
    fileInput = exodus(nameFileInput,"r")



    #
    # Get values of whole domain variables
    #

    # get number of dimensions
    num_dims = fileInput.num_dimensions()
    print "Dimensions"
    print num_dims

    # Get number of elements
    num_elements = dict([(id,fileInput.num_elems_in_blk(id)) for id in fileInput.get_elem_blk_ids()])
    print 'Number of elements'
    print np.sum(num_elements.values())

    # Get output times
    times = fileInput.get_times()
    num_times = len(times)
    print "Print the times"
    print times[0], times[-1]


    # Get list of nodal variables
    node_var_names= fileInput.get_node_variable_names()
    print "Printing the names of the nodal variables"
    print node_var_names


    # Get list of element variables
    elem_var_names = fileInput.get_element_variable_names()
    num_variables_unique = 0
    names_variable_unique = []
    print "Printing the names of the element variables"

    # Get unique element variables (combine across integration points)
    for name in elem_var_names:
        new_name = True
        for name_unique in names_variable_unique:
          if (name.startswith(name_unique+"_")):
            new_name = False
        if (new_name == True):
          indices = [int(s) for s in name if s.isdigit()]
          names_variable_unique.append(name[0:name.find(str(indices[0]))-1])
    print names_variable_unique


    # Get number of element blocks and block ids
    block_ids = fileInput.get_elem_blk_ids()
    num_blocks = fileInput.num_blks()

    # Calculate number of integration points
    num_points = 0
    for name in elem_var_names:
      if (name.startswith("Weights_")):
          num_points += 1

    # Check that "Weights" exist as an element variable
    if (num_points == 0):
      raise Exception("The weights field is not available...try again.")
          
    print "Number of Integration points"
    print num_points



    #
    # Create the mesh stucture
    #
    domain = objDomain(
        num_dims = num_dims,
        num_elements = np.sum(num_elements.values()))

    for block_id in block_ids:

        domain.blocks[block_id] = objBlock(
            num_elements = num_elements[block_id],
            num_points = num_points)
        # TODO: change line above to line below
        # num_points =  = num_points[block_id]

        for indexElement in range(num_elements[block_id]):

            domain.blocks[block_id].elements[indexElement] = objElement()

            for indexPoint in range(num_points):

                domain.blocks[block_id].elements[indexElement].points[indexPoint] = objPoint()


    #
    # Get material data
    #
    orientations = dict()
    readXml(nameFileBase, orientations = orientations)
    for block_id in block_ids:
        nameBlock = fileInput.get_elem_blk_name(block_id)
        domain.blocks[block_id].orientation = orientations[nameBlock]



    #
    # Record the integration point weights and block volumes
    #
    setWeightsVolumes(fileInput, domain)
    
    if debug != 0:
        print 'Volume: ', domain.volume

    #
    # Average the quantities of interest
    #
    for nameVariable in names_variable_unique:

        #
        # Handle the cauchy stress
        #
        if (nameVariable == 'Cauchy_Stress'):

            print nameVariable

            setValuesTensor(fileInput, nameVariable, domain)

        #
        # Handle the deformation gradient
        #
        elif (nameVariable == 'F'):

            print nameVariable

            setValuesTensor(fileInput, nameVariable, domain)

    #
    # Calculate the deformed orientations
    #

    print 'Computing deformed orientations'

    for keyBlock in domain.blocks:

        block = domain.blocks[keyBlock]

        for keyElement in block.elements:

            element = block.elements[keyElement]

            element.R = dict()
            element.U = dict()
            element.orientation = dict()

            for step in times:

                element.R[step], element.U[step] = polar(element.F[step])

                element.orientation[step] = element.R[step] * block.orientation

    # for step in times:

    #     orientations = [x for x in element.orientation[step] for element in 





    return domain





    #
    # Write data to exodus output file
    #
    if 'nameFileOutput' in kwargs:
        nameFileOutput = kwargs['nameFileOutput']
    else:
        nameFileOutput = nameFileBase + '_Postprocess.' + nameFileExtension
    
    if os.path.isfile(nameFileOutput):
        cmdLine = "rm %s" % nameFileOutput
        os.system(cmdLine)
    fileOutput = copy_mesh(nameFileInput, nameFileOutput)

    # write times to fileOutput
    for step in range(len(times)):
      fileOutput.put_time(step+1,times[step])

    # write out displacement vector
    dx = []
    dy = []
    dz = []

    fileOutput.set_node_variable_number(int(3))
    fileOutput.put_node_variable_name('displacement_x', 1)
    fileOutput.put_node_variable_name('displacement_y', 2)
    fileOutput.put_node_variable_name('displacement_z', 3)

    for step in range(len(times)):

      dx = fileInput.get_node_variable_values('displacement_x',step+1)
      dy = fileInput.get_node_variable_values('displacement_y',step+1)
      dz = fileInput.get_node_variable_values('displacement_z',step+1)

      fileOutput.put_node_variable_values('displacement_x',step+1,dx)
      fileOutput.put_node_variable_values('displacement_y',step+1,dy)
      fileOutput.put_node_variable_values('displacement_z',step+1,dz)

    #
    # create variables in output file
    #
    fileOutput.set_element_variable_number(2 * num_dims**2)

    for dim_i in range(num_dims):

        for dim_j in range(num_dims):

            fileOutput.put_element_variable_name(
                'Cauchy_Stress_' + str(dim_i + 1) + str(dim_j + 1), 
                dim_i * num_dims + dim_j + 1)

            fileOutput.put_element_variable_name(
                'F_' + str(dim_i + 1) + str(dim_j + 1), 
                num_dims**2 + dim_i * num_dims + dim_j + 1)

    fileOutput.close()

    strain = dict()
    stress = dict()
    for step in range(num_times):
        defgrad_step = np.zeros((3,3))
        stress_step = np.zeros((3,3))
        for dim_i in range(num_dims):
            for dim_j in range(num_dims):
                defgrad_step[dim_i][dim_j] = defgrad[step][(dim_i, dim_j)]
                stress_step[dim_i][dim_j] = stress_cauchy[step][(dim_i, dim_j)]
        strain[step] = 0.5 * logm(np.transpose(defgrad_step) * defgrad_step)
        stress[step] = stress_step


    # writeData(strain.values(), nameFileInput.split()[0]+'.dat', precision = 8)

    file = open(nameFileInput.split('.')[0]+'.dat', 'w')

    for (strain_step, stress_step) in zip(strain.values(), stress.values()):

        strain_step.tofile(file, sep=' ')
        file.write(' ')
        stress_step.tofile(file, sep=' ')
        # for datum in data_step:
        #     print datum, precision
        #     file.write('{:.{}e} '.format(datum, precision))
        file.write('\n')

    file.close()

    rcParams.update({'figure.autolayout': True})

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=22)

    fig = plt.figure()

    for dim_i in range(num_dims):

        for dim_j in range(num_dims):

            fig.clf()
            plt.plot(
                [strain[step][dim_i][dim_j] for step in range(num_times)],
                [stress_cauchy[step][(dim_i, dim_j)] for step in range(num_times)])
            plt.xlabel('Logarithmic Strain $\epsilon_{'+ str(dim_i + 1) + str(dim_j + 1) +'}$')
            plt.ylabel('Cauchy Stress $\sigma_{'+ str(dim_i + 1) + str(dim_j + 1) +'}$ (MPa)')
            plt.savefig('stress_strain_'+ str(dim_i + 1) + str(dim_j + 1) +'.pdf')

    return domain

# end def postprocess(nameFileInput, nameFileOutput):



if __name__ == '__main__':
    nameFileInput = sys.argv[1]
    try:
        nameFileOutput = sys.argv[2]
        domain = postprocess(nameFileInput, nameFileOutput)
    except:
        domain = postprocess(nameFileInput)