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
        self.variables = []

        for (key, value) in kwargs.items():
            setattr(self, key, value)
    #         self.variables.append(key)

    # def __setattr__(self, name, value, variable = False):

    #     super(objDomain).__setattr__(name, value)
    #     if variable == True:
    #         self.variables.append(name)



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



def readInputFile(fileInput, namesVariable = ''):

    returnDict = {}

    for name in namesVariable:

        # get number of dimensions
        if name == 'num_dims':
            returnDict[name] = fileInput.num_dimensions()

        # Get number of elements
        elif name == 'num_elements': 
            returnDict[name] = dict([(id,fileInput.num_elems_in_blk(id)) for id in fileInput.get_elem_blk_ids()])

        # Get output times
        elif name == 'times':
            returnDict[name] = fileInput.get_times()

        # Get list of nodal variables
        elif name == 'node_var_names':
            returnDict[name] = fileInput.get_node_variable_names()

        # Get list of element variables
        elif name == 'elem_var_names':
            returnDict[name] = fileInput.get_element_variable_names()

            num_variables_unique = 0
            names_variable_unique = []

        # Get the unique variable names (deal with integration points and arrays)
        elif name == 'names_variable_unique':
           
            for nameElementVariable in fileInput.get_element_variable_names():
                new_name = True
                for name_unique in names_variable_unique:
                  if (nameElementVariable.startswith(name_unique+"_")):
                    new_name = False
                if (new_name == True):
                  indices = [int(s) for s in nameElementVariable if s.isdigit()]
                  names_variable_unique.append(nameElementVariable[0:nameElementVariable.find(str(indices[0]))-1])

            returnDict[name]= names_variable_unique

        # Get number of element blocks and block ids
        elif name == 'block_ids':
            returnDict[name] = fileInput.get_elem_blk_ids()

        elif name == 'num_blocks':
            returnDict[name] = fileInput.num_blks()

        # Calculate number of integration points
        elif name == 'num_points':
            num_points = 0
            for nameElementVariable in fileInput.get_element_variable_names():
              if (nameElementVariable.startswith("Weights_")):
                  num_points += 1

            # Check that "Weights" exist as an element variable
            if (num_points == 0):
              raise Exception("The weights field is not available...try again.")

            returnDict[name] = num_points

    return returnDict

# end def readInputFile(fileInput, **kwargs):




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





















def getDataLog(linesNorm):

    #
    # Get the lines for the converged and failed Newton steps
    #    
    linesConverged = [line for line in linesNorm if line.find('Converged') != -1]
    linesFailed = [line for line in linesNorm if line.find('Failed') != -1]        

    #
    # Calculate the number of converged and failed Newton steps
    #
    numStepsConverged = len(linesConverged)
    numStepsFailed = len(linesFailed)    

    #
    # Create a list for the converged and failed step numbers
    #
    iterationsConverged = [-1]
    for line in linesConverged:
        iterationsConverged.append(linesNorm.index(line,iterationsConverged[-1]+1))
    
    iterationsFailed = [-1]
    for line in linesFailed:
        iterationsFailed.append(linesNorm.index(line,iterationsFailed[-1]+1))

    #
    # Extract the data from the lines from the output log
    #
    listNormF = [float(line.split()[2]) for line in linesNorm]
    
    normF = [tuple(listNormF[iterationsConverged[i] + 1 : iterationsConverged[i + 1] + 1]) for i in range(numStepsConverged)]
    normResidualFailed  = [tuple(listNormF[iterationsFailed[i] + 1 : iterationsFailed[i+1] + 1]) for i in range(numStepsFailed)]  

    listNormDu = [float(line.split()[8]) for line in linesNorm]
    
    normDu = [tuple(listNormDu[iterationsConverged[i] + 1 : iterationsConverged[i + 1] + 1]) for i in range(numStepsConverged)]
    normIncrementFailed = [tuple(listNormDu[iterationsFailed[i] + 1 : iterationsFailed[i + 1] + 1]) for i in range(numStepsFailed)]    

    dataConverged = (normF, normDu)
    dataFailed = (normResidualFailed, normIncrementFailed)

    return dataConverged, dataFailed

# end def getDataLog(linesNorm):




def readFileLog(filename):

    #
    # Read the log file
    #
    file = open(filename, 'r')
    lines = file.readlines()
    file.close

    #
    # Extract lines that have residual norm information
    #
    linesNorm = [line for line in lines if line.find('||F||') != -1]

    #
    # Write the norm data to file
    #
    file = open('normF.dat', 'w')
    file.writelines(linesNorm)
    file.close

    dataConverged, dataFailed = getDataLog(linesNorm)

    normF, normDu = dataConverged
    numStepsConverged = len(normF)
    
    normResidualFailed, normIncrementFailed = dataFailed
    numStepsFailed = len(normResidualFailed)

    #
    # Loop through tuples of the data for plotting
    #
    for (dataResidual,dataIncrement,numSteps,label) in [(normF,normDu,numStepsConverged,'Converged'), (normResidualFailed,normIncrementFailed,numStepsFailed,'Failed')]:

        stringLegend = ['Step '+str(i+1) for i in range(numSteps)]

        #print 'Legend'
        #print stringLegend

        fig = plt.figure()
        plt.hold(True)
        for dataStep in dataResidual:
            #print 'dataStep'
            #print dataStep
            if np.max(dataStep) > 0.0:
                dataPlot = [point for point in dataStep if point > 0.0]
                plt.plot(dataPlot)
        plt.yscale('log')
        plt.legend(stringLegend)
    #    plt.show()
        plt.savefig('normF_Step_'+label+'.pdf')
        fig.clf()
        
        plt.hold(True)
        for dataStep in dataResidual:
            if np.max(dataStep) > 0.0:
                dataPlot = [point for point in dataStep if point > 0.0]
                plt.plot(dataPlot[:-1],dataPlot[1:])
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(stringLegend, loc = 'upper left')
    #    plt.show()
        plt.xlabel('$\| F_n \|$')
        plt.ylabel('$\| F_{n+1} \|$')
        plt.savefig('normF_Convergence_'+label+'.pdf')
        fig.clf()
        
        plt.hold(True)
        for dataStep in dataIncrement:
            if np.max(dataStep) > 0.0:
                dataPlot = [point for point in dataStep if point > 0.0]
                plt.plot(dataPlot)
        plt.yscale('log')
        plt.legend(stringLegend)
        plt.savefig('normDu_Step_'+label+'.pdf')
    #    plt.show(fig)
        fig.clf()
        
        plt.hold(True)
        for dataStep in dataIncrement:
            if np.max(dataStep) > 0.0:
                dataPlot = [point for point in dataStep if point > 0.0]
                plt.plot(dataPlot[1:-1],dataPlot[2:])
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(stringLegend, loc = 'upper left')
        plt.xlabel('Increment $\Delta u_n$')
        plt.ylabel('Increment $\Delta u_{n+1}$')
    #    plt.show()
        plt.savefig('normDu_Convergence_'+label+'.eps')    
    
    return dataConverged, dataFailed

# end def readFileLog(filename):

























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



def setValuesScalar(fileInput, nameVariable, domain):

    times = fileInput.get_times()

    setattr(
        domain, 
        nameVariable,
        dict([(step, 0.0) for step in times]))

    # Note: exodus function get_element_variable_values returns values by block,    
    # so outer loop over block

    for keyBlock in domain.blocks:

        block = domain.blocks[keyBlock]

        setattr(
            block, 
            nameVariable,
            dict([(step, 0.0) for step in times]))

        for keyElement in block.elements:

            element = block.elements[keyElement]

            setattr(
                element, 
                nameVariable,
                dict([(step, 0.0) for step in times]))

            for keyPoint in element.points:

                point = element.points[keyPoint]

                setattr(
                    point, 
                    nameVariable,
                    dict([(step, 0.0) for step in times]))
                
        for step in range(len(times)):

            for indexPoint in range(block.num_points):

                indexVariable = indexPoint

                keyVariable = nameVariable + '_{:1d}'.format(indexVariable + 1)

                valuesBlock = fileInput.get_element_variable_values(
                    keyBlock, 
                    keyVariable, 
                    step + 1)

                for keyElement in block.elements:

                    element = block.elements[keyElement]

                    for keyPoint in element.points:

                        point = element.points[keyPoint]

                        getattr(point, nameVariable)[times[step]] = valuesBlock[keyElement]

            for keyElement in block.elements:

                element = block.elements[keyElement]

                for keyPoint in element.points:

                    point = element.points[keyPoint]

                    getattr(element, nameVariable)[times[step]] += \
                        getattr(point, nameVariable)[times[step]] * \
                        point.weight / element.volume

                getattr(block, nameVariable)[times[step]] += \
                    getattr(element, nameVariable)[times[step]] * \
                    element.volume / block.volume

            getattr(domain, nameVariable)[times[step]] += \
                getattr(block, nameVariable)[times[step]] * \
                block.volume / domain.volume

# end def setValuesScalar(fileInput, nameVariable, domain): 




def plotInversePoleFigure(**kwargs):

    #
    # Read data 
    #
    if 'nameFileInput' in kwargs:

        nameFileInput = kwargs['nameFileInput']
        nameFileBase = nameFileInput.split('.')[0]
        orientations = np.loadtxt(nameFileInput)

    elif 'domain' in kwargs:

        domain = kwargs['domain']

        if 'time' in kwargs:
            time = kwargs['time']
        else:
            time = 0.0
        nameFileBase = str(time)

        listOrientations = []

        for keyBlock in domain.blocks:
            block = domain.blocks[keyBlock]
            for keyElement in block.elements:
                element = block.elements[keyElement]
                listOrientations.append(element.orientation[time].flatten())
        orientations = np.array(listOrientations)
    else:

        raise TypeError('Need either nameFileInput or orientations keyword args')

    #
    # Compute IPF quantities
    #
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
    plt.close(fig)

# end plotInversePoleFigure(**kwargs):



def writeData(domain, nameFileData, precision = 8):

    file = open(nameFileData, 'w')

    strFormat = '%.'+str(precision)+'e'

    file.write('Deformation Gradient\n')
    for step in domain.F:
        domain.F[step].tofile(file, sep = ' ', format = strFormat)
        file.write('\n')

    file.write('Cauchy Stress\n')
    for step in domain.Cauchy_Stress:
        domain.Cauchy_Stress[step].tofile(file, sep = ' ', format = strFormat)
        file.write('\n')

    file.close()

# end def writeData(data, nameFileData)




def writeExodusFile(domain, fileInput, nameFileOutput):

    times = domain.times
    num_dims = domain.num_dims

    if os.path.isfile(nameFileOutput):
        cmdLine = "rm %s" % nameFileOutput
        os.system(cmdLine)
    fileOutput = fileInput.copy(nameFileOutput)

    # write times to fileOutput
    for step in range(len(times)):
       fileOutput.put_time(step + 1, times[step])

    #
    # write out displacement vector
    #
    fileOutput.set_node_variable_number(int(3))
    fileOutput.put_node_variable_name('displacement_x', 1)
    fileOutput.put_node_variable_name('displacement_y', 2)
    fileOutput.put_node_variable_name('displacement_z', 3)

    for step in range(len(times)):

        fileOutput.put_node_variable_values(
            'displacement_x',
            step + 1,
            fileInput.get_node_variable_values('displacement_x',step+1))

        fileOutput.put_node_variable_values(
            'displacement_y',
            step + 1,
            fileInput.get_node_variable_values('displacement_y',step+1))

        fileOutput.put_node_variable_values(
            'displacement_z',
            step + 1,
            fileInput.get_node_variable_values('displacement_z',step+1))

    #
    # create variables in output file
    #
    fileOutput.set_element_variable_number(2 * num_dims**2 + 2)

    for dim_i in range(num_dims):

        for dim_j in range(num_dims):

            nameStress = 'Cauchy_Stress_' + str(dim_i + 1) + str(dim_j + 1)

            fileOutput.put_element_variable_name(
                nameStress, 
                dim_i * num_dims + dim_j + 1)

            nameDefGrad = 'F_' + str(dim_i + 1) + str(dim_j + 1)

            fileOutput.put_element_variable_name(
                nameDefGrad, 
                num_dims**2 + dim_i * num_dims + dim_j + 1)

            for keyBlock in domain.blocks:

                block = domain.blocks[keyBlock]

                for step in range(len(times)):

                    fileOutput.put_element_variable_values(
                        keyBlock,
                        nameStress,
                        step + 1,
                        [block.elements[keyElement].Cauchy_Stress[times[step]][dim_i][dim_j] for keyElement in block.elements])

                    fileOutput.put_element_variable_values(
                        keyBlock,
                        nameDefGrad,
                        step + 1,
                        [block.elements[keyElement].F[times[step]][dim_i][dim_j] for keyElement in block.elements])

    fileOutput.put_element_variable_name(
        'Mises_Stress', 
        2 * num_dims**2 + 1)

    for keyBlock in domain.blocks:

        block = domain.blocks[keyBlock]

        for step in range(len(times)):

            fileOutput.put_element_variable_values(
                keyBlock,
                'Mises_Stress',
                step + 1,
                [block.elements[keyElement].Mises_Stress[times[step]] for keyElement in block.elements])

    fileOutput.put_element_variable_name(
        'eqps', 
        2 * num_dims**2 + 2)

    for keyBlock in domain.blocks:

        block = domain.blocks[keyBlock]

        for step in range(len(times)):

            fileOutput.put_element_variable_values(
                keyBlock,
                'eqps',
                step + 1,
                [block.elements[keyElement].eqps[times[step]] for keyElement in block.elements])

            

    fileOutput.close()

# end def writeExodusFile(nameFileOutput):




def plotStressStrain(domain):

    num_dims = domain.num_dims
    times = domain.times

    rcParams.update({'figure.autolayout': True})

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=22)

    fig = plt.figure()

    strain_domain = dict()

    for keyStep in times:
        
        strain_domain[keyStep] = 0.5 * logm(np.inner(domain.F[keyStep].T, domain.F[keyStep].T))

    setattr(domain, 'Log_Strain', strain_domain)

    for keyBlock in domain.blocks:

        block = domain.blocks[keyBlock]

        strain_block = dict()

        for keyStep in times:

            strain_block[keyStep] = 0.5 * logm(np.inner(block.F[keyStep].T, block.F[keyStep].T))

        setattr(block, 'Log_Strain', strain_block)


    for dim_i in range(num_dims):

        for dim_j in range(num_dims):

            fig.clf()
            plt.hold(True)

            plt.plot(
                [domain.Log_Strain[keyStep][dim_i][dim_j] for keyStep in times],
                [domain.Cauchy_Stress[keyStep][(dim_i, dim_j)] for keyStep in times],
                marker = 'o')

            strLegend = ['Domain']

            for keyBlock in domain.blocks:

                block = domain.blocks[keyBlock]

                plt.plot(
                    [block.Log_Strain[keyStep][dim_i][dim_j] for keyStep in times],
                    [block.Cauchy_Stress[keyStep][(dim_i, dim_j)] for keyStep in times],
                    linestyle = ':')

                strLegend.append('Block ' + str(keyBlock))


            plt.xlabel('Logarithmic Strain $\epsilon_{'+ str(dim_i + 1) + str(dim_j + 1) +'}$')
            plt.ylabel('Cauchy Stress $\sigma_{'+ str(dim_i + 1) + str(dim_j + 1) +'}$ (MPa)')
            plt.legend(strLegend, loc = 'upper left', fontsize = 15)

            plt.savefig('stress_strain_'+ str(dim_i + 1) + str(dim_j + 1) +'.pdf')

# end def plotStressStrain(domain):




def postprocess(nameFileInput, **kwargs):

    debug = 1


    # Set i/o units 
    nameFileBase = nameFileInput.split('.')[0]
    nameFileExtension = nameFileInput.split('.')[-1]



    #
    # Get values of whole domain variables
    #
    print 'Reading input file...'

    fileInput = exodus(nameFileInput,"r")
    
    inputValues = readInputFile(
        fileInput,
        namesVariable = [
            'num_dims', 
            'num_elements', 
            'times', 
            'elem_var_names', 
            'names_variable_unique', 
            'block_ids', 
            'num_points'])

    num_dims = inputValues['num_dims']
    num_elements = inputValues['num_elements']
    times = inputValues['times']
    elem_var_names = inputValues['elem_var_names']
    names_variable_unique = inputValues['names_variable_unique']
    block_ids = inputValues['block_ids']
    num_points = inputValues['num_points']



    #
    # Create the mesh stucture
    #
    print 'Creating domain object...'

    domain = objDomain(
        num_dims = num_dims,
        num_elements = np.sum(num_elements.values()),
        times = times)

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
    print 'Retrieving material data...'

    orientations = dict()

    readXml(
        nameFileBase, 
        orientations = orientations)
    
    for block_id in block_ids:
        nameBlock = fileInput.get_elem_blk_name(block_id)
        domain.blocks[block_id].orientation = orientations[nameBlock]



    print 'Compiling output data...'
    
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
        # Handle the equivalent plastic strain
        #
        elif (nameVariable == 'eqps'):

            print nameVariable

            setValuesScalar(fileInput, nameVariable, domain)




    for keyBlock in domain.blocks:

        block = domain.blocks[keyBlock]

        for keyElement in block.elements:

            element = block.elements[keyElement]

            for keyPoint in element.points:

                point = element.points[keyPoint]

                setattr(
                    point, 
                    'Mises_Stress',
                    dict([(step, 0.0) for step in times]))

                for keyStep in times:

                    stressCauchy = point.Cauchy_Stress[keyStep]

                    stressDeviatoric = stressCauchy - 1. / 3. * np.trace(stressCauchy) * np.eye(3)

                    Mises_Stress = np.sqrt(3. / 2. * np.sum(np.tensordot(stressDeviatoric, stressDeviatoric, axes = 2)))

                    point.Mises_Stress[keyStep] = Mises_Stress

            setattr(
                element, 
                'Mises_Stress',
                dict([(step, 0.0) for step in times]))

            for keyStep in times:

                stressCauchy = element.Cauchy_Stress[keyStep]

                stressDeviatoric = stressCauchy - 1. / 3. * np.trace(stressCauchy) * np.eye(3)

                Mises_Stress = np.sqrt(3. / 2. * np.sum(np.tensordot(stressDeviatoric, stressDeviatoric, axes = 2)))

                element.Mises_Stress[keyStep] = Mises_Stress

        setattr(
            block, 
            'Mises_Stress',
            dict([(step, 0.0) for step in times]))

        for keyStep in times:

            stressCauchy = block.Cauchy_Stress[keyStep]

            stressDeviatoric = stressCauchy - 1. / 3. * np.trace(stressCauchy) * np.eye(3)

            Mises_Stress = np.sqrt(3. / 2. * np.sum(np.tensordot(stressDeviatoric, stressDeviatoric, axes = 2)))

            block.Mises_Stress[keyStep] = Mises_Stress

    setattr(
        domain, 
        'Mises_Stress',
        dict([(step, 0.0) for step in times]))

    for keyStep in times:

        stressCauchy = domain.Cauchy_Stress[keyStep]

        stressDeviatoric = stressCauchy - 1. / 3. * np.trace(stressCauchy) * np.eye(3)

        Mises_Stress = np.sqrt(3. / 2. * np.sum(np.tensordot(stressDeviatoric, stressDeviatoric, axes = 2)))

        block.Mises_Stress[keyStep] = Mises_Stress



    #
    # Calculate the deformed orientations
    #
    print 'Computing deformed orientations...'

    for keyBlock in domain.blocks:

        block = domain.blocks[keyBlock]

        for keyElement in block.elements:

            element = block.elements[keyElement]

            element.R = dict()
            element.U = dict()
            element.orientation = dict()

            for step in times:

                element.R[step], element.U[step] = polar(element.F[step])

                element.orientation[step] = np.inner(element.R[step], block.orientation.T)


    #
    # Plot the inverse pole figures
    #
    print 'Plotting inverse pole figures...'

    for step in times:

        plotInversePoleFigure(domain = domain, time = step) 



    #
    # Write data to exodus output file
    #
    print 'Writing data to exodus file...'

    if 'nameFileOutput' in kwargs:
        nameFileOutput = kwargs['nameFileOutput']
    else:
        nameFileOutput = nameFileBase + '_Postprocess.' + nameFileExtension
    
    writeExodusFile(domain, fileInput, nameFileOutput)

    fileInput.close()



    #
    # Plot stress-strain data
    #
    print 'Plotting stress-strain data...'

    plotStressStrain(domain)


    dataConverged, dataFailed = readFileLog(nameFileBase + '_Log.out')


    #
    # Return topology and data
    #
    return domain

# end def postprocess(nameFileInput, nameFileOutput):




if __name__ == '__main__':

    nameFileInput = sys.argv[1]

    if len(sys.argv) == 3:

        domain = postprocess(nameFileInput, nameFileOutput = sys.argv[2])

    else:

        domain = postprocess(nameFileInput)