#!/usr/bin/env python

import sys
import exodus
import string
import numpy
import matplotlib.pyplot as plt

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "\nUsage:  getStress.py <exodus_file_name>\n"
        sys.exit(1)

    inFileName = sys.argv[1]
    inFile = exodus.exodus(inFileName, mode='r')

    outFileLabel = string.splitfields(inFileName, '.')[0] + "_"

    # Print database parameters from inFile
    print " "
    print "Database version:         " + str(round(inFile.version.value,2))
    print "Database title:           " + inFile.title()
    print "Database dimensions:      " + str(inFile.num_dimensions())
    print "Number of nodes:          " + str(inFile.num_nodes())
    print "Number of elements:       " + str(inFile.num_elems())
    print "Number of element blocks: " + str(inFile.num_blks())
    print "Number of node sets:      " + str(inFile.num_node_sets())
    print "Number of side sets:      " + str(inFile.num_side_sets())
    print " "

    # Extract quantity at nodes or elements


    elementQuantities = ['Cauchy_Stress_09','tau_hard_1_1','gamma_1_1','Fp_09','F_09']
    # elementQuantities = ['gamma_1_1','Fp_01','Fp_02','Fp_03','Fp_04','Fp_05','Fp_06','Fp_07','Fp_08','Fp_09']
    # elementQuantities += ['Velocity_Gradient_Plastic_01','Velocity_Gradient_Plastic_02','Velocity_Gradient_Plastic_03',\
    #                      'Velocity_Gradient_Plastic_04','Velocity_Gradient_Plastic_05','Velocity_Gradient_Plastic_06',\
    #                      'Velocity_Gradient_Plastic_07','Velocity_Gradient_Plastic_08','Velocity_Gradient_Plastic_09']

    numNodes = inFile.num_nodes()
    numElements = inFile.num_elems()
    numTimeSteps = inFile.num_times()
    times = inFile.get_times()
    nodeVariableNames = inFile.get_node_variable_names()
    elementVariableNames = inFile.get_element_variable_names()

    #print elementVariableNames

    for name in elementQuantities:
        if name not in elementVariableNames:
            print "\nERROR:  Failed to extract " + name + "\n"
            sys.exit(1)

    data = numpy.zeros((numTimeSteps,len(elementQuantities)))

    for timeStep in range(numTimeSteps):
        for i in range(len(elementQuantities)):
            data[timeStep,i] = inFile.get_element_variable_values(1,elementQuantities[i], timeStep+1)[0]

    inFile.close()

    outFileName = outFileLabel + 'quantities.txt'
    dataFile = open(outFileName, 'w')

    s = '{0:30}'.format('time')
    dataFile.write(s)
    for element in elementQuantities:
        s = ' {0:30}'.format(element)
        dataFile.write(s)
    dataFile.write('\n')

    for timeStep in range(numTimeSteps):
        s = '{0:30.15e}'.format(times[timeStep])
        dataFile.write(s)

        for i in range(len(elementQuantities)):
            s = ' {0:30.15e}'.format(data[timeStep,i])
            dataFile.write(s)

        dataFile.write("\n")
    dataFile.close()
    print "stress data for written to", outFileName
    
    fig, ax = plt.subplots()
    ax.plot(times[:],data[:,0],color='blue',marker='o',label='1 elem/block')
    plt.xlabel('time (s)')
    plt.ylabel('stress (MPa)')
    # Guy -> let's keep the graphs clean
    #plt.xticks(numpy.arange(0,100,1))
    plt.grid()    
    lg = plt.legend(loc = 4)
    lg.draw_frame(False)
    plt.tight_layout()
    plt.show()
    fig.savefig('time_stress.pdf')
#
