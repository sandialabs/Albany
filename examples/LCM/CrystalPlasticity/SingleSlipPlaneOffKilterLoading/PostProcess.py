#!/usr/bin/env python

# This script requires exodus.py
# It can be tricky to get exodus.py to work because it requires libnetcdf.so and libexodus.so
# Here's one approach that worked on the CEE LAN:
# 1) Append sys.path, as shown below, to include the bin subdirectory of your Trilinos install directory.
#    The file exodus.py is in this directory.
# 2) Edit exodus.py as follows (approximately line 71):
#    accessPth = "/projects/seacas/linux_rhel6/current"
#    The path above is valid on the CEE LAN.  On other systems, you need to provide a path to a SEACAS build
#    that includes shared libraries.
import sys
sys.path.append('/ascldap/users/djlittl/Albany_TPL/trilinos/trilinos-votd/GCC_4.7.2_OPT/bin')
import exodus
import math

def compute_principal_stresses(stress):

    sigma_1 = 0.0
    sigma_2 = 0.0
    sigma_3 = 0.0

    s11 = stress[0]
    s22 = stress[4]
    s33 = stress[8]
    s12 = stress[1]
    s23 = stress[5]
    s31 = stress[6]

    I1 = s11 + s22 + s33
    I2 = s11*s22 + s22*s33 + s33*s11 - s12*s12 - s23*s23 - s31*s31
    I3 = s11*s22*s33 - s11*s23*s23 - s22*s31*s31 - s33*s12*s12 + 2.0*s12*s23*s31
    denominator = 2.0*math.pow(I1*I1 - 3.0*I2, 3.0/2.0)
    if math.fabs(denominator) > 1.0e-30:
        acos_arg = (2.0*I1*I1*I1 - 9.0*I1*I2 + 27.0*I3) / denominator
        if acos_arg < -1:
            acos_arg = -1
        if acos_arg > 1:
            acos_arg = 1
        phi = (1.0/3.0) * math.acos(acos_arg)
        sigma_1 = I1/3.0 + (2.0/3.0)*math.sqrt(I1*I1 - 3.0*I2)*math.cos(phi)
        sigma_2 = I1/3.0 + (2.0/3.0)*math.sqrt(I1*I1 - 3.0*I2)*math.cos(phi + 2.0*math.pi/3.0)
        sigma_3 = I1/3.0 + (2.0/3.0)*math.sqrt(I1*I1 - 3.0*I2)*math.cos(phi + 4.0*math.pi/3.0)

    return (sigma_1, sigma_2, sigma_3)

if __name__ == "__main__":

    inFile = exodus.exodus('SingleSlipPlaneOffKilterLoading.exo', mode='r')

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

    # Extract Cauchy stresses
    numElem = inFile.num_elems()
    numTimeSteps = inFile.num_times()
    times = inFile.get_times()
    elementVariableNames = inFile.get_element_variable_names()

    numGaussPoints = 8

    for iElem in range(numElem):
        for iGP in range(numGaussPoints):
            for iTensor in range(9):
                index = iTensor + 9*iGP + 1
                if index < 10:
                    gaussPointIdentifier = "0" + str(index)
                else:
                    gaussPointIdentifier = str(index)
                stressName = "Cauchy_Stress_" + gaussPointIdentifier
                if stressName not in elementVariableNames:
                    print "\nERROR:  Failed to extract Cauchy_Stress data\n"
                    sys.exit(1)

    element_1_stresses = []
    element_2_stresses = []

    for timeStep in range(numTimeSteps):

        element_1_stress = [0.0]*(numGaussPoints*9)
        element_2_stress = [0.0]*(numGaussPoints*9)

        for iGP in range(numGaussPoints):
            for iTensor in range(9):
                index = iTensor + 9*iGP + 1
                if index < 10:
                    gaussPointIdentifier = "0" + str(index)
                else:
                    gaussPointIdentifier = str(index)
                stressName = "Cauchy_Stress_" + gaussPointIdentifier

                blockId = 1
                stress_block_1 = inFile.get_element_variable_values(blockId, stressName, timeStep+1)
                blockId = 2
                stress_block_2 = inFile.get_element_variable_values(blockId, stressName, timeStep+1)
                element_1_stress[index-1] = stress_block_1[0]
                element_2_stress[index-1] = stress_block_2[0]

        element_1_stresses.append(element_1_stress)
        element_2_stresses.append(element_2_stress)

    inFile.close()

    for iTimeStep in range(len(element_1_stresses)):
        print "Cauchy stress at time step", iTimeStep+1
        stress = element_1_stresses[iTimeStep]
        for iGP in range(numGaussPoints):
            stress_str = "Elem 1 ("
            for iTensor in range(9):
                stress_val = stress[9*iGP + iTensor]
                stress_str += "{0:.4g}".format(stress_val)
                if iTensor < 8:
                    stress_str += ", "
                else:
                    stress_str += ")"
            print stress_str
        stress = element_2_stresses[iTimeStep]
        for iGP in range(numGaussPoints):
            stress_str = "Elem 2 ("
            for iTensor in range(9):
                stress_val = stress[9*iGP + iTensor]
                stress_str += "{0:.4g}".format(stress_val)
                if iTensor < 8:
                    stress_str += ", "        
                else:
                    stress_str += ")"
            print stress_str
        print

    # Plot the maximum principal stress for the first Gauss point over time
    max_principal_stress_element_1 = []
    max_principal_stress_element_2 = []
    for iTimeStep in range(len(times)):
        msg = "Time = " + "{0:.4f}".format(times[iTimeStep])
        stress = element_1_stresses[iTimeStep][0:9]
        sigma_1, sigma_2, sigma_3 = compute_principal_stresses(stress)
        max_principal_stress_element_1.append(sigma_1)
        msg += ", principal stresses element 1 = (" + "{0:.4g}".format(sigma_1) + ", " + "{0:.4g}".format(sigma_2) + ", " + "{0:.4g}".format(sigma_3) + ")"
        stress = element_2_stresses[iTimeStep][0:9]
        sigma_1, sigma_2, sigma_3 = compute_principal_stresses(stress)
        max_principal_stress_element_2.append(sigma_1)
        msg += ", principal stresses element 2 = (" + "{0:.4g}".format(sigma_1) + ", " + "{0:.4g}".format(sigma_2) + ", " + "{0:.4g}".format(sigma_3) + ")"
        print msg

    outFile = open("max_principal_stress_versus_time.txt", 'w')
    for i in range(len(times)):
        outFile.write(str(times[i]) + "  " + str(max_principal_stress_element_1[i]) + "  " + str(max_principal_stress_element_2[i]) + "\n")
    outFile.close()
    print "\nData written to max_principal_stress_versus_time.txt\n"
