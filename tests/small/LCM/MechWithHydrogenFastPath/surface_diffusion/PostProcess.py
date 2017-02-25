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
sys.path.append('/home/glbergel/LCM/trilinos-install-serial-gcc-release/bin')
import exodus
import string

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "\nUsage:  PostProcess.py <exodus_file_name>\n"
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

    # Extract nodal displacements and CLs
    numNodes = inFile.num_nodes()
    numTimeSteps = inFile.num_times()
    nodeVariableNames = inFile.get_node_variable_names()
    if 'disp_x' not in nodeVariableNames:
        print "\nERROR:  Failed to extract disp_x data\n"
        sys.exit(1)
    if 'disp_y' not in nodeVariableNames:
        print "\nERROR:  Failed to extract disp_y data\n"
        sys.exit(1)
    if 'disp_z' not in nodeVariableNames:
        print "\nERROR:  Failed to extract disp_z data\n"
        sys.exit(1)        
    if 'CL' not in nodeVariableNames:
        print "\nERROR:  Failed to extract CL data\n"
        sys.exit(1)
    if 'tauH' not in nodeVariableNames:
        print "\nERROR:  Failed to extract tauH data\n"
        sys.exit(1)        

    # Read node sets
    nodeSetIds = inFile.get_node_set_ids()
    nodeSetNodes = {}
    for nodeSetId in nodeSetIds:
        nodeIds = inFile.get_node_set_nodes(nodeSetId)
        nodeSetNodes[nodeSetId] = nodeIds[:]

    # In this particular case, we want to plot CL-displacement curves
    # where the CLs and displacements are on a specific node set

    nodeset_displacement_x = []
    nodeset_displacement_y = []
    nodeset_displacement_z = []
    nodeset_tauH = []
    nodeset_CL = []
    nodeset_Coords = []

    # parse data at final time step
    timeStep = numTimeSteps
    disp_x = inFile.get_node_variable_values('disp_x', timeStep)
    disp_y = inFile.get_node_variable_values('disp_y', timeStep)
    disp_z = inFile.get_node_variable_values('disp_z', timeStep)
    tauH =  inFile.get_node_variable_values('tauH', timeStep)
    CL = inFile.get_node_variable_values('CL', timeStep)
    Coords = inFile.get_coords()

    #nodeset
    nodeSet = nodeSetNodes[3] 
    displacement_x = 0.0
    displacement_y = 0.0
    displacement_z = 0.0
    tauH_ = 0.0
    CL_ = 0.0
    Coords_ = 0.0
    for nodeId in nodeSet:
        # in this case: save displacement, concentration, and pressure at each node
        displacement_x = disp_x[nodeId-1]
        displacement_y = disp_y[nodeId-1]
        displacement_z = disp_z[nodeId-1]
        tauH_ =  tauH[nodeId-1]
        CL_ = CL[nodeId-1]
        Coords_ = Coords[0][nodeId-1]
        nodeset_displacement_x.append(displacement_x)
        nodeset_displacement_y.append(displacement_y)
        nodeset_displacement_z.append(displacement_z)
        nodeset_tauH.append(tauH_)
        nodeset_CL.append(CL_)
        nodeset_Coords.append(Coords_)
            

    inFile.close()

    Count = 0
    outFileName = outFileLabel + 'CL_disp_tauH.txt'
    dataFile = open(outFileName, 'w')
    
#   only write data for last time step
    for nodeID in nodeSet:
         # convert to same units used in mathematica file
         # note: displacement is actually original coordinates! (I will change later)
         dataFile.write(str(nodeset_Coords[Count]*10.0**(-6)) + "  " + str(nodeset_CL[Count]*10**6) + " " + str(tauH[Count]) + "\n")
         Count += 1
    dataFile.close()
    print "CL-displacement data for written to", outFileName

    print
    
