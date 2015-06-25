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
#
# The script writes up to three files, one for each direction in three dimensionsion.
#    Only directions for which opposing nodeset are found, will have files written for them.
#    The files contain force, displacement, stress, strain, and eqiv plastic strain data,
#    depending on what it finds in the exodus database.

import sys
import os
path_home = os.environ.get('HOME')
print "Set home directory to " + path_home
sys.path.append(path_home + "/LCM/trilinos-install-gcc-release/bin")
import exodus
import string
from math import log

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print "\nUsage: " + os.path.basename(sys.argv[0]) + " <exodus_file_name> <output_file_name_base>\n"
        sys.exit(1)

    inFileName = sys.argv[1]
    outFileName = sys.argv[2]
    inFile = exodus.exodus(inFileName, mode='r')

    outFileLabel = string.splitfields(inFileName, '.')[0] + "_"

    # Print database parameters from inFile
    print " "
    print "Database version:         " + str(inFile.version.value)
    print "Database title:           " + inFile.title()
    print "Database dimensions:      " + str(inFile.num_dimensions())
    print "Number of nodes:          " + str(inFile.num_nodes())
    print "Number of elements:       " + str(inFile.num_elems())
    print "Number of element blocks: " + str(inFile.num_blks())
    print "Number of node sets:      " + str(inFile.num_node_sets())
    print "Number of side sets:      " + str(inFile.num_side_sets())
    print " "

    # Extract nodal displacements and forces
    numNodeSets = inFile.num_node_sets()
    numNodes = inFile.num_nodes()
    numElems = inFile.num_elems()
    numTimeSteps = inFile.num_times()
    nodeVariableNames = inFile.get_node_variable_names()
    elemVariableNames = inFile.get_element_variable_names()
    # if 'displacement_x' not in nodeVariableNames:
        # print "\nERROR:  Failed to extract displacement_x data\n"
        # sys.exit(1)
    # if 'force_x' not in nodeVariableNames:
        # print "\nERROR:  Failed to extract force_x data\n"
        # sys.exit(1)
    print
    displ_var_name = ["<NULL>" for x in range(3)]
    if 'displacement_x' in nodeVariableNames:
        displ_var_name[0] = "displacement_x"
        displ_var_name[1] = "displacement_y"
        displ_var_name[2] = "displacement_z"
	stres_var_base    = "Cauchy_Stress_"
	eqps__var_base    = "eqps_"
        fem_code = "LCM"
    elif 'DISPLACEMENT_X' in nodeVariableNames:
        displ_var_name[0] = "DISPLACEMENT_X"
        displ_var_name[1] = "DISPLACEMENT_Y"
        displ_var_name[2] = "DISPLACEMENT_Z"
	stres_var_base    = "CAUCHY_STRESS_"
	eqps__var_base    = "EQPS_"
        fem_code = "LCM"
    elif 'DISPLX' in nodeVariableNames:
        displ_var_name[0] = "DISPLX"
        displ_var_name[1] = "DISPLY"
        displ_var_name[2] = "DISPLZ"
	stres_var_base    = "SIG"
	eqps__var_base    = "EQPS"
        fem_code = "Jas3D"
    else:
        print "ERROR: Couldn't find displacement variable displacement_x or DISPLACEMENT_X or DISPLX."
        sys.exit(1)
    print "Found displacement variables " + displ_var_name[0] + ", " + displ_var_name[1] + ", " + displ_var_name[2]
    print "Assuming this output was created by " + fem_code

    force_var_name = ["<NULL>" for x in range(3)]
    if 'force_x' in nodeVariableNames:
        force_var_name[0] = "force_x"
        force_var_name[1] = "force_y"
        force_var_name[2] = "force_z"
	flag_found_force = 1
    if 'FORCE_X' in nodeVariableNames:
        force_var_name[0] = "FORCE_X"
        force_var_name[1] = "FORCE_Y"
        force_var_name[2] = "FORCE_Z"
	flag_found_force = 1
    elif 'FINTX' in nodeVariableNames:
        # Specifying FINT restricts us to displacement-controlled I think...
	#    Load control would probably need to use FEXT.
        force_var_name[0] = "FINTX"
        force_var_name[1] = "FINTY"
        force_var_name[2] = "FINTZ"
	flag_found_force = 1
    else:
        print "WARNING: Couldn't find force variables force_x or FORCE_X or FINTX."
        # sys.exit(1)
	flag_found_force = 0
    if flag_found_force:
        print "Found force variables " + force_var_name[0] + ", " + force_var_name[1] + ", " + force_var_name[2]

    if 'eqps_1' in elemVariableNames:
	flag_found_eqps = 1
    if 'EQPS_1' in elemVariableNames:
	flag_found_eqps = 1
    elif 'EQPS' in elemVariableNames:
	flag_found_eqps = 1
    else:
        print "WARNING: Couldn't find EQPS variable(s)."
        # sys.exit(1)
	flag_found_eqps = 0

    # Read time
    timeVals = inFile.get_times()

    # Read node sets
    nodeSetIds = inFile.get_node_set_ids()
    nodeSetNodes = {}
    for nodeSetId in nodeSetIds:
        nodeIDs = inFile.get_node_set_nodes(nodeSetId)
        nodeSetNodes[nodeSetId] = nodeIDs[:]

    # node_id_map = inFile.get_node_id_map()
    # for nodeID in range(numNodes):
    #     print str(node_id_map[nodeID])
    node_coord = {}
    for nodeNo in range(numNodes):
        nc = inFile.get_coord(nodeNo+1)
        node_coord[(nodeNo*3)+0] = nc[0]
        node_coord[(nodeNo*3)+1] = nc[1]
        node_coord[(nodeNo*3)+2] = nc[2]

    # Matrix = [[0 for x in range(15)] for x in range(5)]
    #   THIS MAKES Matrix[5][15], ***NOT*** Matrix[15][5]!!!
    nodeset_delta_DIR = {}
    nodeset_id_DIR_SID= [[0 for x in range(numNodeSets)] for x in range(3)]
    num_nsets_DIR = [0 for x in range(3)]
    for nodeSetId in nodeSetIds:
        nodeSet = nodeSetNodes[nodeSetId]
        for d in range(0, 3):
            nodeset_delta_DIR[d] = 0.0 
        for nodeID_1 in nodeSet:
            for nodeID_2 in nodeSet:
                for d in range(0, 3):
                    temp = node_coord[((nodeID_1-1)*3)+d] - node_coord[((nodeID_2-1)*3)+d]
                    nodeset_delta_DIR[d] += temp*temp
        for d in range(0, 3):
            if nodeset_delta_DIR[d] < 1E-6:
                nodeset_id_DIR_SID[d][num_nsets_DIR[d]] = nodeSetId
                num_nsets_DIR[d] += 1
    for d in range(0, 3):
        if d == 0:
            dir_text = "X"
        elif d == 1:
            dir_text = "Y"
        else:
            dir_text = "Z"
        sys.stdout.write("Found nodeset IDs for " + dir_text + " direction:  ")
        for i in range(num_nsets_DIR[d]):
            sys.stdout.write(str(nodeset_id_DIR_SID[d][i]) + " ")
        sys.stdout.write("\n")
        if num_nsets_DIR[d] > 2:
            print "  WARNING: Will only use the first two nodesets in " + str(d) + "-direction!"
    flag_found_nsets_DIR = [1 for x in range(3)]
    for d in range(0, 3):
        if num_nsets_DIR[d] < 2:
	    flag_found_nsets_DIR[d] = 0
    print
    temp = 0
    for d in range(0, 3):
        temp += flag_found_nsets_DIR[d]
    if not temp:
        print "ERROR: Couldn't find any opposing nodesets."
	sys.exit(2)

    # In this particular case, we want to plot force-displacement curves
    # where the forces and displacements are on a specific node set

    node_pos = {}

    # outFileName = outFileLabel + 'force_displacement.txt'
    if flag_found_eqps:
	eqps_header_text = "EQPS"
    else:
        eqps_header_text = "<NULL>"
    if flag_found_nsets_DIR[0]:
	dataFile_X = open(outFileName + "_X-dir.txt", 'w')
	dataFile_X.write("# [1]Step  [2]Time  [3]" + str(displ_var_name[0]) + "  [4]" + str(force_var_name[0]) + "  [5]TrueStrain_Elastic  [6]TrueStrain_Plastic  [7]TrueStrain_Total  [8]AvgCauchyStress  [9]" + eqps_header_text + "  [10]" + eqps_header_text + "/TrueStrain_Plastic\n")
    if flag_found_nsets_DIR[1]:
	dataFile_Y = open(outFileName + "_Y-dir.txt", 'w')
	dataFile_Y.write("# [1]Step  [2]Time  [3]" + str(displ_var_name[1]) + "  [4]" + str(force_var_name[1]) + "  [5]TrueStrain_Elastic  [6]TrueStrain_Plastic  [7]TrueStrain_Total  [8]AvgCauchyStress  [9]" + eqps_header_text + "  [10]" + eqps_header_text + "/TrueStrain_Plastic\n")
    if flag_found_nsets_DIR[2]:
	dataFile_Z = open(outFileName + "_Z-dir.txt", 'w')
	dataFile_Z.write("# [1]Step  [2]Time  [3]" + str(displ_var_name[2]) + "  [4]" + str(force_var_name[2]) + "  [5]TrueStrain_Elastic  [6]TrueStrain_Plastic  [7]TrueStrain_Total  [8]AvgCauchyStress  [9]" + eqps_header_text + "  [10]" + eqps_header_text + "/TrueStrain_Plastic\n")

    elastic_modulus_DIR = [0.0 for x in range(3)]
    displ_DIR = [[0.0 for x in range(numNodes)] for x in range(3)]
    force_DIR = [[0.0 for x in range(numNodes)] for x in range(3)]

    for timeStep in range(numTimeSteps):

        eqps = 0.0
	if flag_found_eqps:
            if fem_code == "Jas3D":
        	for elemNo in range(numElems):
                    varName = eqps__var_base
                    temp = inFile.get_element_variable_values(elemNo+1, varName, timeStep+1)
                    eqps += temp[0]
        	eqps /= numElems
            elif fem_code == "LCM":
        	for elemNo in range(numElems):
                    for gaussPt in range(1, 9):
                	varName = eqps__var_base + str(gaussPt)
                	temp = inFile.get_element_variable_values(elemNo+1, varName, timeStep+1)
                	eqps += temp[0]
        	eqps /= 8.0*numElems

        for d in range(0, 3):

            if not flag_found_nsets_DIR[d]:
	        continue

            displ_DIR[d] = inFile.get_node_variable_values(displ_var_name[d], timeStep+1)
	    if flag_found_force:
                force_DIR[d] = inFile.get_node_variable_values(force_var_name[d], timeStep+1)

            for nodeNo in range(numNodes):
                node_pos[(nodeNo*3)+d] = node_coord[(nodeNo*3)+d] + displ_DIR[d][nodeNo]

            stress = 0.0
            if fem_code == "Jas3D":
                if d == 0:
                    dir_text = "XX"
                elif d == 1:
                    dir_text = "YY"
                else:
                    dir_text = "ZZ"
                for elemNo in range(numElems):
                    varName = stres_var_base + dir_text
                    temp = inFile.get_element_variable_values(elemNo+1, varName, timeStep+1)
                    stress += temp[0]
                stress /= numElems
            elif fem_code == "LCM":
                if d == 0:
                    cauchyIDs = [ "01", "10", "19", "28", "37", "46", "55", "64" ]
                elif d == 1:
                    cauchyIDs = [ "05", "14", "23", "32", "41", "50", "59", "68" ]
                else:
                    cauchyIDs = [ "09", "18", "27", "36", "45", "54", "63", "72" ]
                for elemNo in range(numElems):
                    for cauchyID in cauchyIDs:
                        varName = stres_var_base + cauchyID
                        temp = inFile.get_element_variable_values(elemNo+1, varName, timeStep+1)
                        stress += temp[0]
                stress /= 8.0*numElems

            if d == 0:
                dir_text = "X"
            elif d == 1:
                dir_text = "Y"
            else:
                dir_text = "Z"

            nodeSet = nodeSetNodes[nodeset_id_DIR_SID[d][0]]
            coord_1 = 0.0
            displ_1 = 0.0
            force_1 = 0.0
            for nodeID in nodeSet:
                coord_1 += node_coord[((nodeID-1)*3)+d]
                displ_1 += - displ_DIR[d][nodeID-1]
                force_1 += - force_DIR[d][nodeID-1]
            coord_1 /= len(nodeSetNodes[nodeset_id_DIR_SID[d][0]])
            displ_1 /= len(nodeSetNodes[nodeset_id_DIR_SID[d][0]])

            nodeSet = nodeSetNodes[nodeset_id_DIR_SID[d][1]]
            coord_2 = 0.0
            displ_2 = 0.0
            force_2 = 0.0
            for nodeID in nodeSet:
                coord_2 += node_coord[((nodeID-1)*3)+d]
                displ_2 += displ_DIR[d][nodeID-1]
                force_2 += force_DIR[d][nodeID-1]
            coord_2 /= len(nodeSetNodes[nodeset_id_DIR_SID[d][1]])
            displ_2 /= len(nodeSetNodes[nodeset_id_DIR_SID[d][1]])

            displ = displ_1 + displ_2
            lengt = coord_2 - coord_1
            force = (force_1 + force_2)/2.0

            strain = log((lengt + displ)/lengt)

            # Try to figure out what the elastic modulus is.
            strain_elastic = strain_plastic = 0.0
            if timeStep == 0:
                elastic_strain_1 = strain
                elastic_stress_1 = stress
                strain_elastic = strain
                strain_plastic = strain - strain_elastic
            elif timeStep == 1:
                elastic_strain_2 = strain
                elastic_stress_2 = stress
                elastic_modulus_DIR[d]  = (elastic_stress_2 - elastic_stress_1)/(elastic_strain_2 - elastic_strain_1)
                print "Found " + dir_text + "-direction elastic modulus = " + str(elastic_modulus_DIR[d])
                strain_elastic = stress/elastic_modulus_DIR[d]
                strain_plastic = strain - strain_elastic
            else:
                strain_elastic = stress/elastic_modulus_DIR[d]
                strain_plastic = strain - strain_elastic

            if strain_plastic < 0.0:
                strain_plastic = 0.0
            if strain_plastic == 0.0:
                eqps_strain_ratio = 0.0
            else:
                eqps_strain_ratio = eqps/strain_plastic

            dataFile = dataFile_X
            if d == 1:
              dataFile = dataFile_Y
            elif d == 2:
              dataFile = dataFile_Z
            dataFile.write(str(timeStep+1) + " " + str(timeVals[timeStep]) + " " + str(displ) + "  " + str(force) + "  " + str(strain_elastic) + "  " + str(strain_plastic) + "  " + str(strain) + "  " + str(stress) + "  " + str(eqps) + "  " + str(eqps_strain_ratio) + "\n")

    if flag_found_nsets_DIR[0]:
	dataFile_X.close()
    if flag_found_nsets_DIR[1]:
	dataFile_Y.close()
    if flag_found_nsets_DIR[2]:
	dataFile_Z.close()

    print
    inFile.close()

    print
    print "Force-displacement data for written to " + outFileName + "_[XYZ]-dir.txt"
    print
