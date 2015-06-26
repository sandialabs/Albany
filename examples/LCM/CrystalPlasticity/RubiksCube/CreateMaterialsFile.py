#!/usr/bin/env python

import sys
import string

INDENTATION = 2

def ParseMaterialParametersFile(file_name, mat_params):

    print "\nReading material parameters from", file_name

    mat_params_file = open(file_name, mode='r')
    raw_data = mat_params_file.readlines()
    mat_params_file.close()
    for param in raw_data:
        vals = string.splitfields(param)
        if len(vals) == 2:
            name = vals[0]
            value = vals[1]
            print " ", name, value
            if name in mat_params.keys():
                mat_params[name] = value
            else:
                print "\n**** Error, unexpected material parameter:", name
                print "\n     Allowable material parameters:"
                for key in mat_params.keys():
                    print "      ", key
                print
                sys.exit(1)
    for key in mat_params.keys():
        if mat_params[key] == None:
            print "\n**** Error, missing material parameter:", key
            print "\n     Expected material parameters:"
            for key in mat_params.keys():
                print "      ", key
            print
            sys.exit(1)

    return

def ParseRotationMatricesFile(file_name, rotations):

    print "\nReading rotation matrices from", file_name

    rotations_file = open(file_name, mode='r')
    raw_data = rotations_file.readlines()
    rotations_file.close()
    for matrix in raw_data:
        vals = string.splitfields(matrix)
        if len(vals) == 9:
            rot = [[float(vals[0]), float(vals[1]), float(vals[2])],
                   [float(vals[3]), float(vals[4]), float(vals[5])],
                   [float(vals[6]), float(vals[7]), float(vals[8])]]
            rotations.append(rot)

    print "  read", len(rotations), "rotation matrices"

    return

def ConstructBasisVectors(R):

    e1 = [1.0, 0.0, 0.0]
    e2 = [0.0, 1.0, 0.0]
    e3 = [0.0, 0.0, 1.0]

    BasisVector1 = [0.0, 0.0, 0.0]
    BasisVector2 = [0.0, 0.0, 0.0]
    BasisVector3 = [0.0, 0.0, 0.0]
    
    for i in range(3):
        for j in range(3):
            BasisVector1[i] += R[i][j]*e1[j]
            BasisVector2[i] += R[i][j]*e2[j]
            BasisVector3[i] += R[i][j]*e3[j]

    return (BasisVector1, BasisVector2, BasisVector3)

def VectorToString(vec):

    vec_string = "{" + str(vec[0]) + ", " + str(vec[1]) + ", " + str(vec[2]) + "}"
    return vec_string

def FCCSlipSystems():

    slip_systems = []

    # system 1
    direction = (-1.0, 1.0, 0.0)
    normal = (1.0, 1.0, 1.0)
    slip_systems.append((direction, normal))

    # system 2
    direction = (0.0, -1.0, 1.0)
    normal = (1.0, 1.0, 1.0)
    slip_systems.append((direction, normal))

    # system 3
    direction = (1.0, 0.0, -1.0)
    normal = (1.0, 1.0, 1.0)
    slip_systems.append((direction, normal))

    # system 4
    direction = (-1.0, -1.0, 0.0)
    normal = (-1.0, 1.0, 1.0)
    slip_systems.append((direction, normal))

    # system 5
    direction = (1.0, 0.0, 1.0)
    normal = (-1.0, 1.0, 1.0)
    slip_systems.append((direction, normal))

    # system 6
    direction = (0.0, 1.0, -1.0)
    normal = (-1.0, 1.0, 1.0)
    slip_systems.append((direction, normal))

    # system 7
    direction = (1.0, -1.0, 0.0)
    normal = (-1.0, -1.0, 1.0)
    slip_systems.append((direction, normal))

    # system 8
    direction = (0.0, 1.0, 1.0)
    normal = (-1.0, -1.0, 1.0)
    slip_systems.append((direction, normal))

    # system 9
    direction = (-1.0, 0.0, -1.0)
    normal = (-1.0, -1.0, 1.0)
    slip_systems.append((direction, normal))

    # system 10
    direction = (1.0, 1.0, 0.0)
    normal = (1.0, -1.0, 1.0)
    slip_systems.append((direction, normal))

    # system 11
    direction = (-1.0, 0.0, 1.0)
    normal = (1.0, -1.0, 1.0)
    slip_systems.append((direction, normal))

    # system 12
    direction = (0.0, -1.0, -1.0)
    normal = (1.0, -1.0, 1.0)
    slip_systems.append((direction, normal))
    
    return slip_systems

def StartParamList(param_list_name, file, indent):

    file.write(" "*INDENTATION*indent)
    file.write("<ParameterList")
    if param_list_name != None:
        file.write(" name=\"" + param_list_name + "\"")
    file.write(">\n")
    indent += 1
    
    return indent

def EndParamList(file, indent):

    indent -= 1
    file.write(" "*INDENTATION*indent)
    file.write("</ParameterList>\n")
    return indent

def WriteParameter(name, type, value, file, indent):

    file.write(" "*INDENTATION*indent)
    file.write("<Parameter name=\"" + name + "\" type=\"" + type + "\" value=\"" + str(value) + "\"/>\n")
    return

def WriteBool(name, type, value, file, indent):

    file.write(" "*INDENTATION*indent)
    file.write("<Parameter name=\"" + name + "\" type=\"" + type + "\" value=\"" + value + "\"/>\n")
    return


def WriteMaterialsFile(file_name, mat_params, rotations, num_blocks):

    block_names = []
    material_names = []
    for i in range(num_blocks):
        block_names.append("block_" + str(i+1))
        material_names.append("Block " + str(i+1) + " Material")

    indent = 0

    mat_file = open(file_name, mode='w')
    indent = StartParamList(None, mat_file, indent)

    # Associate a material model with each block
    indent = StartParamList("ElementBlocks", mat_file, indent)
    for iBlock in range(num_blocks):
        indent = StartParamList(block_names[iBlock], mat_file, indent)
        WriteParameter("material", "string", material_names[iBlock], mat_file, indent)
        WriteBool("Weighted Volume Average J", "bool", "true", mat_file, indent)
        WriteBool("Volume Average Pressure", "bool", "true", mat_file, indent)
        indent = EndParamList(mat_file, indent)
    indent = EndParamList(mat_file, indent)

    # Give the material model parameters for each material
    indent = StartParamList("Materials", mat_file, indent)
    for iBlock in range(num_blocks):
        indent = StartParamList(material_names[iBlock], mat_file, indent)

        # Obtain the specifics for this grain
        (vec1, vec2, vec3) = ConstructBasisVectors(rotations[iBlock])
        slip_systems = FCCSlipSystems()
        num_slip_systems = len(slip_systems)

        # Material model name
        indent = StartParamList("Material Model", mat_file, indent)
        WriteParameter("Model Name", "string", "CrystalPlasticity", mat_file, indent)
        indent = EndParamList(mat_file, indent)

        # Integration scheme
        WriteParameter("Integration Scheme", "string", mat_params["integration_scheme"], mat_file, indent)
        if mat_params["integration_scheme"] == "Implicit":
            WriteParameter("Implicit Integration Relative Tolerance", "double", mat_params["implicit_integration_relative_tolerance"], mat_file, indent)
            WriteParameter("Implicit Integration Absolute Tolerance", "double", mat_params["implicit_integration_absolute_tolerance"], mat_file, indent)
            WriteParameter("Implicit Integration Max Iterations", "int", mat_params["implicit_integration_max_iterations"], mat_file, indent)

        # Specify output to exodus
        WriteParameter("Output Cauchy Stress", "bool", "true", mat_file, indent)
        WriteParameter("Output Fp", "bool", "true", mat_file, indent)
        #WriteParameter("Output L", "bool", "true", mat_file, indent)
        WriteParameter("Output CP_Residual", "bool", "true", mat_file, indent)
        #for i in range(num_slip_systems):
        #    WriteParameter("Output tau_" + str(i+1), "bool", "true", mat_file, indent)
        #for i in range(num_slip_systems):
        #    WriteParameter("Output tau_hard_" + str(i+1), "bool", "true", mat_file, indent)
        #for i in range(num_slip_systems):
        #    WriteParameter("Output gamma_" + str(i+1), "bool", "true", mat_file, indent)

        for i in range(1):
            WriteParameter("Output tau_" + str(i+1), "bool", "true", mat_file, indent)
        for i in range(1):
            WriteParameter("Output tau_hard_" + str(i+1), "bool", "true", mat_file, indent)
        for i in range(1):
            WriteParameter("Output gamma_" + str(i+1), "bool", "true", mat_file, indent)

        # Elastic modulii and lattice orientation
        indent = StartParamList("Crystal Elasticity", mat_file, indent)
        WriteParameter("C11", "double", mat_params["C11"], mat_file, indent)
        WriteParameter("C12", "double", mat_params["C12"], mat_file, indent)
        WriteParameter("C44", "double", mat_params["C44"], mat_file, indent)
        WriteParameter("Basis Vector 1", "Array(double)", VectorToString(vec1), mat_file, indent)
        WriteParameter("Basis Vector 2", "Array(double)", VectorToString(vec2), mat_file, indent)
        WriteParameter("Basis Vector 3", "Array(double)", VectorToString(vec3), mat_file, indent)
        indent = EndParamList(mat_file, indent)        

        # Crystal plasticity slip systems
        WriteParameter("Number of Slip Systems", "int", len(slip_systems), mat_file, indent)
        for iSS in range(num_slip_systems):
            direction = slip_systems[iSS][0]
            normal = slip_systems[iSS][1]
            indent = StartParamList("Slip System " + str(iSS+1), mat_file, indent)
            WriteParameter("Slip Direction", "Array(double)", VectorToString(direction), mat_file, indent)
            WriteParameter("Slip Normal", "Array(double)", VectorToString(normal), mat_file, indent)
            WriteParameter("Tau Critical", "double", mat_params["tau_critical"], mat_file, indent)
            WriteParameter("Gamma Dot", "double", mat_params["gamma_dot"], mat_file, indent)
            WriteParameter("Gamma Exponent", "double", mat_params["gamma_exponent"], mat_file, indent)
            WriteParameter("Hardening", "double", mat_params["hardening"], mat_file, indent)
            WriteParameter("Hardening Exponent", "double", mat_params["hardening_exponent"], mat_file, indent)
            indent = EndParamList(mat_file, indent)

        indent = EndParamList(mat_file, indent)

    indent = EndParamList(mat_file, indent)
    indent = EndParamList(mat_file, indent)
    mat_file.close()

    print "\nMaterials file written to", file_name

    return

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print "\nUsage:  CreateMaterialsFile.py <mat_props.txt> <rotation_matrices.txt> <num_blocks>\n"
        sys.exit(1)

    mat_params_file_name = sys.argv[1]
    rotations_file_name = sys.argv[2]
    num_blocks = int(sys.argv[3])

    # List of material parameters that are expected to be in the input file
    # If it's set to None, then it is a required parameter
    mat_params = {}
    mat_params["C11"] = None
    mat_params["C12"] = None
    mat_params["C44"] = None
    mat_params["tau_critical"] = None
    mat_params["gamma_dot"] = None
    mat_params["gamma_exponent"] = None
    mat_params["hardening"] = None
    mat_params["hardening_exponent"] = None
    mat_params["integration_scheme"] = None
    mat_params["implicit_integration_relative_tolerance"] = "unspecified"
    mat_params["implicit_integration_absolute_tolerance"] = "unspecified"
    mat_params["implicit_integration_max_iterations"] = "unspecified"

    ParseMaterialParametersFile(mat_params_file_name, mat_params)

    rotations = []
    ParseRotationMatricesFile(rotations_file_name, rotations)

    materials_file_name = "Materials.xml"
    WriteMaterialsFile(materials_file_name, mat_params, rotations, num_blocks)

    print "\nComplete.\n"
