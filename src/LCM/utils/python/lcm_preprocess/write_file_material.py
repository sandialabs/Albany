#!/usr/bin/env python

import os
import sys
import string
from create_slip_systems import create_slip_systems
from exodus import exodus
from lcm_postprocess._core import InputError

INDENTATION = 2

def create_mat_params_default():

    mat_params = {}

    mat_params["verbosity"] = "None"

    mat_params["orientation_on_mesh"] = "unspecified"

    mat_params["crystal_structure"] = "fcc"
    mat_params["slip_families"] = "unspecified"
    mat_params["ratio_c_a"] = "unspecified"

    mat_params["C11"] = "unspecified"
    mat_params["C12"] = "unspecified"
    mat_params["C13"] = "unspecified"
    mat_params["C33"] = "unspecified"
    mat_params["C44"] = "unspecified"

    mat_params["M11"] = "unspecified"
    mat_params["M12"] = "unspecified"
    mat_params["M13"] = "unspecified"
    mat_params["M33"] = "unspecified"
    mat_params["M44"] = "unspecified"

    mat_params["temperature_initial"] = "unspecified"
    mat_params["temperature_reference"] = "unspecified"    

    # Flow rule parameters
    mat_params["flow_rule"] = "unspecified"                    # "Flow Rule" POWER_LAW, THERMAL_ACTIVATION, or POWER_LAW_DRAG
    mat_params["rate_slip_reference"] = "unspecified"          # "Reference Slip Rate" power_law  power_law_drag  thermal_activation  and saturation hardening law
    mat_params["exponent_rate"] = "unspecified"                # "Rate Exponent"       power_law  power_law_drag
    mat_params["drag_coeff"] = "unspecified"                   # "Drag Coefficient"               power_law_drag
    mat_params["energy_activation"] = "unspecified"            # "Activation Energy"                              thermal_activation
    mat_params["resistance_thermal"] = "unspecified"           # "Thermal Resistance"                             thermal_activation
    mat_params["exponent_p"] = "unspecified"                   # "P Exponent"                                     thermal_activation
    mat_params["exponent_q"] = "unspecified"                   # "Q Exponent"                                     thermal_activation

    # Hardening law parameters
    mat_params["hardening_law"] = "unspecified"                # "Hardening Rule" LINEAR_MINUS_RECOVERY, SATURATION, or DISLOCATION_DENSITY
    mat_params["modulus_hardening"] = "unspecified"            # "Hardening Modulus"       linear_minus_recovery
    mat_params["modulus_recovery"] = "unspecified"             # "Recovery Modulus"        linear_minus_recovery
    mat_params["state_hardening_initial"] = "unspecified"      # "Initial Hardening State" linear_minus_recovery  saturation  dislocation_density
    mat_params["rate_hardening"] = "unspecified"               # "Hardening Rate"                                 saturation
    mat_params["stress_saturation_initial"] = "unspecified"    # "Initial Saturation Stress"                      saturation
    mat_params["exponent_saturation"] = "unspecified"          # "Saturation Exponent"                            saturation
    mat_params["factor_geometric"] = "unspecified"             # "Geometric Factor"                                           dislocation_density
    mat_params["factor_generation"] = "unspecified"            # "Generation Factor"                                          dislocation_density
    mat_params["factor_annihilation"] = "unspecified"          # "Annihilation Factor"                                        dislocation_density
    mat_params["magnitude_burgers"] = "unspecified"            # "Burgers Vector Magnitute"                                   dislocation_density
    mat_params["modulus_shear"] = "unspecified"                # "Shear Modulus"                                              dislocation_density

    mat_params["integration_scheme"] = "unspecified"
    mat_params["type_residual"] = "unspecified"
    mat_params["nonlinear_solver_step_type"] = "unspecified"
    mat_params["implicit_integration_relative_tolerance"] = "unspecified"
    mat_params["implicit_integration_absolute_tolerance"] = "unspecified"
    mat_params["implicit_integration_max_iterations"] = "unspecified"
    mat_params["predictor_slip"] = "unspecified"

    return mat_params

def create_vars_output_default():

    vars_output = {}
    vars_output["cauchy_stress"] = "true"
    vars_output["F"] = "true"
    vars_output["integration_weights"] = "true"
    vars_output["Lp"] = "true"
    vars_output["Fp"] = "unspecified"
    vars_output["L"] = "unspecified"
    vars_output["eqps"] = "unspecified"
    vars_output["gamma"] = "unspecified"
    vars_output["gamma_dot"] = "unspecified"
    vars_output["tau"] = "unspecified"
    vars_output["tau_hard"] = "unspecified"
    vars_output["cp_residual"] = "unspecified"
    vars_output["cp_residual_iter"] = "unspecified"

    return vars_output

def read_names_block(name_file_exodus):

    file_exodus = exodus(name_file_exodus)
    ids_block = file_exodus.get_elem_blk_ids()
    names_block = file_exodus.get_elem_blk_names()
    file_exodus.close()

    for idx in range(len(ids_block)):
        if (names_block[idx] == ""):
            names_block[idx] = "block_" + str(ids_block[idx])

    return names_block

def read_element_type(name_file_exodus):

    file_exodus = exodus(name_file_exodus)
    ids_block = file_exodus.get_elem_blk_ids()
    element_block_info = file_exodus.elem_blk_info(ids_block[0])
    element_type = element_block_info[0]
    file_exodus.close() 
    return element_type

def ParseMaterialParametersFile(file_name, mat_params, vars_output):

    print "\nReading material parameters from", file_name

    mat_params_file = open(file_name, mode='r')
    raw_data = mat_params_file.readlines()
    mat_params_file.close()
    for param in raw_data:
        vals = string.splitfields(param)

        print vals
        if len(vals) > 1:
            name = vals[0]
            value = vals[1]
            for val in vals[2:]:
                value = value + " " + val
        
        print " ", name, value
        if name in mat_params.keys():
            mat_params[name] = value
        elif name in vars_output.keys():
            vars_output[name] = value
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

def ParseRotationMatricesFile(file_name):

    print "\nReading rotation matrices from", file_name

    rotations = []
    rotations_file = open(file_name, mode = 'r')
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

    return rotations

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

def StartParamList(param_list_name, file, indent):

    file.write("\n")
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

    if value != 'unspecified':
        file.write(" "*INDENTATION*indent)
        file.write("<Parameter name=\"" + name + "\" type=\"" + type + "\" value=\"" + str(value) + "\"/>\n")
    return

def WriteBool(name, type, value, file, indent):

    file.write(" "*INDENTATION*indent)
    file.write("<Parameter name=\"" + name + "\" type=\"" + type + "\" value=\"" + value + "\"/>\n")
    return


def WriteMaterialsFile(file_name, mat_params, vars_output, rotations, names_block, element_type):

    num_blocks = len(names_block)
    material_names = []
    for name in names_block:
        material_names.append(name + " Material")

    indent = 0

    mat_file = open(file_name, mode = 'w')
    indent = StartParamList(None, mat_file, indent)

    # Associate a material model with each block
    indent = StartParamList("ElementBlocks", mat_file, indent)
    for iBlock in range(num_blocks):
        indent = StartParamList(names_block[iBlock], mat_file, indent)
        WriteParameter("material", "string", material_names[iBlock], mat_file, indent)
        WriteBool("Weighted Volume Average J", "bool", "true", mat_file, indent)
        WriteBool("Volume Average Pressure", "bool", "true", mat_file, indent)
        if element_type == "TETRA10":
            WriteBool("Use Composite Tet 10", "bool", "true", mat_file, indent)
        indent = EndParamList(mat_file, indent)
    indent = EndParamList(mat_file, indent)

    # Give the material model parameters for each material
    indent = StartParamList("Materials", mat_file, indent)
    for iBlock in range(num_blocks):
        mat_file.write("\n")
        indent = StartParamList(material_names[iBlock], mat_file, indent)

        # Obtain the specifics for this grain
        (vec1, vec2, vec3) = ConstructBasisVectors(rotations[iBlock])

        slip_systems = create_slip_systems(
            crystal_structure = mat_params["crystal_structure"],
            slip_families = mat_params["slip_families"],
            ratio_c_a = mat_params["ratio_c_a"])

        # Material model name
        indent = StartParamList("Material Model", mat_file, indent)
        WriteParameter("Model Name", "string", "CrystalPlasticity", mat_file, indent)
        indent = EndParamList(mat_file, indent)

        # Output verbosity
        WriteParameter("Verbosity", "string", mat_params["verbosity"], mat_file, indent)

        WriteParameter("Read Lattice Orientation From Mesh", "bool", mat_params["orientation_on_mesh"], mat_file, indent)

        # Number of slip systems
        num_slip_systems = len(slip_systems)
        if len(slip_systems) > 0:
            WriteParameter("Number of Slip Systems", "int", num_slip_systems, mat_file, indent) 

        # Integration scheme
        WriteParameter("Integration Scheme", "string", mat_params["integration_scheme"], mat_file, indent)
        if mat_params["integration_scheme"] == "Implicit":
            WriteParameter("Nonlinear Solver Step Type", "string", mat_params["nonlinear_solver_step_type"], mat_file, indent) 
            WriteParameter("Implicit Integration Relative Tolerance", "double", mat_params["implicit_integration_relative_tolerance"], mat_file, indent)
            WriteParameter("Implicit Integration Absolute Tolerance", "double", mat_params["implicit_integration_absolute_tolerance"], mat_file, indent)
            WriteParameter("Implicit Integration Max Iterations", "int", mat_params["implicit_integration_max_iterations"], mat_file, indent)
            if mat_params["type_residual"] == "Slip Hardness":
                WriteParameter("Residual Type", "string", mat_params["type_residual"], mat_file, indent)
            if mat_params["predictor_slip"] != "unspecified":
                WriteParameter("Slip Predictor", "string", mat_params["predictor_slip"], mat_file, indent)

        # Specify output to exodus
        WriteParameter("Output Cauchy Stress", "bool", vars_output["cauchy_stress"], mat_file, indent)
        WriteParameter("Output Fp", "bool", vars_output["Fp"], mat_file, indent)
        WriteParameter("Output Deformation Gradient", "bool", vars_output["F"], mat_file, indent)
        WriteParameter("Output L", "bool", vars_output["L"], mat_file, indent)
        WriteParameter("Output Lp", "bool", vars_output["Lp"], mat_file, indent)
        WriteParameter("Output CP_Residual", "bool", vars_output["cp_residual"], mat_file, indent)
        WriteParameter("Output CP_Residual_Iter", "bool", vars_output["cp_residual_iter"], mat_file, indent)
        WriteParameter("Output eqps", "bool", vars_output["eqps"], mat_file, indent)
        WriteParameter("Output Integration Weights", "bool", vars_output["integration_weights"], mat_file, indent)
        for i in range(num_slip_systems):
           WriteParameter("Output tau_" + str(i+1), "bool", vars_output["tau"], mat_file, indent)
        for i in range(num_slip_systems):
           WriteParameter("Output tau_hard_" + str(i+1), "bool", vars_output["tau_hard"], mat_file, indent)
        for i in range(num_slip_systems):
           WriteParameter("Output gamma_" + str(i+1), "bool", vars_output["gamma"], mat_file, indent)
        for i in range(num_slip_systems):
           WriteParameter("Output gamma_dot_" + str(i+1), "bool", vars_output["gamma_dot"], mat_file, indent)

        # Elastic modulii and lattice orientation
        indent = StartParamList("Crystal Elasticity", mat_file, indent)
        WriteParameter("C11", "double", mat_params["C11"], mat_file, indent)
        WriteParameter("C12", "double", mat_params["C12"], mat_file, indent)
        WriteParameter("C44", "double", mat_params["C44"], mat_file, indent)

        # Temperature-dependence properties
        M11 = mat_params.get("M11", None)
        if M11 != None:
            WriteParameter("M11", "double", M11, mat_file, indent)
        M12 = mat_params.get("M12", None)
        if M12 != None:
            WriteParameter("M12", "double", M12, mat_file, indent)
        M44 = mat_params.get("M44", None)
        if M44 != None:
            WriteParameter("M44", "double", M44, mat_file, indent)
        temperature_initial = mat_params.get("temperature_initial", None)
        if temperature_initial != None:
            WriteParameter("Initial Temperature", "double", temperature_initial, mat_file, indent)
        temperature_reference = mat_params.get("temperature_reference", None)
        if temperature_reference != None:
            WriteParameter("Reference Temperature", "double", temperature_reference, mat_file, indent)

        WriteParameter("Basis Vector 1", "Array(double)", VectorToString(vec1), mat_file, indent)
        WriteParameter("Basis Vector 2", "Array(double)", VectorToString(vec2), mat_file, indent)
        WriteParameter("Basis Vector 3", "Array(double)", VectorToString(vec3), mat_file, indent)
        indent = EndParamList(mat_file, indent)        

        if num_slip_systems > 0:

            indent = StartParamList("Slip System Family 0", mat_file, indent)

            # Flow rule
            if mat_params["flow_rule"] == "Power Law":
                indent = StartParamList("Flow Rule", mat_file, indent)
                WriteParameter("Type", "string", "Power Law", mat_file, indent)
                WriteParameter("Reference Slip Rate", "double", mat_params["rate_slip_reference"], mat_file, indent)
                WriteParameter("Rate Exponent", "double", mat_params["exponent_rate"], mat_file, indent)
                indent = EndParamList(mat_file, indent)
    
            if mat_params["flow_rule"] == "Thermal Activation":
                indent = StartParamList("Flow Rule", mat_file, indent)
                WriteParameter("Type", "string", "Thermal Activation", mat_file, indent)
                WriteParameter("Reference Slip Rate", "double", mat_params["rate_slip_reference"], mat_file, indent)
                WriteParameter("Activation Energy", "double", mat_params["energy_activation"], mat_file, indent)
                WriteParameter("Thermal Resistance", "double", mat_params["resistance_thermal"], mat_file, indent)
                WriteParameter("P Exponent", "double", mat_params["exponent_p"], mat_file, indent)
                WriteParameter("Q Exponent", "double", mat_params["exponent_q"], mat_file, indent)
                indent = EndParamList(mat_file, indent)
    
            elif mat_params["flow_rule"] == "Power Law with Drag":
                indent = StartParamList("Flow Rule", mat_file, indent)
                WriteParameter("Type", "string", "Power Law with Drag", mat_file, indent)
                WriteParameter("Reference Slip Rate", "double", mat_params["rate_slip_reference"], mat_file, indent)
                WriteParameter("Rate Exponent", "double", mat_params["exponent_rate"], mat_file, indent)
                WriteParameter("Drag Coefficient", "double", mat_params["drag_coeff"], mat_file, indent)
                indent = EndParamList(mat_file, indent)
    
            # Hardening laws
            if mat_params["hardening_law"] == "Linear Minus Recovery":
                indent = StartParamList("Hardening Law", mat_file, indent)
                WriteParameter("Type", "string", "Linear Minus Recovery", mat_file, indent)
                WriteParameter("Hardening Modulus", "double", mat_params["modulus_hardening"], mat_file, indent)
                WriteParameter("Recovery Modulus", "double", mat_params["modulus_recovery"], mat_file, indent)
                WriteParameter("Initial Hardening State", "double", mat_params["state_hardening_initial"], mat_file, indent)
                indent = EndParamList(mat_file, indent)
    
            elif mat_params["hardening_law"] == "Saturation":
                indent = StartParamList("Hardening Law", mat_file, indent)
                WriteParameter("Type", "string", "Saturation", mat_file, indent)
                WriteParameter("Hardening Rate", "double", mat_params["rate_hardening"], mat_file, indent)
                WriteParameter("Initial Saturation Stress", "double", mat_params["stress_saturation_initial"], mat_file, indent)
                WriteParameter("Saturation Exponent", "double", mat_params["exponent_saturation"], mat_file, indent)
                WriteParameter("Reference Slip Rate", "double", mat_params["rate_slip_reference"], mat_file, indent)
                WriteParameter("Initial Hardening State", "double", mat_params["state_hardening_initial"], mat_file, indent)
                indent = EndParamList(mat_file, indent)
    
            elif mat_params["hardening_law"] == "Dislocation Density":
                indent = StartParamList("Hardening Law", mat_file, indent)
                WriteParameter("Type", "string", "Dislocation Density", mat_file, indent)
                WriteParameter("Geometric Factor", "double", mat_params["factor_geometric"], mat_file, indent)
                WriteParameter("Generation Factor", "double", mat_params["factor_generation"], mat_file, indent)
                WriteParameter("Annihilation Factor", "double", mat_params["factor_annihilation"], mat_file, indent)
                WriteParameter("Shear Modulus", "double", mat_params["modulus_shear"], mat_file, indent)
                WriteParameter("Burgers Vector Magnitude", "double", mat_params["magnitude_burgers"], mat_file, indent)
                WriteParameter("Initial Hardening State", "double", mat_params["state_hardening_initial"], mat_file, indent)
                indent = EndParamList(mat_file, indent)
    
            indent = EndParamList(mat_file, indent)
        

        # Crystal plasticity slip systems
        for iSS in range(num_slip_systems):
            direction = slip_systems[iSS][0]
            normal = slip_systems[iSS][1]
            indent = StartParamList("Slip System " + str(iSS+1), mat_file, indent)
            WriteParameter("Slip Direction", "Array(double)", VectorToString(direction), mat_file, indent)
            WriteParameter("Slip Normal", "Array(double)", VectorToString(normal), mat_file, indent)
            indent = EndParamList(mat_file, indent)

        indent = EndParamList(mat_file, indent)

    indent = EndParamList(mat_file, indent)
    indent = EndParamList(mat_file, indent)
    mat_file.close()

    print "\nMaterials file written to", file_name

    return

if __name__ == "__main__":

    if len(sys.argv) is 1:

        names_file = os.walk('.').next()[2]

        names_potential = [name for name in names_file if name.lower().endswith('_matprops.txt')]
        if len(names_potential) is 1:
            name_base = '_'.join(names_potential[0].split('_')[:-1])
            mat_params_file_name = names_potential[0]
        else:
            raise InputError('Non-unique or missing assumed base file name')

        names_potential = [name for name in names_file if name.lower().endswith('_rotations.txt')]
        if len(names_potential) is 1:
            if name_base != '_'.join(names_potential[0].split('_')[:-1]):
                raise InputError('Inconsistently named rotations file. ' + name_base + '_Rotations.txt must exist to use assumed names.')
            name_base = '_'.join(names_potential[0].split('_')[:-1])
            rotations_file_name = names_potential[0]
        else:
            raise InputError('Non-unique or missing rotations file. ' + name_base + '_Rotations.txt must exist to use assumed names.')

        names_potential = [name for name in names_file if name.lower().endswith('_rotations.txt')]
        if names_file.__contains__(name_base + '.g'):
            name_file_exodus = name_base + '.g'
        else:
            raise InputError('Inconsistently named or missing mesh file. ' + name_base + '.g must exist to use assumed names.')

    elif len(sys.argv) is 2:

        name_base = sys.argv[1]
        mat_params_file_name = name_base + '_MatProps.txt'
        rotations_file_name = name_base + '_Rotations.txt'
        name_file_exodus = name_base + '.g'

        assert os.path.isfile(mat_params_file_name)
        assert os.path.isfile(rotations_file_name)
        assert os.path.isfile(name_file_exodus)

    elif len(sys.argv) != 4:
        print "\nUsage: python -m lcm_preprocess.write_file_material <mat_props.txt> <rotation_matrices.txt> <mesh_filename>\n"
        sys.exit(1)

    else:

        mat_params_file_name = sys.argv[1]
        rotations_file_name = sys.argv[2]
        name_file_exodus = sys.argv[3]

    names_block = read_names_block(name_file_exodus)

    element_type = read_element_type(name_file_exodus)

    # List of material parameters that are expected to be in the input file
    # If it's set to None, then it is a required parameter
    mat_params = create_mat_params_default()

    vars_output = create_vars_output_default()

    ParseMaterialParametersFile(mat_params_file_name, mat_params, vars_output)

    rotations = ParseRotationMatricesFile(rotations_file_name)

    materials_file_name = name_file_exodus.split('.')[0] + '_Material.xml'
    WriteMaterialsFile(materials_file_name, mat_params, vars_output, rotations, names_block, element_type)

    print "\nComplete.\n"
