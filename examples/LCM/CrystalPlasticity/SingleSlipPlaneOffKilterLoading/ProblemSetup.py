#! /usr/bin/env python

from math import pi, sqrt, sin, cos

def ConstructActiveRotationTensor(phi1, Phi, phi2):

    R =  [[0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]]

    R[0][0] =  cos(phi1)*cos(phi2) - sin(phi1)*sin(phi2)*cos(Phi)
    R[0][1] = -cos(phi1)*sin(phi2) - sin(phi1)*cos(phi2)*cos(Phi)
    R[0][2] =  sin(phi1)*sin(Phi)
    R[1][0] =  sin(phi1)*cos(phi2) + cos(phi1)*sin(phi2)*cos(Phi)
    R[1][1] = -sin(phi1)*sin(phi2) + cos(phi1)*cos(phi2)*cos(Phi)
    R[1][2] = -cos(phi1)*sin(Phi)
    R[2][0] =  sin(phi2)*sin(Phi)
    R[2][1] =  cos(phi2)*sin(Phi)
    R[2][2] =  cos(Phi)

    return R

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

def bc_string(node_set_id, dof, times, values):

    msg =  "      <ParameterList name=\"Time Dependent DBC on NS nodelist_" + str(node_set_id) + " for DOF " + dof + "\">\n"
    msg += "        <Parameter name=\"Number of points\" type=\"int\" value=\"" + str(len(times)) + "\"/>\n"
    msg += "        <Parameter name=\"Time Values\" type=\"Array(double)\" value=\"{"
    for i in range(len(times)):
        msg += str(times[i])
        if i < len(times) - 1:
            msg += ", "
    msg += "}\"/>\n"
    msg += "        <Parameter name=\"BC Values\" type=\"Array(double)\" value=\"{"
    for i in range(len(values)):
        msg += str(values[i])
        if i < len(values) - 1:
            msg += ", "
    msg += "}\"/>\n"
    msg += "      </ParameterList>"

    return msg

if __name__ == "__main__":

    YoungsModulus = 200.0
    PoissonsRatio = 0.3
    C11 = (1.0 - PoissonsRatio)            * ( YoungsModulus / ((1.0 + PoissonsRatio)*(1.0 - 2.0*PoissonsRatio)) )
    C12 = PoissonsRatio                    * ( YoungsModulus / ((1.0 + PoissonsRatio)*(1.0 - 2.0*PoissonsRatio)) )
    C44 = ((1.0 - 2.0*PoissonsRatio)/2.0)  * ( YoungsModulus / ((1.0 + PoissonsRatio)*(1.0 - 2.0*PoissonsRatio)) )
    print "\nElastic constants:"
    print "  C11", C11
    print "  C12", C12
    print "  C44", C44

    Bunge_Euler_phi1 = 5.123*pi/180.0
    Bunge_Euler_Phi = 0.0
    Bunge_Euler_phi2 = 0.0

    R_active = ConstructActiveRotationTensor(Bunge_Euler_phi1, Bunge_Euler_Phi, Bunge_Euler_phi2)

#    R_passive = [[R_active[0][0], R_active[1][0], R_active[2][0]],
#                 [R_active[0][1], R_active[1][1], R_active[2][1]],
#                 [R_active[0][2], R_active[1][2], R_active[2][2]]]

    BasisVector1, BasisVector2, BasisVector3 = ConstructBasisVectors(R_active)

    nodes = []
    nodes.append((1, 0, 1))
    nodes.append((1, 1, 1))
    nodes.append((0, 1, 1))
    nodes.append((0, 0, 1))
    nodes.append((1, 1, 0))
    nodes.append((1, 0, 0))
    nodes.append((0, 0, 0))
    nodes.append((0, 1, 0))

    time_step = 0.01
    num_time_steps = 12
    final_time = time_step*(num_time_steps-1)
    times = [n*time_step for n in range(num_time_steps)]

    prescribed_displacements = [[], [], [], [], [], [], [], []]

    final_tensile_strain = 0.001

    for time in times:

        F11 = 1.0 + final_tensile_strain*time/final_time
#        PoissonsRatio = 0.3
#        F22 = F33 = 1.0 - PoissonsRatio * (F11 - 1.0)
        F22 = F33 = 1.0/sqrt(F11)

        DeformationGradient = [[F11, 0.0, 0.0],
                               [0.0, F22, 0.0],
                               [0.0, 0.0, F33]]

        Temp = [[0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]]    

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    Temp[i][j] += DeformationGradient[i][k]*R_active[j][k]

        DeformationGradient = [[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    DeformationGradient[i][j] += R_active[i][k]*Temp[k][j]

        for iNode in range(len(nodes)):
            node = nodes[iNode]
            displacement = [0.0, 0.0, 0.0]
            for i in range(3):
                for j in range(3):
                    displacement[i] += DeformationGradient[i][j]*node[j]
            for i in range(3):
                displacement[i] -= node[i]
            prescribed_displacements[iNode].append(displacement)

    print "\nEuler phi_1 ", Bunge_Euler_phi1
    print "Euler Phi   ", Bunge_Euler_Phi
    print "Euler phi_2 ", Bunge_Euler_phi2

    print "\nBasis vectors\n"
    msg = "        <Parameter name=\"Basis Vector 1\" type=\"Array(double)\" value=\"{" + str(BasisVector1[0]) + "," + str(BasisVector1[1]) + "," + str(BasisVector1[2]) + "}\"/>"
    print msg
    msg = "        <Parameter name=\"Basis Vector 2\" type=\"Array(double)\" value=\"{" + str(BasisVector2[0]) + "," + str(BasisVector2[1]) + "," + str(BasisVector2[2]) + "}\"/>"
    print msg
    msg = "        <Parameter name=\"Basis Vector 3\" type=\"Array(double)\" value=\"{" + str(BasisVector3[0]) + "," + str(BasisVector3[1]) + "," + str(BasisVector3[2]) + "}\"/>"
    print msg

    print "\nActive Rotation Tensor"
    for i in range(3):
        print R_active[i]
        
    print "\nDeformation Gradient"
    for i in range(3):
        print DeformationGradient[i]

    print "\nPrescribed displacement boundary conditions\n"
    bc_xml = ""
    for iNode in range(len(prescribed_displacements)):
        disp_x = []
        disp_y = []
        disp_z = []
        for iTimeStep in range(len(prescribed_displacements[iNode])):
            disp_x.append(prescribed_displacements[iNode][iTimeStep][0])
            disp_y.append(prescribed_displacements[iNode][iTimeStep][1])
            disp_z.append(prescribed_displacements[iNode][iTimeStep][2])                   
        bc_xml += bc_string(iNode+1, "X", times, disp_x) + "\n"
        bc_xml += bc_string(iNode+1, "Y", times, disp_y) + "\n"
        bc_xml += bc_string(iNode+1, "Z", times, disp_z) + "\n"

    template_file = open("SingleSlipPlaneOffKilterLoading.template.xml")
    lines = template_file.readlines()
    template_file.close()
    xml_file = open("SingleSlipPlaneOffKilterLoading.xml", 'w')
    for line in lines:
        if "PLACEHOLDER" in line:
            xml_file.write(bc_xml)
        else:
            xml_file.write(line)
    xml_file.close()
        
    print "Input deck written to SingleSlipPlaneOffKilterLoading.xml\n"
