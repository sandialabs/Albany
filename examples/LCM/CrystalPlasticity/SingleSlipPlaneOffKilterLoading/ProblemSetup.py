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

def bc_string(node_set_id, dof, value):

    msg =  "      <ParameterList name=\"Time Dependent DBC on NS nodelist_" + str(node_set_id) + " for DOF " + dof + "\">\n"
    msg += "        <Parameter name=\"Number of points\" type=\"int\" value=\"2\"/>\n"
    msg += "        <Parameter name=\"Time Values\" type=\"Array(double)\" value=\"{ 0.0, 1.0}\"/>\n"
    msg += "        <Parameter name=\"BC Values\" type=\"Array(double)\" value=\"{ 0.0, " + str(value) + "}\"/>\n"
    msg += "      </ParameterList>"

    return msg

if __name__ == "__main__":

    print "\nThis test has a single element with a single slip plane."
    print "The slip plane is orientated such that it is not aligned with"
    print "the coordinate axes.  Loading is then applied such that there"
    print "is zero shear stress on the slip plane.  There should be no"
    print "plastic slip in this case.  A second case is also tested in which"
    print "the slip plane is at a 45-degree angle to the load.  This results"
    print "in the maximum possible resolved shear stress, and hence plastic"
    print "flow."

    F11 = 1.01
    F22 = F33 = sqrt(1.0/F11)

    DeformationGradient = [[F11, 0.0, 0.0],
                           [0.0, F22, 0.0],
                           [0.0, 0.0, F33]]

    Bunge_Euler_phi1 = 5.123*pi/180.0
    Bunge_Euler_Phi = 0.0
    Bunge_Euler_phi2 = 0.0

    R_active = ConstructActiveRotationTensor(Bunge_Euler_phi1, Bunge_Euler_Phi, Bunge_Euler_phi2)

    BasisVector1, BasisVector2, BasisVector3 = ConstructBasisVectors(R_active)

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

    nodes = []
    nodes.append((1, 0, 1))
    nodes.append((1, 1, 1))
    nodes.append((0, 1, 1))
    nodes.append((0, 0, 1))
    nodes.append((1, 1, 0))
    nodes.append((1, 0, 0))
    nodes.append((0, 0, 0))
    nodes.append((0, 1, 0))

    prescribed_displacements = []
    for node in nodes:
        displacement = [0.0, 0.0, 0.0]
        for i in range(3):
            for j in range(3):
                displacement[i] += DeformationGradient[i][j]*node[j]
        for i in range(3):
            displacement[i] -= node[i]
        prescribed_displacements.append(displacement)

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
    for i in range(len(prescribed_displacements)):
        print bc_string(i+1, "X", prescribed_displacements[i][0])
        print bc_string(i+1, "Y", prescribed_displacements[i][1])
        print bc_string(i+1, "Z", prescribed_displacements[i][2])
        
    print
