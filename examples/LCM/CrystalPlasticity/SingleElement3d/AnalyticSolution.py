#! /usr/bin/env python

from math import sin, cos
from copy import deepcopy

def Invert3x3(matrix):

    minor0 =  matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]
    minor1 =  matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]
    minor2 =  matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]
    minor3 =  matrix[0][1] * matrix[2][2] - matrix[0][2] * matrix[2][1]
    minor4 =  matrix[0][0] * matrix[2][2] - matrix[2][0] * matrix[0][2]
    minor5 =  matrix[0][0] * matrix[2][1] - matrix[0][1] * matrix[2][0]
    minor6 =  matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]
    minor7 =  matrix[0][0] * matrix[1][2] - matrix[0][2] * matrix[1][0]
    minor8 =  matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    determinant = matrix[0][0] * minor0 - matrix[0][1] * minor1 + matrix[0][2] * minor2

    inverse = [[0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0]]

    inverse[0][0] = minor0/determinant
    inverse[0][1] = -1.0*minor3/determinant
    inverse[0][2] = minor6/determinant
    inverse[1][0] = -1.0*minor1/determinant
    inverse[1][1] = minor4/determinant
    inverse[1][2] = -1.0*minor7/determinant
    inverse[2][0] = minor2/determinant
    inverse[2][1] = -1.0*minor5/determinant
    inverse[2][2] = minor8/determinant

    return (determinant, inverse)

def ConstructZeroTensor():

    ZeroTensor = [[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]]

    return ZeroTensor

def ConstructActiveRotationTensor(phi1, Phi, phi2):

    R = ConstructZeroTensor()

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

def Transpose(matrix):

    transpose = ConstructZeroTensor()
    for i in range(3):
        for j in range(3):
            transpose[i][j] = matrix[j][i]

    return transpose

def ComputeCauchyStress(C11, C12, C44, R, DeformationGradient):

    R_transpose = Transpose(R)

    ElasticityTensorVoigtUnrotated = [[C11, C12, C12, 0.0, 0.0, 0.0],
                                      [C12, C11, C12, 0.0, 0.0, 0.0],
                                      [C12, C12, C11, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, C44, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, C44, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0, C44]]

    # Very carefully rotate the elasticity tensor
    # Store it as a proper 4th-order tensor and rotate it using C'_ijkl = R_im R_jn R_ko R_lp C_mnop

    C = deepcopy(ElasticityTensorVoigtUnrotated)

    ElasticityTensorUnrotated = [[[[ C[0][0], C[0][3], C[0][5] ], [ C[0][3], C[0][1], C[0][4] ], [ C[0][5], C[0][4], C[0][2] ]],
                                  [[ C[3][0], C[3][3], C[3][5] ], [ C[3][3], C[3][1], C[3][4] ], [ C[3][5], C[3][4], C[3][2] ]],
                                  [[ C[5][0], C[5][3], C[5][5] ], [ C[5][3], C[5][1], C[5][4] ], [ C[5][5], C[5][4], C[5][2] ]]],
                                 [[[ C[3][0], C[3][3], C[3][5] ], [ C[3][3], C[3][1], C[3][4] ], [ C[3][5], C[3][4], C[3][2] ]],
                                  [[ C[1][0], C[1][3], C[1][5] ], [ C[1][3], C[1][1], C[1][4] ], [ C[1][5], C[1][4], C[1][2] ]],
                                  [[ C[4][0], C[4][3], C[4][5] ], [ C[4][3], C[4][1], C[4][4] ], [ C[4][5], C[4][4], C[4][2] ]]],
                                 [[[ C[5][0], C[5][3], C[5][5] ], [ C[5][3], C[5][1], C[5][4] ], [ C[5][5], C[5][4], C[5][2] ]],
                                  [[ C[4][0], C[4][3], C[4][5] ], [ C[4][3], C[4][1], C[4][4] ], [ C[4][5], C[4][4], C[4][2] ]],
                                  [[ C[2][0], C[2][3], C[2][5] ], [ C[2][3], C[2][1], C[2][4] ], [ C[2][5], C[2][4], C[2][2] ]]]]
    
    ZeroFourthOrderTensor =  [[[[ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ]],
                               [[ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ]],
                               [[ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ]]],
                              [[[ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ]],
                               [[ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ]],
                               [[ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ]]],
                              [[[ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ]],
                               [[ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ]],
                               [[ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ]]]]

    ElasticityTensorRotated = deepcopy(ZeroFourthOrderTensor)

    for m in range(3):
        for n in range(3):
            for o in range(3):
                for l in range(3):
                    for p in range(3):
                        ElasticityTensorRotated[m][n][o][l] += ElasticityTensorUnrotated[m][n][o][p] * R_transpose[p][l]

    Temp = deepcopy(ElasticityTensorRotated)
    ElasticityTensorRotated = deepcopy(ZeroFourthOrderTensor)
    
    for m in range(3):
        for n in range(3):
            for k in range(3):
                for l in range(3):
                    for o in range(3):
                        ElasticityTensorRotated[m][n][k][l] += Temp[m][n][o][l] * R_transpose[o][k]

    Temp = deepcopy(ElasticityTensorRotated)
    ElasticityTensorRotated = deepcopy(ZeroFourthOrderTensor)
    
    for m in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for n in range(3):
                        ElasticityTensorRotated[m][j][k][l] += R[j][n] * Temp[m][n][k][l]

    Temp = deepcopy(ElasticityTensorRotated)
    ElasticityTensorRotated = deepcopy(ZeroFourthOrderTensor)
        
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        ElasticityTensorRotated[i][j][k][l] += R[i][m] * Temp[m][j][k][l]

    C = deepcopy(ElasticityTensorRotated)

    # The elasticity tensor has been rotated, now store it once again in Voigt form

    ElasticityTensorVoigt = [[C[0][0][0][0], C[0][0][1][1], C[0][0][2][2], C[0][0][1][2], C[0][0][0][2], C[0][0][0][1]],
                             [C[1][1][0][0], C[1][1][1][1], C[1][1][2][2], C[1][1][1][2], C[1][1][0][2], C[1][1][0][1]],
                             [C[2][2][0][0], C[2][2][1][1], C[2][2][2][2], C[2][2][1][2], C[2][2][0][2], C[2][2][0][1]],
                             [C[1][2][0][0], C[1][2][1][1], C[1][2][2][2], C[1][2][1][2], C[1][2][0][2], C[1][2][0][1]],
                             [C[0][2][0][0], C[0][2][1][1], C[0][2][2][2], C[0][2][1][2], C[0][2][0][2], C[0][2][0][1]],
                             [C[0][1][0][0], C[0][1][1][1], C[0][1][2][2], C[0][1][1][2], C[0][1][0][2], C[0][1][0][1]]]

    IdentityTensor = [[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]

    DeformationGradientTranspose = Transpose(DeformationGradient)

    J, DeformationGradientInverse = Invert3x3(DeformationGradient)

    CauchyGreenDeformationTensor = ConstructZeroTensor()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                CauchyGreenDeformationTensor[i][j] += DeformationGradientTranspose[i][k]*DeformationGradient[k][j]

    GreenLagrangeStrain = ConstructZeroTensor()
    for i in range(3):
        for j in range(3):
            GreenLagrangeStrain[i][j] = 0.5*(CauchyGreenDeformationTensor[i][j] - IdentityTensor[i][j])

    # Voigt notation: xx, yy, zz, yz, xz, xy
    GreenLagrangeStrainVoigt = [GreenLagrangeStrain[0][0],
                                GreenLagrangeStrain[1][1],
                                GreenLagrangeStrain[2][2],
                                2.0*GreenLagrangeStrain[1][2],
                                2.0*GreenLagrangeStrain[0][2],
                                2.0*GreenLagrangeStrain[0][1]]

    SecondPiolaKirchhoffStressVoigt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(6):
        for j in range(6):
            SecondPiolaKirchhoffStressVoigt[i] += ElasticityTensorVoigt[i][j]*GreenLagrangeStrainVoigt[j]

    SecondPiolaKirchhoffStress = [[SecondPiolaKirchhoffStressVoigt[0], SecondPiolaKirchhoffStressVoigt[5], SecondPiolaKirchhoffStressVoigt[4]],
                                  [SecondPiolaKirchhoffStressVoigt[5], SecondPiolaKirchhoffStressVoigt[1], SecondPiolaKirchhoffStressVoigt[3]],
                                  [SecondPiolaKirchhoffStressVoigt[4], SecondPiolaKirchhoffStressVoigt[3], SecondPiolaKirchhoffStressVoigt[2]]]

    CauchyStress = ConstructZeroTensor()
    Temp = ConstructZeroTensor()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                Temp[i][j] += SecondPiolaKirchhoffStress[i][k]*DeformationGradientTranspose[k][j]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                CauchyStress[i][j] += (1.0/J)*DeformationGradient[i][k]*Temp[k][j]      

    return CauchyStress

if __name__ == "__main__":

    C11 = 168.4
    C12 = 121.4
    C44 = 75.4

    DeformationGradient = [[1.0, 0.1, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]]
    
    print "\n**** CASE 1:  CRYSTAL LATTICE ALIGNED WITH COORDINATE AXES ****"

    Bunge_Euler_phi1 = 0.0
    Bunge_Euler_Phi = 0.0
    Bunge_Euler_phi2 = 0.0

    R_active = ConstructActiveRotationTensor(Bunge_Euler_phi1, Bunge_Euler_Phi, Bunge_Euler_phi2)

    BasisVector1, BasisVector2, BasisVector3 = ConstructBasisVectors(R_active)

    # Note that for this test case R is the identity matrix, so R_active = R_passive
    CauchyStress = ComputeCauchyStress(C11, C12, C44, R_active, DeformationGradient)

    print "\nElastic Constants"
    print "C11 =", C11
    print "C12 =", C12
    print "C44 =", C44

    print "\nBasis Vector 1", BasisVector1
    print "Basis Vector 2", BasisVector2
    print "Basis Vector 3", BasisVector3

    print "\nActive Rotation Tensor"
    for i in range(3):
        print R_active[i]
        
    print "\nDeformation Gradient"
    for i in range(3):
        print DeformationGradient[i]
        
    print "\nCauchy Stress"
    for i in range(3):
        print CauchyStress[i]

    print "\n**** CASE 2:  ROTATED CRYSTAL LATTICE ****"

    Bunge_Euler_phi1 = 0.12
    Bunge_Euler_Phi = 0.03
    Bunge_Euler_phi2 = 0.28
    
    R_active = ConstructActiveRotationTensor(Bunge_Euler_phi1, Bunge_Euler_Phi, Bunge_Euler_phi2)

    BasisVector1, BasisVector2, BasisVector3 = ConstructBasisVectors(R_active)

    # The ComputeCauchyStress() function uses R_passive to determine the components of the elasticity
    # tesnsor, which is aligned with the crystal lattice, as seen by an observer in the global reference frame
    R_passive = Transpose(R_active)

    CauchyStress = ComputeCauchyStress(C11, C12, C44, R_passive, DeformationGradient)

    print "\nElastic Constants"
    print "C11 =", C11
    print "C12 =", C12
    print "C44 =", C44

    print "\nBasis Vector 1", BasisVector1
    print "Basis Vector 2", BasisVector2
    print "Basis Vector 3", BasisVector3

    print "\nActive Rotation Tensor"
    for i in range(3):
        print R_active[i]

    print "\nDeformation Gradient"
    for i in range(3):
        print DeformationGradient[i]
        
    print "\nCauchy Stress"
    for i in range(3):
        print CauchyStress[i]

    print
