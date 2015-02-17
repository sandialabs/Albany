#! /usr/bin/env python

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

if __name__ == "__main__":

    ZeroTensor = [[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]]
    
    IdentityTensor = [[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]

    C11 = 168.4
    C12 = 121.4
    C44 = 75.4
    ElasticityTensorVoigt = [[C11, C12, C12, 0.0, 0.0, 0.0],
                             [C12, C11, C12, 0.0, 0.0, 0.0],
                             [C12, C12, C11, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, C44, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, C44, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, C44]]

    DeformationGradient = [[1.0, 0.1, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]]

    DeformationGradientTranspose = deepcopy(ZeroTensor)
    for i in range(3):
        for j in range(3):
            DeformationGradientTranspose[i][j] = DeformationGradient[j][i]

    J, DeformationGradientInverse = Invert3x3(DeformationGradient)

    CauchyGreenDeformationTensor = deepcopy(ZeroTensor)

    for i in range(3):
        for j in range(3):
            for k in range (3):
                CauchyGreenDeformationTensor[i][j] += DeformationGradientTranspose[i][k]*DeformationGradient[k][j]

    GreenLagrangeStrain = deepcopy(ZeroTensor)

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

    CauchyStress = deepcopy(ZeroTensor)
    TempTensor = deepcopy(ZeroTensor)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                TempTensor[i][j] += SecondPiolaKirchhoffStress[i][k]*DeformationGradientTranspose[k][j]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                CauchyStress[i][j] += (1.0/J)*DeformationGradient[i][k]*TempTensor[k][j]

    print "\nElasticity Tensor"
    for i in range(6):
        print ElasticityTensorVoigt[i]

    print "\nDeformation Gradient"
    for i in range(3):
        print DeformationGradient[i]

    print "\nJ = ", J

    print "\nDeformation Gradient Inverse"
    for i in range(3):
        print DeformationGradientInverse[i]

    print "\nCauchy-Green Deformation Tensor"
    for i in range(3):
        print CauchyGreenDeformationTensor[i]

    print "\nGreen-Lagrange Strain"
    for i in range(3):
        print GreenLagrangeStrain[i]

    print "\nSecond Piola-Kirchhoff Stress"
    for i in range(3):
        print SecondPiolaKirchhoffStress[i]
        
    print "\nCauchy Stress"
    for i in range(3):
        print CauchyStress[i] 

    print
