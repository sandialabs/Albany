#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

#################################################
#             Scalar Interpolation              #
#################################################

create_unit_test (NAME    DOFInterpolation
                  SOURCES DOFInterpolation.cpp)

#################################################
#               Gather Solution                 #
#################################################

create_unit_test (NAME    GatherSolution
                  SOURCES GatherSolution.cpp)

#################################################
#           Gather Distributed Param            #
#################################################

create_unit_test (NAME    GatherDistributedParameter
                  SOURCES GatherDistributedParameter.cpp)

#################################################
#               Scatter Residual                #
#################################################

create_unit_test (NAME    ScatterResidual
                  SOURCES ScatterResidual.cpp)

#################################################
#          Scatter Scalar Response              #
#################################################

create_unit_test (NAME    ScatterScalarResponse
                  SOURCES ScatterScalarResponse.cpp)
