#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

#################################################
#             Scalar Interpolation              #
#################################################

create_unit_test (NAME    DOFInterpolation
                  PREFIX  ${testNameRoot}
                  SOURCES DOFInterpolation.cpp)

#################################################
#               Gather Solution                 #
#################################################

create_unit_test (NAME    GatherSolution
                  PREFIX  ${testNameRoot}
                  SOURCES GatherSolution.cpp)

#################################################
#           Gather Distributed Param            #
#################################################

create_unit_test (NAME    GatherDistributedParameter
                  PREFIX  ${testNameRoot}
                  SOURCES GatherDistributedParameter.cpp)

#################################################
#               Scatter Residual                #
#################################################

create_unit_test (NAME    ScatterResidual
                  PREFIX  ${testNameRoot}
                  SOURCES ScatterResidual.cpp)

#################################################
#          Scatter Scalar Response              #
#################################################

create_unit_test (NAME    ScatterScalarResponse
                  PREFIX  ${testNameRoot}
                  SOURCES ScatterScalarResponse.cpp)
