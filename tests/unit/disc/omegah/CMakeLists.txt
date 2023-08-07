#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

# Copy mesh files from source to binary dir
set(meshName "gis_unstruct_basal_populated.osh")
file(COPY        ${CMAKE_CURRENT_SOURCE_DIR}/OmegahMeshes/${meshName}
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

#################################################
#          Omega_h Discretization tests         #
#################################################

create_unit_test (NAME    OmegahConMgr
                  SOURCES OmegahConMgr.cpp)
