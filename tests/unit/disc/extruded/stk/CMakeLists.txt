#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(grandParentPath ${parentPath} PATH)
get_filename_component(grandParentDirName ${grandParentPath} NAME)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${grandParentDirName}_${parentDirName}_${dirName})

# Copy the populated basal mesh file (shared with the omegah unit tests)
set(meshName "gis_unstruct_basal_populated.exo")
set(meshSrcDir "${CMAKE_CURRENT_SOURCE_DIR}/../../omegah/OmegahMeshes")
file(COPY        ${meshSrcDir}/${meshName}
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

#################################################
#  Extruded disc with STK Ioss basal mesh tests  #
#################################################

create_unit_test (NAME    ExtrudedWithSTKBasal
                  PREFIX  ${testNameRoot}
                  SOURCES ExtrudedWithSTKBasal.cpp)
