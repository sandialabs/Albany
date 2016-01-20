##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

if (ALBANY_IFPACK2)
# Create a symlink to exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
# Create a symlink to AlbanyT
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyTPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyT)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyPath} ${CMAKE_CURRENT_BINARY_DIR}/Albany)

# input files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAllLocalHarder.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputAllLocalHarder.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAllLocalHarderT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputAllLocalHarderT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAllLocalHarderTera.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputAllLocalHarderTera.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/BluntCrackRatio2d_in_m_rev5_reflect_microns_3d_cse.exo
               ${CMAKE_CURRENT_BINARY_DIR}/BluntCrackRatio2d_in_m_rev5_reflect_microns_3d_cse.exo  COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/BluntCrackRatio2d_in_m_rev5_reflect_microns_3d_cse.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/BluntCrackRatio2d_in_m_rev5_reflect_microns_3d_cse.exo.4.0  COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/BluntCrackRatio2d_in_m_rev5_reflect_microns_3d_cse.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/BluntCrackRatio2d_in_m_rev5_reflect_microns_3d_cse.exo.4.1  COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/BluntCrackRatio2d_in_m_rev5_reflect_microns_3d_cse.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/BluntCrackRatio2d_in_m_rev5_reflect_microns_3d_cse.exo.4.2  COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/BluntCrackRatio2d_in_m_rev5_reflect_microns_3d_cse.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/BluntCrackRatio2d_in_m_rev5_reflect_microns_3d_cse.exo.4.3  COPYONLY)

# material files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialsScaledLocalHarder.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materialsScaledLocalHarder.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materialsScaledLocalHarderTera.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materialsScaledLocalHarderTera.xml COPYONLY)
endif()




