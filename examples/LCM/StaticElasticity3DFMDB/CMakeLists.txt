#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MeshModel/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MeshModel/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MeshModel/inputSerial.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MeshModel/inputSerial.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MeshModel/inputSerialExoOut.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MeshModel/inputSerialExoOut.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MeshModel/inputEdgeDBC.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MeshModel/inputEdgeDBC.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MeshModel/inputVtxDBC.xml
               ${CMAKE_CURRENT_BINARY_DIR}/MeshModel/inputVtxDBC.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MeshModel/unitCube.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/MeshModel/unitCube.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MeshModel/unitCubeFMDB.sms
               ${CMAKE_CURRENT_BINARY_DIR}/MeshModel/unitCubeFMDB.sms COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Parasolid/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/Parasolid/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Parasolid/inputSerial.xml
               ${CMAKE_CURRENT_BINARY_DIR}/Parasolid/inputSerial.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Parasolid/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/Parasolid/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Parasolid/unitCube.xmt_txt
               ${CMAKE_CURRENT_BINARY_DIR}/Parasolid/unitCube.xmt_txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Parasolid/unitCubeFMDB.sms
               ${CMAKE_CURRENT_BINARY_DIR}/Parasolid/unitCubeFMDB.sms COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
IF(ENABLE_MESHMODEL)
  add_test(
      NAME ${testName}_Mmodel_Traction 
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/MeshModel
      COMMAND ${Albany.exe} input.xml)
  add_test(
      NAME ${testName}_Mmodel_EdgeDBC 
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/MeshModel
      COMMAND ${Albany.exe} inputEdgeDBC.xml)
  add_test(
      NAME ${testName}_Mmodel_VtxDBC 
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/MeshModel
      COMMAND ${Albany.exe} inputVtxDBC.xml)
ENDIF()

IF(ENABLE_PARASOLID)
  add_test(
      NAME ${testName}_Parasolid_Traction 
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Parasolid
      COMMAND ${Albany.exe} input.xml)
ENDIF()

