# Name the test with the directory name
get_filename_component(testDirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(testName ${testDirName})

# Copy Input file from source to binary dir
IF (ALBANY_EPETRA) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_oc.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_oc.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_oc_dp.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_oc_dp.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_nlopt.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_nlopt.yaml COPYONLY)
ENDIF() 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_ocT.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_ocT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_oc_dpT.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_oc_dpT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nodal_nloptT.yaml ${CMAKE_CURRENT_BINARY_DIR}/nodal_nloptT.yaml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.0 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.1 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.2 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mitchell3D.gen.4.3 ${CMAKE_CURRENT_BINARY_DIR}/mitchell3D.gen.4.3 COPYONLY)

# Create the test with this name and standard executable
IF (SEACAS_EPU AND SEACAS_EXODIFF)

  IF (ALBANY_EPETRA) 
    add_test(NAME ATO:${testName} 
           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
           -DOUT_NAME=mitchell3D_oc -DOUT_FILE=exodiff_oc.txt -DDIFF_NAME=${testName}
           -DTEST_NAME=${testName} -DTEST_ARGS=nodal_oc.yaml -DMPIMNP=${MPIMNP} 
           -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} 
           -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
    add_test(NAME ATO:${testName}_DistParam
           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${Albany.exe}"
           -DOUT_NAME=mitchell3D_oc_dp -DOUT_FILE=exodiff_oc_dp.txt -DDIFF_NAME=mitchell3D_oc_dp
           -DTEST_NAME=${testName} -DTEST_ARGS=nodal_oc_dp.yaml -DMPIMNP=${MPIMNP} 
           -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} 
           -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
    IF (ATO_NLOPT)
      add_test(ATO:${testName}_Nodal_NLOPT ${Albany.exe} nodal_nlopt.yaml)
    ENDIF()
  ENDIF() 
  
  add_test(NAME ATOT:${testName}
           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
           -DOUT_NAME=mitchell3D_ocT -DOUT_FILE=exodiff_ocT.txt -DDIFF_NAME=${testName}
           -DTEST_NAME=${testName} -DTEST_ARGS=nodal_ocT.yaml -DMPIMNP=${MPIMNP} 
           -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} 
           -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
#  add_test(NAME ATOT:${testName}_DistParam
#           COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyT.exe}"
#           -DOUT_NAME=mitchell3D_oc_dpT -DOUT_FILE=exodiff_oc_dpT.txt -DDIFF_NAME=mitchell3D_oc_dp
#           -DTEST_NAME=${testName} -DTEST_ARGS=nodal_oc_dpT.yaml -DMPIMNP=${MPIMNP} 
#           -DSEACAS_EPU=${SEACAS_EPU} -DSEACAS_EXODIFF=${SEACAS_EXODIFF} 
#           -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
  IF (ATO_NLOPT)
    add_test(ATOT:${testName}_Nodal_NLOPT ${AlbanyT.exe} nodal_nloptT.yaml)
  ENDIF()

ENDIF()
