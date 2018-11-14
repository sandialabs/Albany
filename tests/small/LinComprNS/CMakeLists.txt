
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2DunsteadyNS.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input2DunsteadyNS.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D_standingWave.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input2D_standingWave.yaml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input2D.yaml COPYONLY)
IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  add_test(${testName}_2DSteady ${SerialAlbany.exe} input2D.yaml)
ENDIF()

# 4. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2Dunsteady.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input2Dunsteady.yaml COPYONLY)

if (ALBANY_IFPACK2)
add_test(${testName}_2DUnsteadyInvPressPulse ${AlbanyT.exe} input2Dunsteady.yaml)
ENDIF()

# 5. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2DunsteadyMMS.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input2DunsteadyMMS.yaml COPYONLY)
IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  add_test(${testName}_2DUnsteadyEulerMMS ${SerialAlbany.exe} input2DunsteadyMMS.yaml)
ENDIF()

# 6. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D_drivenPulse.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input2D_drivenPulse.yaml COPYONLY)
IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  add_test(${testName}_2DDrivenPulse ${SerialAlbany.exe} input2D_drivenPulse.yaml)
ENDIF()

# 7. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input3Dunsteady.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input3Dunsteady.yaml COPYONLY)
IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  add_test(${testName}_3DUnsteadyInvPressPulse ${SerialAlbany.exe} input3Dunsteady.yaml)
ENDIF()

# 8. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input3DunsteadyNS.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input3DunsteadyNS.yaml COPYONLY)
IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  add_test(${testName}_3DUnsteadyInvPressPulseNS ${SerialAlbany.exe} input3DunsteadyNS.yaml)
ENDIF()

# 9. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input1D_standingWave.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input1D_standingWave.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_1DStandingWave ${Albany.exe}
input1D_standingWave.yaml)
endif()
