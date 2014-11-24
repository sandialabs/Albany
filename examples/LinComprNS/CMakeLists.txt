
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2DunsteadyNS.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2DunsteadyNS.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D_standingWave.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2D_standingWave.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2D.xml COPYONLY)
IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  add_test(${testName}_2DSteady ${SerialAlbany.exe} input2D.xml)
ENDIF()

# 4. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2Dunsteady.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2Dunsteady.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_2DUnsteadyInvPressPulse ${Albany.exe} input2Dunsteady.xml)
endif()

# 5. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2DunsteadyMMS.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2DunsteadyMMS.xml COPYONLY)
IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  add_test(${testName}_2DUnsteadyEulerMMS ${SerialAlbany.exe} input2DunsteadyMMS.xml)
ENDIF()

# 6. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input2D_drivenPulse.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input2D_drivenPulse.xml COPYONLY)
IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  add_test(${testName}_2DDrivenPulse ${SerialAlbany.exe} input2D_drivenPulse.xml)
ENDIF()

# 7. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input3Dunsteady.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input3Dunsteady.xml COPYONLY)
IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  add_test(${testName}_3DUnsteadyInvPressPulse ${SerialAlbany.exe} input3Dunsteady.xml)
ENDIF()

# 8. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input3DunsteadyNS.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input3DunsteadyNS.xml COPYONLY)
IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  add_test(${testName}_3DUnsteadyInvPressPulseNS ${SerialAlbany.exe} input3DunsteadyNS.xml)
ENDIF()

# 9. Create the test with this name and standard executable
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input1D_standingWave.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input1D_standingWave.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_1DStandingWave ${Albany.exe}
input1D_standingWave.xml)
endif()
