# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Copy materials file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)

# Copy mesh files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fuel_volume.exo
               ${CMAKE_CURRENT_BINARY_DIR}/fuel_volume.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fuel_volume.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/fuel_volume.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fuel_volume.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/fuel_volume.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fuel_volume.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/fuel_volume.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fuel_volume.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/fuel_volume.exo.4.3 COPYONLY)

# Create the the data for the test
# CylHeatProfile executable needs the ncdump file for mesh coordinates
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fuel_volume.ncdump
               ${CMAKE_CURRENT_BINARY_DIR}/fuel_volume.ncdump COPYONLY)

add_executable(CylHeatProfile CylHeatProfile.cpp)
target_link_libraries(CylHeatProfile  ${Trilinos_EXTRA_LD_FLAGS})

if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)

  add_test(NAME ${testName}_Epetra
       COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${Albany.exe}"
       "-DTEST_ARGS=input.yaml"
       "-DAUX_ENV=${ALBANY_PRETEST_EXEC}"
       "-DAUX_PROG=${CMAKE_CURRENT_BINARY_DIR}/CylHeatProfile" -P
       ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")
endif()

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)

  add_test(NAME ${testName}_Tpetra
       COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${Albany.exe}"
       "-DTEST_ARGS=inputT.yaml"
       "-DAUX_ENV=${ALBANY_PRETEST_EXEC}"
       "-DAUX_PROG=${CMAKE_CURRENT_BINARY_DIR}/CylHeatProfile" -P
       ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
       WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Forward")
endif ()
