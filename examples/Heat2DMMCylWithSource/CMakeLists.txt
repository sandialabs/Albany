

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
# 1'. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
# 2. Copy materials file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.xml COPYONLY)

# 3. Copy mesh files from source to binary dir
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

# 4. CylHeatProfile executable needs the ncdump file for mesh coordinates
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fuel_volume.ncdump
               ${CMAKE_CURRENT_BINARY_DIR}/fuel_volume.ncdump COPYONLY)

# 5. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 6. Create the test (actually the data for the test) with this name and standard executable

add_executable(CylHeatProfile CylHeatProfile.C)

#add_test(${testName}_Prep CylHeatProfile)

# 7. Create the test with this name and standard executable

#add_test(${testName} ${Albany.exe})
#set_tests_properties(${testName}  PROPERTIES DEPENDS ${testName}_Prep)

# 7'. Create the test with this name and standard executable
#add_test(${testName} ${Albany.exe})
#set_tests_properties(${testName}  PROPERTIES DEPENDS ${testName}_Prep)

if (ALBANY_EPETRA) 
add_test(NAME ${testName}
     COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${Albany.exe}" -DTEST_ARGS=
     "-DAUX_ENV=${ALBANY_PRETEST_EXEC}"
     "-DAUX_PROG=${CMAKE_CURRENT_BINARY_DIR}/CylHeatProfile" -P
     ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

if (ALBANY_IFPACK2)
add_test(NAME ${testName}_Tpetra
     COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${AlbanyT.exe}"
     "-DTEST_ARGS=inputT.xml"
     "-DAUX_ENV=${ALBANY_PRETEST_EXEC}"
     "-DAUX_PROG=${CMAKE_CURRENT_BINARY_DIR}/CylHeatProfile" -P
     ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif ()
