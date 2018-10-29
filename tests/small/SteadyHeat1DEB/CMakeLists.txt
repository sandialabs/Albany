

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
# 1'. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
# 2. Copy materials file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/materials.yaml COPYONLY)

# 3. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 5. Create the test (actually the data for the test) with this name and standard executable

add_executable(HeatProfile HeatProfile.cpp)

#add_test(${testName}_Prep "${ALBANY_PRETEST_EXEC} ${CMAKE_CURRENT_BINARY_DIR}/HeatProfile")

# 5. Create the test with this name and standard executable

#if (ALBANY_IFPACK2)
#  add_test(${testName}_Tpetra ${AlbanyT.exe} inputT.yaml)
#  set_tests_properties(${testName}_Tpetra  PROPERTIES DEPENDS ${testName}_Prep)
#endif ()

#if (ALBANY_EPETRA) 
#add_test(${testName} ${Albany.exe})
#set_tests_properties(${testName}  PROPERTIES DEPENDS ${testName}_Prep)
#endif()

if (ALBANY_EPETRA)
add_test(NAME ${testName}
     COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${Albany.exe}"
     "-DTEST_ARGS=input.yaml"
     "-DAUX_ENV=${ALBANY_PRETEST_EXEC}"
     "-DAUX_PROG=${CMAKE_CURRENT_BINARY_DIR}/HeatProfile" -P
     ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

if (ALBANY_IFPACK2)
add_test(NAME ${testName}_Tpetra
     COMMAND ${CMAKE_COMMAND}  "-DTEST_PROG=${AlbanyT.exe}"
     "-DTEST_ARGS=inputT.yaml"
     "-DAUX_ENV=${ALBANY_PRETEST_EXEC}"
     "-DAUX_PROG=${CMAKE_CURRENT_BINARY_DIR}/HeatProfile" -P
     ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif ()
