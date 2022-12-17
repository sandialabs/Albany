# The root name for the tests in this directory is the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

####################################
###          Main tests          ###
####################################

if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)

  add_test(${testNameRoot}_Epetra ${Albany.exe} input.yaml)
  set_tests_properties(${testNameRoot}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")
endif()

if (ALBANY_IFPACK2)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)

  add_test(${testNameRoot}_Tpetra ${Albany.exe} inputT.yaml)
  set_tests_properties(${testNameRoot}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Forward")
endif ()

####################################
###         Ascii tests          ###
####################################

IF (NOT ALBANY_PARALLEL_ONLY)
  set(testName ${testNameRoot}_10x10x10_ascii)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/xyz
                 ${CMAKE_CURRENT_BINARY_DIR}/xyz COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eles
                 ${CMAKE_CURRENT_BINARY_DIR}/eles COPYONLY)

  IF(ALBANY_EPETRA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_10x10x10_ascii.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/input_10x10x10_ascii.yaml COPYONLY)

    add_test(${testName}_Epetra ${SerialAlbany.exe} input_10x10x10_ascii.yaml)
    set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")
  ENDIF()

  if (ALBANY_IFPACK2)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_10x10x10_ascii.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputT_10x10x10_ascii.yaml COPYONLY)

    add_test(${testName}_Tpetra ${SerialAlbany.exe} inputT_10x10x10_ascii.yaml)
    set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Forward")
  endif ()
ENDIF ()

####################################
###     Node GID Array tests     ###
####################################

set(testName ${testNameRoot}_nodeGIDArrayResponse)

if (ALBANY_EPETRA) 
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_nodeGIDArrayResponse.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_nodeGIDArrayResponse.yaml COPYONLY)

  add_test(${testName}_Epetra ${Albany.exe} input_nodeGIDArrayResponse.yaml)
  set_tests_properties(${testName}_Epetra PROPERTIES LABELS "Basic;Epetra;Forward")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_nodeGIDArrayResponse.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT_nodeGIDArrayResponse.yaml COPYONLY)

add_test(${testName}_Tpetra ${Albany.exe} inputT_nodeGIDArrayResponse.yaml)
set_tests_properties(${testName}_Tpetra PROPERTIES LABELS "Basic;Tpetra;Forward")
