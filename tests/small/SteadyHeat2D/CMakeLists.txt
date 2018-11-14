

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} input.yaml)
endif()

if (ALBANY_IFPACK2)
# 1'. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_RegressFail.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT_RegressFail.yaml COPYONLY)
# 2'. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3'. Create the test with this name and standard executable
add_test(${testName}_SERIAL_Tpetra ${SerialAlbanyT.exe} inputT.yaml)
add_test(${testName}_Tpetra_RegressFail ${SerialAlbanyT.exe} inputT_RegressFail.yaml)
set_tests_properties(${testName}_Tpetra_RegressFail PROPERTIES WILL_FAIL TRUE)
add_test(${testName}_Tpetra ${AlbanyT.exe} inputT.yaml)
endif ()

if (ALBANY_MUELU_EXAMPLES)
# 1'. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Tpetra_MueLu NAME)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_MueLu.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT_MueLu.yaml COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_MueLu_RegressFail.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT_MueLu_RegressFail.yaml COPYONLY)
    # 3'. Create the test with this name and standard executable
    add_test(${testName}_SERIAL ${SerialAlbanyT.exe} inputT_MueLu.yaml)
    add_test(${testName}_RegressFail ${SerialAlbanyT.exe} inputT_MueLu_RegressFail.yaml)
    set_tests_properties(${testName}_RegressFail PROPERTIES WILL_FAIL TRUE)
    add_test(${testName} ${AlbanyT.exe} inputT_MueLu.yaml)
endif ()

# 5. Repeat process for Dakota problems if "dakota.in" exists
if (ALBANY_DAKOTA)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
                 ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_nls.data COPYONLY)
  endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota NAME)
  add_test(${testName} ${AlbanyDakota.exe})
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
endif()

# 6. Repeat process for Analysis problems if "inputAnalysis.yaml" exists
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis.yaml COPYONLY)

  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
  endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)

  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Moocho.opt)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Moocho.opt
                   ${CMAKE_CURRENT_BINARY_DIR}/Moocho.opt COPYONLY)
  endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Moocho.opt)

  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Analysis NAME)
  add_test(${testName} ${AlbanyAnalysis.exe})
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml)

# 7. 2D Steady Heat (Poisson) problem with trigonometric source and
# Dirichlet/Neumann BCs.
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_trigSource.yaml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_trigSource.yaml COPYONLY)
#add_test(${testName}_TrigonometricSource ${Albany.exe} input_trigSource.yaml)

# 8. 2D Steady Heat (Poisson) problem with trigonometric source and periodic
# BCs.
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_trigSource_periodicBCs.yaml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_trigSource_periodicBCs.yaml COPYONLY)
#add_test(${testName}_TrigonometricSource_PeriodicBCs ${Albany.exe} input_trigSource_periodicBCs.yaml)
