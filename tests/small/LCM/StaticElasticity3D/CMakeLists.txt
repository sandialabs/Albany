
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTraction.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTraction.yaml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe})
add_test(${testName}_Traction ${Albany.exe} inputTraction.yaml)
endif()

if (ALBANY_MUELU_EXAMPLES) 
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTractionT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTractionT.yaml COPYONLY)
    add_test(${testName}_Tpetra ${AlbanyT.exe} inputT.yaml)
    add_test(${testName}_Traction_Tpetra ${AlbanyT.exe} inputTractionT.yaml)
endif ()


# 4. Repeat process for SG problems if "inputSG.yaml" exists
if (ALBANY_STOKHOS)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputSG.yaml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSG.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputSG.yaml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_SG NAME)
  add_test(${testName} ${AlbanySG.exe})
endif ()
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
  endif ()
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota NAME)
  add_test(${testName} ${AlbanyDakota.exe})
endif ()
endif()

# 6. Repeat process for Analysis problems if "inputAnalysis.yaml" exists
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis.yaml COPYONLY)

  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
  endif ()

  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Moocho.opt)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Moocho.opt
                   ${CMAKE_CURRENT_BINARY_DIR}/Moocho.opt COPYONLY)
  endif ()

  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Analysis NAME)
  add_test(${testName} ${AlbanyAnalysis.exe})
endif ()

# 7. Repeat process for Dakota problems if "dakota.in" exists
if (ALBANY_ENSEMBLE)
  if (ALBANY_DAKOTA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_mp.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_mp.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMP.yaml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputMP.yaml COPYONLY)
    get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota_Ensemble NAME)
    add_test(${testName} ${AlbanyDakota.exe} inputMP.yaml)
  endif ()
endif ()
