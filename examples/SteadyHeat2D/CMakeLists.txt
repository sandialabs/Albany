

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe})
endif()

if (ALBANY_IFPACK2)
# 1'. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
# 2'. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3'. Create the test with this name and standard executable
add_test(${testName}_SERIAL_Tpetra ${SerialAlbanyT.exe} inputT.xml)
add_test(${testName}_Tpetra ${AlbanyT.exe} inputT.xml)
endif ()

if (ALBANY_MUELU_EXAMPLES)
# 1'. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Tpetra_MueLu NAME)
  if (ALBANY_64BIT_INT)
    # 2'. Copy Input file from source to binary dir
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_MueLu_64.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT_MueLu_64.xml COPYONLY)
    # 3'. Create the test with this name and standard executable
    add_test(${testName}_SERIAL ${SerialAlbanyT.exe} inputT_MueLu_64.xml)
    add_test(${testName} ${AlbanyT.exe} inputT_MueLu_64.xml)
  else ()
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_MueLu.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT_MueLu.xml COPYONLY)
    # 3'. Create the test with this name and standard executable
    add_test(${testName}_SERIAL ${SerialAlbanyT.exe} inputT_MueLu.xml)
    add_test(${testName} ${AlbanyT.exe} inputT_MueLu.xml)
  endif ()
endif ()

# 4. Repeat process for SG problems if "inputSG.xml" exists
if (ALBANY_STOKHOS)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputSG.xml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSG.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputSG.xml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_SG NAME)
  add_test(${testName} ${AlbanySG.exe})
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputSG.xml)
endif ()

# 5. Repeat process for Dakota problems if "dakota.in" exists
if (ALBANY_DAKOTA)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
                 ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_nls.data COPYONLY)
  endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota NAME)
  add_test(${testName} ${AlbanyDakota.exe})
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
endif()

# 6. Repeat process for Analysis problems if "inputAnalysis.xml" exists
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis.xml COPYONLY)

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
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml)

# 7. 2D Steady Heat (Poisson) problem with trigonometric source and
# Dirichlet/Neumann BCs.
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_trigSource.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_trigSource.xml COPYONLY)
#add_test(${testName}_TrigonometricSource ${Albany.exe} input_trigSource.xml)

# 8. 2D Steady Heat (Poisson) problem with trigonometric source and periodic
# BCs.
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_trigSource_periodicBCs.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_trigSource_periodicBCs.xml COPYONLY)
#add_test(${testName}_TrigonometricSource_PeriodicBCs ${Albany.exe} input_trigSource_periodicBCs.xml)

# 9. Repeat process for Dakota + Ensemble problems if "dakota.in" exists
if (ALBANY_ENSEMBLE)
  if (ALBANY_DAKOTA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_list.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_list.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputDakota_list.xml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputDakota_list.xml COPYONLY)
    get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota_List NAME)
    add_test(${testName} ${AlbanyDakota.exe} inputDakota_list.xml)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_list_mp.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_list_mp.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMPDakota_list.xml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputMPDakota_list.xml COPYONLY)
    get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota_List_Ensemble NAME)
    add_test(${testName} ${AlbanyDakota.exe} inputMPDakota_list.xml)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_uniform.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_uniform.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputDakota_uniform.xml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputDakota_uniform.xml COPYONLY)
    get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota_Uniform NAME)
    add_test(${testName} ${AlbanyDakota.exe} inputDakota_uniform.xml)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_uniform_mp.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_uniform_mp.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMPDakota_uniform.xml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputMPDakota_uniform.xml COPYONLY)
    get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota_Uniform_Ensemble NAME)
    add_test(${testName} ${AlbanyDakota.exe} inputMPDakota_uniform.xml)
  endif ()
endif ()
