
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTraction.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTraction.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe})
add_test(${testName}_Traction ${Albany.exe} inputTraction.xml)
endif()

if (ALBANY_MUELU_EXAMPLES) 
  if (ALBANY_64BIT_INT)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_64.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT_64.xml COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTractionT_64.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTractionT_64.xml COPYONLY)
    add_test(${testName}_Tpetra_64 ${AlbanyT.exe} inputT_64.xml)
    add_test(${testName}_Traction_Tpetra_64 ${AlbanyT.exe} inputTractionT_64.xml)
  else ()
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTractionT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTractionT.xml COPYONLY)
    add_test(${testName}_Tpetra ${AlbanyT.exe} inputT.xml)
    add_test(${testName}_Traction_Tpetra ${AlbanyT.exe} inputTractionT.xml)
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
endif ()
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
  endif ()
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota NAME)
  add_test(${testName} ${AlbanyDakota.exe})
endif ()
endif()

# 6. Repeat process for Analysis problems if "inputAnalysis.xml" exists
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis.xml COPYONLY)

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
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMP.xml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputMP.xml COPYONLY)
    get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota_Ensemble NAME)
    add_test(${testName} ${AlbanyDakota.exe} inputMP.xml)
  endif ()
endif ()
