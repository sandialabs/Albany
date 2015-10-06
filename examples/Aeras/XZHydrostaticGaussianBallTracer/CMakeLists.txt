 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectralT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectralT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectral_np3_RK4_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectral_np3_RK4_T.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectral_np3_BE_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectral_np3_BE_T.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectral_np4_BE_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectral_np4_BE_T.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectral_np5_BE_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectral_np5_BE_T.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe})
endif()
add_test(${testName}_Tpetra ${AlbanyT.exe} inputT.xml)
if (NOT ALBANY_EPETRA) 
add_test(Aeras_${testName}_Tpetra_Spectral_np2 ${AlbanyT.exe} input_spectralT.xml) 
endif()

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

  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Moocnp3.opt)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Moocnp3.opt
                   ${CMAKE_CURRENT_BINARY_DIR}/Moocnp3.opt COPYONLY)
  endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Moocnp3.opt)

  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Analysis NAME)
  add_test(${testName} ${AlbanyAnalysis.exe})
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml)
