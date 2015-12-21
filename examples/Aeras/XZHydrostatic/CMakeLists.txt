 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectralT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectralT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_with_topography.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_with_topography.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plot_xz_matlab.m
               ${CMAKE_CURRENT_BINARY_DIR}/plot_xz_matlab.m COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectral_ho_RK4_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectral_ho_RK4_T.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectral_ho_BE_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectral_ho_BE_T.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(Aeras_${testName} ${Albany.exe})
endif()

add_test(Aeras_${testName}_Tpetra ${AlbanyT.exe} inputT.xml) 

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
  add_test(Aeras_${testName} ${AlbanySG.exe})
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputSG.xml)
endif ()

# 5. Repeat process for Dakota problems if "dakota.in" exists
if (ALBANY_DAKOTA AND ALBANY_EPETRA)
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
  add_test(Aeras_${testName} ${AlbanyDakota.exe})
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
endif()

# 6. Repeat process for Analysis problems if "inputAnalysis.xml" exists

if (ALBANY_EPETRA) 
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
  add_test(Aeras_${testName} ${AlbanyAnalysis.exe})
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml)
endif()

# 7. Repeat process for Dakota + Ensemble problems if "dakota.in" exists
if (ALBANY_ENSEMBLE)
  if (ALBANY_DAKOTA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_list.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_list.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputDakota_list.xml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputDakota_list.xml COPYONLY)
    get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota_List NAME)
    add_test(Aeras_${testName} ${AlbanyDakota.exe} inputDakota_list.xml)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_list_mp.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_list_mp.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMPDakota_list.xml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputMPDakota_list.xml COPYONLY)
    get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota_List_Ensemble NAME)
    add_test(Aeras_${testName} ${AlbanyDakota.exe} inputMPDakota_list.xml)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_uniform.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_uniform.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputDakota_uniform.xml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputDakota_uniform.xml COPYONLY)
    get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota_Uniform NAME)
    add_test(Aeras_${testName} ${AlbanyDakota.exe} inputDakota_uniform.xml)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_uniform_mp.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_uniform_mp.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMPDakota_uniform.xml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputMPDakota_uniform.xml COPYONLY)
    get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota_Uniform_Ensemble NAME)
    add_test(Aeras_${testName} ${AlbanyDakota.exe} inputMPDakota_uniform.xml)
  endif ()
endif ()
