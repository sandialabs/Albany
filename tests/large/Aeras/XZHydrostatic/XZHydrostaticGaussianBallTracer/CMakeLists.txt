 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectralT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectralT.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectral_np3_RK4_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectral_np3_RK4_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectral_np3_BE_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectral_np3_BE_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectral_np4_BE_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectral_np4_BE_T.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_spectral_np5_BE_T.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_spectral_np5_BE_T.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable

add_test(Aeras_${testName}_Spectral_np2_RungeKutta4 ${AlbanyT.exe} input_spectralT.yaml)
IF(NOT ALBANY_CHECK_FPE)
IF(NOT ALBANY_STRONG_FPE_CHECK) 
IF(NOT Kokkos_ENABLE_Cuda) 
add_test(Aeras_${testName}_Spectral_np4_BackwardEuler ${AlbanyT8.exe} input_spectral_np4_BE_T.yaml) 
ENDIF()
ENDIF()
ENDIF()

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
  add_test(Aeras_${testName} ${AlbanyDakota.exe})
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

  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Moocnp3.opt)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Moocnp3.opt
                   ${CMAKE_CURRENT_BINARY_DIR}/Moocnp3.opt COPYONLY)
  endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Moocnp3.opt)

  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Analysis NAME)
  add_test(Aeras_${testName} ${AlbanyAnalysis.exe})
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml)
