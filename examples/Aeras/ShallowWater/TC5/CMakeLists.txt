 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicitHV_BE_nu8e15_vort_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicitHV_BE_nu8e15_vort_T.xml COPYONLY)
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicitHV_BE_nu8e15_novort_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicitHV_BE_nu8e15_novort_T.xml COPYONLY)               
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicitHV_RK4_nu8e15_novort_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_explicitHV_RK4_nu8e15_novort_T.xml COPYONLY)
               
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_implicitHV_BE_nu8e15_vort_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_implicitHV_BE_nu8e15_vort_T.xml COPYONLY)                                             

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_implicitHV_BE_nu8e15_novort_T.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_implicitHV_BE_nu8e15_novort_T.xml COPYONLY)  
               
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_hv_T.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_hv_T.xml COPYONLY)               
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicit_hv_T.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_explicit_hv_T.xml COPYONLY)              
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_implicitTimeInt.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_implicitTimeInt.xml COPYONLY)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_explicitTimeInt.xml
#               ${CMAKE_CURRENT_BINARY_DIR}/input_explicitTimeInt.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../grids/QUAD9/uniform_16_quad9.g
               ${CMAKE_CURRENT_BINARY_DIR}/uniform_16_quad9.g COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
#add_test(Aeras_${testName}_Tpetra ${AlbanyT.exe} inputT.xml)
#add_test(Aeras_${testName}_HV_Tpetra ${AlbanyT.exe} input_hv_T.xml)

add_test(Aeras_${testName}_explicitHV_BE_nu8e15_vort_Tpetra ${AlbanyT.exe} input_explicitHV_BE_nu8e15_vort_T.xml)
add_test(Aeras_${testName}_explicitHV_BE_nu8e15_novort_Tpetra ${AlbanyT.exe} input_explicitHV_BE_nu8e15_novort_T.xml)

add_test(Aeras_${testName}_explicitHV_RK4_nu8e15_novort_Tpetra ${AlbanyT.exe} input_explicitHV_RK4_nu8e15_novort_T.xml)

add_test(Aeras_${testName}_implicitHV_BE_nu8e15_vort_Tpetra ${AlbanyT.exe} input_implicitHV_BE_nu8e15_vort_T.xml)
add_test(Aeras_${testName}_implicitHV_BE_nu8e15_novort_Tpetra ${AlbanyT.exe} input_implicitHV_BE_nu8e15_novort_T.xml)


# 7. Repeat process for Dakota + Ensemble problems if "dakota.in" exists
if (ALBANY_ENSEMBLE)
  if (ALBANY_DAKOTA)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_uniform.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_uniform.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputDakota_uniform.xml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputDakota_uniform.xml COPYONLY)
    get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota_Uniform NAME)
    #add_test(Aeras_${testName} ${AlbanyDakota.exe} inputDakota_uniform.xml)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_uniform_mp.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_uniform_mp.in COPYONLY)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMPDakota_uniform.xml
                   ${CMAKE_CURRENT_BINARY_DIR}/inputMPDakota_uniform.xml COPYONLY)
    get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota_Uniform_Ensemble NAME)
    #add_test(Aeras_${testName} ${AlbanyDakota.exe} inputMPDakota_uniform.xml)
  endif ()
endif ()
