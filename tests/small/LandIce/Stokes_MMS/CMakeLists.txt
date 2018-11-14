

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sacado.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sacado.yaml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
# First test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_Poly ${Albany.exe} input.yaml)
endif()

if (ALBANY_IFPACK2) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.yaml COPYONLY)
add_test(${testName}_PolyT_yaml ${AlbanyT.exe} inputT.yaml)
endif()

#  Second test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/icouette_periodic.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/icouette_periodic.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_CouettePeriodicX ${Albany.exe} icouette_periodic.yaml)
endif()

#  Third test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sinsin.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sinsin.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinSinPeriodicXY ${Albany.exe} input_sinsin.yaml)
endif()

#  Fourth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sinsin_dirichlet.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sinsin_dirichlet.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinSinDirichlet ${Albany.exe} input_sinsin_dirichlet.yaml)
endif()

#  Fifth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sincosz_dirichlet.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sincosz_dirichlet.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinCosZ3DDirichlet ${Albany.exe} input_sincosz_dirichlet.yaml)
endif()

#  Sixth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sincosz.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sincosz.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinCosZ3DPeriodicXY ${Albany.exe} input_sincosz.yaml)
endif()

#  Seventh test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sincos_glen.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sincos_glen.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinCosGlen ${Albany.exe} input_sincos_glen.yaml)
endif()

#  Eight test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sinexp_basal.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sinexp_basal.yaml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinExpBasal ${Albany.exe} input_sinexp_basal.yaml)
endif()
