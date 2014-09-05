

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sacado.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sacado.xml COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
# First test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_Poly ${Albany.exe} input.xml)
endif()

#  Second test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/icouette_periodic.xml
               ${CMAKE_CURRENT_BINARY_DIR}/icouette_periodic.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_CouettePeriodicX ${Albany.exe} icouette_periodic.xml)
endif()

#  Third test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sinsin.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sinsin.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinSinPeriodicXY ${Albany.exe} input_sinsin.xml)
endif()

#  Fourth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sinsin_dirichlet.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sinsin_dirichlet.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinSinDirichlet ${Albany.exe} input_sinsin_dirichlet.xml)
endif()

#  Fifth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sincosz_dirichlet.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sincosz_dirichlet.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinCosZ3DDirichlet ${Albany.exe} input_sincosz_dirichlet.xml)
endif()

#  Sixth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sincosz.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sincosz.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinCosZ3DPeriodicXY ${Albany.exe} input_sincosz.xml)
endif()

#  Seventh test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sincos_glen.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sincos_glen.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinCosGlen ${Albany.exe} input_sincos_glen.xml)
endif()

#  Eight test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sinexp_basal.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sinexp_basal.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_SinExpBasal ${Albany.exe} input_sinexp_basal.xml)
endif()
