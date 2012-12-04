

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_l1l2_sincos.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_l1l2_sincos.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal_flip.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal_flip.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sacado.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sacado.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_ismip-hom_testA.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_ismip-hom_testA.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal_all_glensLaw.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal_all_glensLaw.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testC.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testC.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testD.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testD.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reference_solution.dat
               ${CMAKE_CURRENT_BINARY_DIR}/reference_solution.dat COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/felix_stokes_fo_cosexp2Dall_in.exo
               ${CMAKE_CURRENT_BINARY_DIR}/felix_stokes_fo_cosexp2Dall_in.exo COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
add_test(${testName}_Poly ${Albany.exe})

#  Second test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/icouette_periodic.xml
               ${CMAKE_CURRENT_BINARY_DIR}/icouette_periodic.xml COPYONLY)
add_test(${testName}_CouettePeriodicX ${Albany.exe} icouette_periodic.xml)

#  Third test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sinsin.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sinsin.xml COPYONLY)
add_test(${testName}_SinSinPeriodicXY ${Albany.exe} input_sinsin.xml)

#  Fourth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sinsin_dirichlet.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sinsin_dirichlet.xml COPYONLY)
add_test(${testName}_SinSinDirichlet ${Albany.exe} input_sinsin_dirichlet.xml)

#  Fifth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sincosz_dirichlet.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sincosz_dirichlet.xml COPYONLY)
add_test(${testName}_SinCosZ3DDirichlet ${Albany.exe} input_sincosz_dirichlet.xml)

#  Sixth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sincosz.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sincosz.xml COPYONLY)
add_test(${testName}_SinCosZ3DPeriodicXY ${Albany.exe} input_sincosz.xml)

#  Seventh test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sincos_glen.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sincos_glen.xml COPYONLY)
add_test(${testName}_SinCosGlen ${Albany.exe} input_sincos_glen.xml)

#  Eighth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincos2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincos2D.xml COPYONLY)
add_test(${testName}_FOSinCosGlen ${Albany.exe} input_fo_sincos2D.xml)

#  Ninth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincosz.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincosz.xml COPYONLY)
add_test(${testName}_FOSinCosZ3DPeriodicXY ${Albany.exe} input_fo_sincosz.xml)

#  Tenth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testA.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testA.xml COPYONLY)
add_test(${testName}_FO_ISMIP-HOM_TestA ${Albany.exe} input_fo_ismip-hom_testA.xml)

#  Eleventh test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sinexp_neumann.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sinexp_neumann.xml COPYONLY)
add_test(${testName}_FO_SINEXP_Neumann ${Albany.exe} input_fo_sinexp_neumann.xml)

#  Twelfth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal.xml COPYONLY)
add_test(${testName}_FO_COSEXP_Basal ${Albany.exe} input_fo_cosexp_basal.xml)

#  Thirteenth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_cosexp_basal_all_glensLaw.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_cosexp_basal_all_glensLaw.xml COPYONLY)
add_test(${testName}_FO_COSEXP_BasalAll ${SerialAlbany.exe}
input_fo_cosexp_basal_all_glensLaw.xml)
