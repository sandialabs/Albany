#/********************************************************************\
#*            Albany, Copyright (2010) Sandia Corporation             *
#*                                                                    *
#* Notice: This computer software was prepared by Sandia Corporation, *
#* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
#* the Department of Energy (DOE). All rights in the computer software*
#* are reserved by DOE on behalf of the United States Government and  *
#* the Contractor as provided in the Contract. You are authorized to  *
#* use this computer software for Governmental purposes but it is not *
#* to be released or distributed to the public. NEITHER THE GOVERNMENT*
#* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
#* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
#* including this sentence must appear on any copies of this software.*
#*    Questions to Andy Salinger, agsalin@sandia.gov                  *
#\********************************************************************/


# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_ismip-hom_testA.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_ismip-hom_testA.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/reference_solution.dat
               ${CMAKE_CURRENT_BINARY_DIR}/reference_solution.dat COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ns.exo
               ${CMAKE_CURRENT_BINARY_DIR}/ns.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ns.exo.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/ns.exo.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ns.exo.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/ns.exo.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ns.exo.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/ns.exo.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ns.exo.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/ns.exo.4.3 COPYONLY)
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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sacado.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sacado.xml COPYONLY)
add_test(${testName}_PolySacado ${Albany.exe} input_sacado.xml)

#  Eighth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_sincos_glen.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_sincos_glen.xml COPYONLY)
add_test(${testName}_SinCosGlen ${Albany.exe} input_sincos_glen.xml)

#  Ninth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincos2D.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincos2D.xml COPYONLY)
add_test(${testName}_FOSinCosGlen ${Albany.exe} input_fo_sincos2D.xml)

#  Tenth test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_sincosz.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_sincosz.xml COPYONLY)
add_test(${testName}_FOSinCosZ3DPeriodicXY ${Albany.exe} input_fo_sincosz.xml)

#  Eleventh test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_ismip-hom_testA.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_ismip-hom_testA.xml COPYONLY)
add_test(${testName}_FO_ISMIP-HOM_TestA ${Albany.exe} input_fo_ismip-hom_testA.xml)
