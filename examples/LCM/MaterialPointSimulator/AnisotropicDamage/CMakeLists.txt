##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Create a symlink to the runtest.py script
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtest.py} ${CMAKE_CURRENT_BINARY_DIR}/runtest.py)

if (ALBANY_IFPACK2) 
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${runtestT.py} ${CMAKE_CURRENT_BINARY_DIR}/runtestT.py)
endif()

# Create a symlink to the MPS
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${MPS.exe} ${CMAKE_CURRENT_BINARY_DIR}/MPS)
if (ALBANY_IFPACK2) 
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${MPST.exe} ${CMAKE_CURRENT_BINARY_DIR}/MPST)
endif()

# Create a symlink to exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy script file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AnisotropicDamage.py
               ${CMAKE_CURRENT_BINARY_DIR}/AnisotropicDamage.py COPYONLY)
if (ALBANY_IFPACK2) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AnisotropicDamageT.py
               ${CMAKE_CURRENT_BINARY_DIR}/AnisotropicDamageT.py COPYONLY)
endif()

# Copy Input files from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AnisotropicDamage-uniaxial.xml
               ${CMAKE_CURRENT_BINARY_DIR}/AnisotropicDamage-uniaxial.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AnisotropicDamage-shear.xml
               ${CMAKE_CURRENT_BINARY_DIR}/AnisotropicDamage-shear.xml COPYONLY)

# Copy the reference solution and exodiff files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AnisotropicDamage-uniaxial.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/AnisotropicDamage-uniaxial.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AnisotropicDamage-uniaxial.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/AnisotropicDamage-uniaxial.exodiff COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AnisotropicDamage-shear.gold.exo
               ${CMAKE_CURRENT_BINARY_DIR}/AnisotropicDamage-shear.gold.exo COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AnisotropicDamage-shear.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/AnisotropicDamage-shear.exodiff COPYONLY)

# Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#MESSAGE(SEACAS_EXODIFF "${SEACAS_EXODIFF}")

# Run the "runtest.py" file in the CWD
add_test(NAME ${testName} COMMAND "python" "AnisotropicDamage.py")
if (ALBANY_IFPACK2) 
add_test(NAME ${testName}_Tpetra COMMAND "python" "AnisotropicDamageT.py")
endif()
set_tests_properties(${testName}  PROPERTIES REQUIRED_FILES "${SEACAS_EXODIFF}")
