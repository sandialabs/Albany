# 1. Copy Input file from source to binary dir
if (ALBANY_EPETRA)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputML1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputML1.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputML3.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputML3.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMLRay.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMLRay.yaml COPYONLY)
endif() 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMueLuShort1.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMueLuShort1.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMueLuShort3.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMueLuShort3.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMueLuShortRay.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMueLuShortRay.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMueLuLongRay.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMueLuLongRay.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thickness.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/thickness.ascii COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/temperature.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/temperature.ascii COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface_height.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/surface_height.ascii COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/basal_friction_reg.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/basal_friction_reg.ascii COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surface_velocity.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/surface_velocity.ascii COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/antarctica_2d.quad.msh
               ${CMAKE_CURRENT_BINARY_DIR}/antarctica_2d.quad.msh COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/antarctica_2d.exo
               ${CMAKE_CURRENT_BINARY_DIR}/antarctica_2d.exo COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#3. Tests 
#  First Test
if (ALBANY_EPETRA AND ALBANY_IOPX) 
add_test(${testName}_16km_ML ${Albany8.exe} inputML3.yaml)
endif(ALBANY_EPETRA AND ALBANY_IOPX)
if(ALBANY_IFPACK2 AND ALBANY_IOPX)
add_test(${testName}_16km_MueLu ${AlbanyT8.exe} inputMueLuShort3.yaml)
endif(ALBANY_IFPACK2 AND ALBANY_IOPX)

