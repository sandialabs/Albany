# 1. Copy Input file from source to binary dir
if (ALBANY_EPETRA)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputML1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputML1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputML3.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputML3.xml COPYONLY)
endif() 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMueLuShort1.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMueLuShort1.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputMueLuShort3.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputMueLuShort3.xml COPYONLY)
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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/antarctica_2d.exo.8.0
               ${CMAKE_CURRENT_BINARY_DIR}/antarctica_2d.exo.8.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/antarctica_2d.exo.8.1
               ${CMAKE_CURRENT_BINARY_DIR}/antarctica_2d.exo.8.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/antarctica_2d.exo.8.2
               ${CMAKE_CURRENT_BINARY_DIR}/antarctica_2d.exo.8.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/antarctica_2d.exo.8.3
               ${CMAKE_CURRENT_BINARY_DIR}/antarctica_2d.exo.8.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/antarctica_2d.exo.8.4
               ${CMAKE_CURRENT_BINARY_DIR}/antarctica_2d.exo.8.4 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/antarctica_2d.exo.8.5
               ${CMAKE_CURRENT_BINARY_DIR}/antarctica_2d.exo.8.5 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/antarctica_2d.exo.8.6
               ${CMAKE_CURRENT_BINARY_DIR}/antarctica_2d.exo.8.6 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/antarctica_2d.exo.8.7
               ${CMAKE_CURRENT_BINARY_DIR}/antarctica_2d.exo.8.7 COPYONLY)

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#3. Tests 
#  First Test
#if (ALBANY_EPETRA) 
#add_test(${testName}_Gis20km ${Albany.exe} input_fo_gis20km_test.xml)
#add_test(${testName}_GisUnstructured ${Albany.exe} input_fo_gis_unstruct.xml)
#endif(ALBANY_EPETRA)
#if(ALBANY_IFPACK2)
#add_test(${testName}_Gis20km_Tpetra ${AlbanyT.exe} input_fo_gis20km_testT.xml)
#endif(ALBANY_IFPACK2)

