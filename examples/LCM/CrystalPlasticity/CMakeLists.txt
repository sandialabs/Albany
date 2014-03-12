# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/elastic_shear_2d.xml
               ${CMAKE_CURRENT_BINARY_DIR}/elastic_shear_2d.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/crystal_elasticity_material_2d.xml
               ${CMAKE_CURRENT_BINARY_DIR}/crystal_elasticity_material_2d.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plastic_shear_2d.xml
               ${CMAKE_CURRENT_BINARY_DIR}/plastic_shear_2d.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/crystal_plasticity_material_2d.xml
               ${CMAKE_CURRENT_BINARY_DIR}/crystal_plasticity_material_2d.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}Simple NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(${testName}_Serial ${SerialAlbany.exe} elastic_shear_2d.xml)
ENDIF()
IF(NOT ALBANY_PARALLEL_ONLY)
  add_test(${testName}_Serial ${SerialAlbany.exe} plastic_shear_2d.xml)
ENDIF()


