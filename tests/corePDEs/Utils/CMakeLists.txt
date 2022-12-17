# Name the test with the directory name
get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testName ${parentDirName}_${dirName})

# Copy input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dummy_field.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/dummy_field.ascii COPYONLY)

if (ALBANY_PANZER_EXPR_EVAL)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_populate_mesh_expr_eval.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_populate_mesh_expr_eval.yaml COPYONLY)

  add_test(${testName} ${Albany.exe} input_populate_mesh_expr_eval.yaml)
else()
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_populate_mesh.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_populate_mesh.yaml COPYONLY)

  add_test(${testName} ${Albany.exe} input_populate_mesh.yaml)
endif()
