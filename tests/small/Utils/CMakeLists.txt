# 1. Copy Input file from source to binary dir
IF (ALBANY_PANZER_EXPR_EVAL)
  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/input_populate_mesh_expr_eval.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_populate_mesh_expr_eval.xml COPYONLY)
ELSE()
  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/input_populate_mesh.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_populate_mesh.xml COPYONLY)
ENDIF()

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/dummy_field.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/dummy_field.ascii COPYONLY)

# 2. Name the test with the directory name
GET_FILENAME_COMPONENT(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable

IF (ALBANY_STK)
  IF (ALBANY_PANZER_EXPR_EVAL)
    ADD_TEST(${testName} ${AlbanyT.exe} input_populate_mesh_expr_eval.xml)
  ELSE()
    ADD_TEST(${testName} ${AlbanyT.exe} input_populate_mesh.xml)
  ENDIF()
ENDIF()
