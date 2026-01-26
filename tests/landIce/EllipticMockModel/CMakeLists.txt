get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testName ${parentDirName}_${dirName})

if (ALBANY_IFPACK2)

  # Copy input file from source to binary di
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/matrices/D_mis.mm
                  ${CMAKE_CURRENT_BINARY_DIR}/matrices/D_mis.mm COPYONLY)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/matrices/A_mis.mm
                  ${CMAKE_CURRENT_BINARY_DIR}/matrices/A_mis.mm COPYONLY)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_slab_mockmodel_fwd.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_slab_mockmodel_fwd.yaml COPYONLY)
  
  # Create the test
  add_test(${testName}_fwd ${Albany.exe} input_slab_mockmodel_fwd.yaml)
  set_tests_properties(${testName}_fwd
                        PROPERTIES
                        LABELS "LandIce; Forward")
                        
      
                        
  # Copy input file from source to binary di  
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/matrices/M.mm
                  ${CMAKE_CURRENT_BINARY_DIR}/matrices/M.mm COPYONLY)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/matrices/A_prior.mm
                  ${CMAKE_CURRENT_BINARY_DIR}/matrices/A_prior.mm COPYONLY)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_slab_mockmodel_analysis.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_slab_mockmodel_analysis.yaml COPYONLY)

  # Create the test
  add_test(${testName}_analysis ${AlbanyAnalysis.exe} input_slab_mockmodel_analysis.yaml)
  set_tests_properties(${testName}_analysis PROPERTIES LABELS "LandIce;Analysis")



  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/matrices/nominal_parameter.ascii
                  ${CMAKE_CURRENT_BINARY_DIR}/matrices/nominal_parameter.ascii COPYONLY)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/matrices/nominal_solution.ascii
                  ${CMAKE_CURRENT_BINARY_DIR}/matrices/nominal_solution.ascii COPYONLY)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_slab_mockmodel_linearized.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_slab_mockmodel_linearized.yaml COPYONLY)

  # Create the test
  add_test(${testName}_linearized_fwd ${Albany.exe} input_slab_mockmodel_linearized.yaml)
  set_tests_properties(${testName}_linearized_fwd PROPERTIES LABELS "LandIce;Forward;Serial")

  

  # Copy input file from source to binary dir
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_slab_sampling.yaml
                  ${CMAKE_CURRENT_BINARY_DIR}/input_slab_sampling.yaml COPYONLY)

  # Create the test
  add_test(${testName}_sampling ${SerialAlbany.exe} input_slab_sampling.yaml)
  set_tests_properties(${testName}_sampling PROPERTIES LABELS "LandIce;Analysis;Serial")
endif()

