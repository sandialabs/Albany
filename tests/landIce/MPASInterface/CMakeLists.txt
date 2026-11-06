##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Tests of the MPAS-Albany interface (src/landIce/interfaceWithMPAS), which call it as MALI does,
# through a mock of the MPAS side (MockMPAS.hpp).
#
#  - landIce_MPASInterface_UnitTests_{Parallel}: compare with exact solutions (and check the
#    parameters the interface sets), so they need no baseline.
#  - landIce_MPASInterface_<case>_{Parallel}: regression tests (MockMALI.cpp). They compare
#    metrics of the fields returned to MPAS with the baseline in <case>/mock_mpas.yaml. If the
#    baseline is missing, the test fails and prints it. Serial and parallel runs share the baseline.
#
# The interface can use two meshes, selected by the Discretization Method in albany_input.yaml: the
# logically extruded mesh (Extruded, the default) and the 3D STK mesh (STKExtruded). The tests above
# use the default; the tests with STKExtruded in the name run the same tests with the 3D STK mesh
# (the regression tests use the same baselines, since the two meshes give the same results).
#
# The interface reads albany_input.yaml from the working directory and writes its timers there,
# so each test runs in its own directory.

get_filename_component(parentPath ${CMAKE_CURRENT_SOURCE_DIR} PATH)
get_filename_component(parentDirName ${parentPath} NAME)
get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set (testNameRoot ${parentDirName}_${dirName})

add_library (mockMPAS MockMPAS.cpp)
target_include_directories (mockMPAS PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (mockMPAS PUBLIC mpasInterface)

#######################################
#             Unit tests              #
#######################################

if (TARGET albany_ut_main)
  add_executable (MPASInterfaceUnitTests MPASInterfaceUnitTests.cpp)
  target_link_libraries (MPASInterfaceUnitTests PUBLIC mockMPAS albany_ut_main)

  # The tests that compare the two meshes run only with the default Method
  foreach (method Default STKExtruded)
    if (method STREQUAL "Default")
      set (suffix Parallel)
      set (methodArg)
    else()
      set (suffix ${method}_Parallel)
      set (methodArg --disc-method=${method})
    endif()
    set (testName ${testNameRoot}_UnitTests_${suffix})
    set (runDir ${CMAKE_CURRENT_BINARY_DIR}/UnitTests_${suffix})
    file (MAKE_DIRECTORY ${runDir})
    add_test (NAME ${testName}
              COMMAND ${PARALLEL_CALL} $<TARGET_FILE:MPASInterfaceUnitTests> ${methodArg}
              WORKING_DIRECTORY ${runDir})
    set_tests_properties (${testName} PROPERTIES LABELS "LandIce;Forward;unit")
  endforeach()
endif()

#######################################
#          Regression tests           #
#######################################

add_executable (MockMALI MockMALI.cpp)
target_link_libraries (MockMALI PUBLIC mockMPAS)

foreach (case dome domeDepthInt marine)
  set (albanyInput ${CMAKE_CURRENT_SOURCE_DIR}/${case}/albany_input.yaml)

  # Default mesh (logically extruded)
  set (testName ${testNameRoot}_${case}_Parallel)
  set (runDir ${CMAKE_CURRENT_BINARY_DIR}/${case}_Parallel)
  configure_file (${albanyInput} ${runDir}/albany_input.yaml COPYONLY)
  # The input is read from the source tree, so a baseline pasted there is used without reconfiguring
  add_test (NAME ${testName}
            COMMAND ${PARALLEL_CALL} $<TARGET_FILE:MockMALI> ${CMAKE_CURRENT_SOURCE_DIR}/${case}/mock_mpas.yaml
            WORKING_DIRECTORY ${runDir})
  set_tests_properties (${testName} PROPERTIES LABELS "LandIce;Forward")

  # 3D STK mesh: the same input, with Method STKExtruded, and the same baseline
  file (READ ${albanyInput} defaultInput)
  string (REPLACE "  Discretization: {}" "  Discretization:\n    Method: STKExtruded" stkInput "${defaultInput}")
  if ("${stkInput}" STREQUAL "${defaultInput}")
    message (FATAL_ERROR "${albanyInput} must contain the line '  Discretization: {}', to set the Method of the STKExtruded test.")
  endif()
  # Reconfigure when the input changes (as configure_file does)
  set_property (DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${albanyInput})
  set (testName ${testNameRoot}_${case}_STKExtruded_Parallel)
  set (runDir ${CMAKE_CURRENT_BINARY_DIR}/${case}_STKExtruded_Parallel)
  file (MAKE_DIRECTORY ${runDir})
  file (WRITE ${runDir}/albany_input.yaml "${stkInput}")
  add_test (NAME ${testName}
            COMMAND ${PARALLEL_CALL} $<TARGET_FILE:MockMALI> ${CMAKE_CURRENT_SOURCE_DIR}/${case}/mock_mpas.yaml
            WORKING_DIRECTORY ${runDir})
  set_tests_properties (${testName} PROPERTIES LABELS "LandIce;Forward")
endforeach()
