##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Tests of the MPAS-Albany interface (src/landIce/interfaceWithMPAS), which call it as MALI does,
# through a mock of the MPAS side (MockMPAS.hpp).
#
#  - landIce_MPASInterface_UnitTests_<mesh>_Parallel: compare with exact solutions (and check the
#    parameters the interface sets), so they need no baseline.
#  - landIce_MPASInterface_<case>_<mesh>_Parallel: regression tests (MockMALI.cpp). They compare
#    metrics of the fields returned to MPAS with the baseline in <case>/mock_mpas.yaml. If the
#    baseline is missing, the test fails and prints it. Serial and parallel runs share the baseline.
#
# The interface can use two meshes, selected by the Discretization Method in albany_input.yaml: the
# logically extruded mesh (Extruded), which numbers its entities column-wise or layer-wise (Columnwise
# Ordering: true or false), and the 3D STK mesh (STKExtruded). <mesh> is one of:
#  - Extruded_Columnwise: Method Extruded, Columnwise Ordering true;
#  - Extruded_Layerwise:  Method Extruded, Columnwise Ordering false;
#  - STKExtruded:         Method STKExtruded.
# The tests set these explicitly, so they do not depend on the defaults of the interface. The
# regression tests of a case use the same baseline with all meshes, since they give the same results
# (up to the tolerances); its albany_input.yaml leaves the Discretization sublist empty
# ('Discretization: {}'), and each test fills it in.
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

  foreach (mesh Extruded_Columnwise Extruded_Layerwise STKExtruded)
    if (mesh STREQUAL "Extruded_Columnwise")
      set (meshArgs --disc-method=Extruded --columnwise-ordering=true)
    elseif (mesh STREQUAL "Extruded_Layerwise")
      set (meshArgs --disc-method=Extruded --columnwise-ordering=false)
    else()
      set (meshArgs --disc-method=STKExtruded)
    endif()
    set (testName ${testNameRoot}_UnitTests_${mesh}_Parallel)
    set (runDir ${CMAKE_CURRENT_BINARY_DIR}/UnitTests_${mesh}_Parallel)
    file (MAKE_DIRECTORY ${runDir})
    add_test (NAME ${testName}
              COMMAND ${PARALLEL_CALL} $<TARGET_FILE:MPASInterfaceUnitTests> ${meshArgs}
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
  file (READ ${albanyInput} caseInput)
  # Reconfigure when the input changes (as configure_file does)
  set_property (DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${albanyInput})

  foreach (mesh Extruded_Columnwise Extruded_Layerwise STKExtruded)
    if (mesh STREQUAL "Extruded_Columnwise")
      set (discretization "  Discretization:\n    Method: Extruded\n    Columnwise Ordering: true")
    elseif (mesh STREQUAL "Extruded_Layerwise")
      set (discretization "  Discretization:\n    Method: Extruded\n    Columnwise Ordering: false")
    else()
      set (discretization "  Discretization:\n    Method: STKExtruded")
    endif()
    string (REPLACE "  Discretization: {}" "${discretization}" meshInput "${caseInput}")
    if ("${meshInput}" STREQUAL "${caseInput}")
      message (FATAL_ERROR "${albanyInput} must contain the line '  Discretization: {}', to set the mesh of the ${mesh} test.")
    endif()
    set (testName ${testNameRoot}_${case}_${mesh}_Parallel)
    set (runDir ${CMAKE_CURRENT_BINARY_DIR}/${case}_${mesh}_Parallel)
    file (MAKE_DIRECTORY ${runDir})
    file (WRITE ${runDir}/albany_input.yaml "${meshInput}")
    # The baseline is read from the source tree, so a baseline pasted there is used without reconfiguring
    add_test (NAME ${testName}
              COMMAND ${PARALLEL_CALL} $<TARGET_FILE:MockMALI> ${CMAKE_CURRENT_SOURCE_DIR}/${case}/mock_mpas.yaml
              WORKING_DIRECTORY ${runDir})
    set_tests_properties (${testName} PROPERTIES LABELS "LandIce;Forward")
  endforeach()
endforeach()
