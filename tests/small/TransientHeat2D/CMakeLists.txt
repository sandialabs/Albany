

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe} input.yaml)
endif()

if (ALBANY_IFPACK2)
# 1'. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rythmos_be_nox_solver.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/rythmos_be_nox_solver.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rythmos_be_rythmos_solver.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/rythmos_be_rythmos_solver.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/rythmos_rk4.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/rythmos_rk4.yaml COPYONLY)
# 2'. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3'. Create the test with this name and standard executable
add_test(${testName}_Tpetra_Rythmos_BackwardEuler_RythmosSolver ${AlbanyT.exe} rythmos_be_rythmos_solver.yaml)
add_test(${testName}_Tpetra_Rythmos_BackwardEuler_NOXSolver ${AlbanyT.exe} rythmos_be_nox_solver.yaml)
add_test(${testName}_Tpetra_Rythmos_RK4 ${AlbanyT.exe} rythmos_rk4.yaml)
if (ALBANY_TEMPUS)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tempus_be_nox_solver.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/tempus_be_nox_solver.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tempus_rk4.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/tempus_rk4.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tempus_rk4_no_piro.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/tempus_rk4_no_piro.yaml COPYONLY)
add_test(${testName}_Tpetra_Tempus_BackwardEuler_NOXSolver ${AlbanyT.exe} tempus_be_nox_solver.yaml)
add_test(${testName}_Tpetra_Tempus_RK4 ${AlbanyT.exe} tempus_rk4.yaml)
add_test(${testName}_Tpetra_Tempus_NoPiro_RK4 ${AlbanyTempus.exe} tempus_rk4_no_piro.yaml)
endif () 
endif ()

# 5. Repeat process for Dakota problems if "dakota.in" exists
if (ALBANY_DAKOTA)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
                 ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.yaml COPYONLY)
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_nls.data COPYONLY)
  endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota NAME)
  add_test(${testName} ${AlbanyDakota.exe})
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
endif()

# 6. Repeat process for Analysis problems if "inputAnalysis.yaml" exists
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis.yaml COPYONLY)

  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
  endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)

  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Moocho.opt)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Moocho.opt
                   ${CMAKE_CURRENT_BINARY_DIR}/Moocho.opt COPYONLY)
  endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Moocho.opt)

  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Analysis NAME)
  add_test(${testName} ${AlbanyAnalysis.exe})
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.yaml)
