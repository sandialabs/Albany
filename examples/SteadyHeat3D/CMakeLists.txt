
# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_10x10x10_ioss.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_10x10x10_ioss.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_10x10x10_ascii.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_10x10x10_ascii.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/eles
               ${CMAKE_CURRENT_BINARY_DIR}/eles COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/xyz
               ${CMAKE_CURRENT_BINARY_DIR}/xyz COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_nodeGIDArrayResponse.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_nodeGIDArrayResponse.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_nodeGIDArrayResponse.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT_nodeGIDArrayResponse.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
if (ALBANY_EPETRA) 
add_test(${testName} ${Albany.exe})

add_test(${testName}_nodeGIDArrayResponse ${Albany.exe} input_nodeGIDArrayResponse.xml)
endif()

# PUMI. Serial for now.
if (ALBANY_SCOREC)
  configure_file (${CMAKE_CURRENT_SOURCE_DIR}/inputT_pumi.xml
    ${CMAKE_CURRENT_BINARY_DIR}/inputT_pumi.xml COPYONLY)
  add_test (${testName}_pumi ${SerialAlbanyT.exe} inputT_pumi.xml)
endif ()

IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  add_test(${testName}_10x10x10_ioss ${SerialAlbany.exe} input_10x10x10_ioss.xml)

  add_test(${testName}_10x10x10_ascii ${SerialAlbany.exe} input_10x10x10_ascii.xml)
ENDIF()

if (ALBANY_IFPACK2)
# 1'. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_10x10x10_ioss.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT_10x10x10_ioss.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT_10x10x10_ascii.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT_10x10x10_ascii.xml COPYONLY)
# 2'. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3'. Create the test with this name and standard executable
add_test(${testName}_Tpetra ${AlbanyT.exe} inputT.xml)
add_test(${testName}_nodeGIDArrayResponse_Tpetra ${AlbanyT.exe} inputT_nodeGIDArrayResponse.xml)

IF(NOT ALBANY_PARALLEL_ONLY)
  #add_test(${testName}_10x10x10_ioss_Tpetra ${SerialAlbanyT.exe} inputT_10x10x10_ioss.xml)

  #add_test(${testName}_10x10x10_ascii_Tpetra ${SerialAlbanyT.exe} inputT_10x10x10_ascii.xml)
ENDIF()
endif ()

# 4. Repeat process for SG problems if "inputSG.xml" exists
if (ALBANY_SG)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputSG.xml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputSG.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputSG.xml COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_SG NAME)
  add_test(${testName} ${AlbanySG.exe})
endif ()
endif ()

# 5. Repeat process for Dakota problems if "dakota.in" exists
if (ALBANY_DAKOTA)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
                 ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota_nls.data COPYONLY)
  endif ()
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota NAME)
  add_test(${testName} ${AlbanyDakota.exe})
endif ()
endif()

# 6. Repeat process for Analysis problems if "inputAnalysis.xml" exists
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis.xml COPYONLY)

  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
                   ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
  endif ()

  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Moocho.opt)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Moocho.opt
                   ${CMAKE_CURRENT_BINARY_DIR}/Moocho.opt COPYONLY)
  endif ()

  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Analysis NAME)
  add_test(${testName} ${AlbanyAnalysis.exe})
endif ()
