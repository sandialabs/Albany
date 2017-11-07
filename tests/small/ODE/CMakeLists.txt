 

# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input.xml COPYONLY)
# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
# 3. Create the test with this name and standard executable
IF(NOT ALBANY_PARALLEL_ONLY AND ALBANY_EPETRA)
  add_test(${testName}_SERIAL ${SerialAlbany.exe})
ENDIF()

# 1'. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputTP.xml
               ${CMAKE_CURRENT_BINARY_DIR}/inputTP.xml COPYONLY)
# 2'. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Tpetra NAME)
# 3'. Create the test with this name and standard executable
if (NOT (ALBANY_CHECK_FPE OR ALBANY_STRONG_FPE_CHECK))
# Fails with FPE if FPE checking is enabled
  add_test(${testName}_SERIAL_Tpetra ${SerialAlbanyT.exe} inputT.xml)
endif ()
add_test(${testName}_SERIAL_Tpetra_Ifpack2 ${SerialAlbanyT.exe} inputTP.xml)

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
  endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota_nls.data)
  get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR}_Dakota NAME)
  add_test(${testName} ${AlbanyDakota.exe})
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dakota.in)
endif()

# 6. Repeat process for Analysis problems if "inputAnalysis.xml" exists
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml
                 ${CMAKE_CURRENT_BINARY_DIR}/inputAnalysis.xml COPYONLY)

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
endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/inputAnalysis.xml)
