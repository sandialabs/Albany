# 1. Copy Input file from source to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_domeAnalysis.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_domeAnalysis.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_domeForward.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_domeForward.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
               ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/domeSyntheticData
               ${CMAKE_CURRENT_BINARY_DIR}/domeSyntheticData COPYONLY)

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#3. Tests
#  First test 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_dome.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_dome.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_Dome ${Albany.exe} input_fo_dome.xml)
endif()

#  Second test 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_confinedShelf.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_confinedShelf.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_ConfinedShelf ${Albany.exe} input_fo_confinedShelf.xml)
endif()

#  Third test 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_confinedShelf_depthIntBC.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_confinedShelf_depthIntBC.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_ConfinedShelf_DepthIntBC ${Albany.exe}
input_fo_confinedShelf_depthIntBC.xml)
endif()

#  Fourth test 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_circularShelf.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_circularShelf.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_CircularShelf ${Albany.exe}
input_fo_circularShelf.xml)
endif()

if (ALBANY_IFPACK2) 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_circularShelfT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_circularShelfT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_confinedShelfT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_confinedShelfT.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_confinedShelf_depthIntBCT.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_confinedShelf_depthIntBCT.xml COPYONLY)
add_test(${testName}_CircularShelf_Tpetra ${AlbanyT.exe}
input_fo_circularShelfT.xml)
endif()

#  Fifth test 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_dome_interpSurf.xml
               ${CMAKE_CURRENT_BINARY_DIR}/input_fo_dome_interpSurf.xml COPYONLY)
if (ALBANY_EPETRA) 
add_test(${testName}_Dome_interpSurf ${Albany.exe}
input_fo_dome_interpSurf.xml)
endif()
