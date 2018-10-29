	# 1. Copy Input file from source to binary dir
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_domeAnalysis.yaml
		       ${CMAKE_CURRENT_BINARY_DIR}/input_fo_domeAnalysis.yaml COPYONLY)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_domeForward.yaml
		       ${CMAKE_CURRENT_BINARY_DIR}/input_fo_domeForward.yaml COPYONLY)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dakota.in
		       ${CMAKE_CURRENT_BINARY_DIR}/dakota.in COPYONLY)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/domeSyntheticData
		       ${CMAKE_CURRENT_BINARY_DIR}/domeSyntheticData COPYONLY)

	# 2. Name the test with the directory name
	get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

	#3. Tests
	#  First test 
	IF(ALBANY_SEACAS)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_dome.yaml
		       ${CMAKE_CURRENT_BINARY_DIR}/input_fo_dome.yaml COPYONLY)
	add_test(${testName}_Dome_SDBC ${AlbanyT.exe} input_fo_dome.yaml)
	ENDIF()

	#  Second test 
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_confinedShelf.yaml
		       ${CMAKE_CURRENT_BINARY_DIR}/input_fo_confinedShelf.yaml COPYONLY)
	if (ALBANY_EPETRA) 
	add_test(${testName}_ConfinedShelf ${Albany.exe} input_fo_confinedShelf.yaml)
	endif()

	#  Third test 
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_confinedShelf_depthIntBC.yaml
		       ${CMAKE_CURRENT_BINARY_DIR}/input_fo_confinedShelf_depthIntBC.yaml COPYONLY)
	if (ALBANY_EPETRA) 
	add_test(${testName}_ConfinedShelf_DepthIntBC ${Albany.exe}
	input_fo_confinedShelf_depthIntBC.yaml)
	endif()

	# Not tested...
	if (ALBANY_IFPACK2) 
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_confinedShelfT.yaml
		       ${CMAKE_CURRENT_BINARY_DIR}/input_fo_confinedShelfT.yaml COPYONLY)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_confinedShelf_depthIntBCT.yaml
		       ${CMAKE_CURRENT_BINARY_DIR}/input_fo_confinedShelf_depthIntBCT.yaml COPYONLY)
	endif()

	#  Fourth test 
	IF(ALBANY_SEACAS)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_circularShelf.yaml
		       ${CMAKE_CURRENT_BINARY_DIR}/input_fo_circularShelf.yaml COPYONLY)
	if (ALBANY_EPETRA) 
	add_test(${testName}_CircularShelf ${Albany.exe}
	input_fo_circularShelf.yaml)
	endif()

	if (ALBANY_IFPACK2) 
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_circularShelfT.yaml
		       ${CMAKE_CURRENT_BINARY_DIR}/input_fo_circularShelfT.yaml COPYONLY)
	add_test(${testName}_CircularShelf_Tpetra ${AlbanyT.exe}
	input_fo_circularShelfT.yaml)
	endif()
	ENDIF()

	#  Fifth test 
	IF(ALBANY_SEACAS)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_fo_dome_interpSurf.yaml
		       ${CMAKE_CURRENT_BINARY_DIR}/input_fo_dome_interpSurf.yaml COPYONLY)
	if (ALBANY_EPETRA) 
	add_test(${testName}_Dome_interpSurf ${Albany.exe}
input_fo_dome_interpSurf.yaml)
endif()
ENDIF()
