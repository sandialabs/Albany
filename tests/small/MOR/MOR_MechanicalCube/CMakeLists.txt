if(SEACAS_EXODIFF)

# Create a symlink to exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyPath} ${CMAKE_CURRENT_BINARY_DIR}/Albany)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${AlbanyRBGenPath} ${CMAKE_CURRENT_BINARY_DIR}/AlbanyRBGen)

# 1. Copy Input files from source to binary dir
	# material file
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/FOMmat.yaml
					 ${CMAKE_CURRENT_BINARY_DIR}/FOMmat.yaml COPYONLY)
			 
	# input files			 
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/FOM.yaml
					 ${CMAKE_CURRENT_BINARY_DIR}/FOM.yaml COPYONLY)
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RBgen.xml
					 ${CMAKE_CURRENT_BINARY_DIR}/RBgen.xml COPYONLY)
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PGROM.yaml
					 ${CMAKE_CURRENT_BINARY_DIR}/PGROM.yaml COPYONLY)

	# exodiff files
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/FOM.exodiff
					 ${CMAKE_CURRENT_BINARY_DIR}/FOM.exodiff COPYONLY)
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RB.exodiff
					 ${CMAKE_CURRENT_BINARY_DIR}/RB.exodiff COPYONLY)
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PGROM.exodiff
					 ${CMAKE_CURRENT_BINARY_DIR}/PGROM.exodiff COPYONLY)
					 
	# gold files
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/FOM.gold.exo
					 ${CMAKE_CURRENT_BINARY_DIR}/FOM.gold.exo COPYONLY)
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RB.gold.exo
					 ${CMAKE_CURRENT_BINARY_DIR}/RB.gold.exo COPYONLY)
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PGROM.gold.exo
					 ${CMAKE_CURRENT_BINARY_DIR}/PGROM.gold.exo COPYONLY)
			 
	# python runtest file
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.py
					   ${CMAKE_CURRENT_BINARY_DIR}/runtest.py COPYONLY)			 

# 2. Name the test with the directory name
	get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
	add_test(NAME ${testName} COMMAND "python" "runtest.py")

endif(SEACAS_EXODIFF)
