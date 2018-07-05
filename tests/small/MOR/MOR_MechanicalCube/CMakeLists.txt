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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RB.xml
		 ${CMAKE_CURRENT_BINARY_DIR}/RB.xml COPYONLY)
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
			 
# cmake runtest file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake
		   ${CMAKE_CURRENT_BINARY_DIR}/runtest.cmake COPYONLY)			 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/runtestRBGen.cmake
		   ${CMAKE_CURRENT_BINARY_DIR}/runtestRBGen.cmake COPYONLY)			 

# 2. Name the test with the directory name
get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 3. Create the test with this name and standard executable
#test 1 - FOM
SET(OUTFILE "FOM.exo")
SET(REF_FILE "FOM.gold.exo")
add_test(NAME ${testName}_FOM
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}"
         -DTEST_NAME=FOM -DTEST_ARGS=FOM.yaml -DMPIMNP=1
         -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
         -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)
#test 2 - RBGen
SET(OUTFILE "RB.exo")
SET(REF_FILE "RB.gold.exo")
add_test(NAME ${testName}_RB
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${AlbanyRBGenPath}"
         -DTEST_NAME=RB -DTEST_ARGS=RB.xml -DMPIMNP=1
         -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
         -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtestRBGen.cmake)
#test 3 - PGROM
SET(OUTFILE "PGROM.exo")
SET(REF_FILE "PGROM.gold.exo")
add_test(NAME ${testName}_PGROM
         COMMAND ${CMAKE_COMMAND} "-DTEST_PROG=${SerialAlbany.exe}"
         -DTEST_NAME=PGROM -DTEST_ARGS=PGROM.yaml -DMPIMNP=1
         -DSEACAS_EXODIFF=${SEACAS_EXODIFF}
         -DREF_FILENAME=${REF_FILE} -DOUTPUT_FILENAME=${OUTFILE}
         -DDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/runtest.cmake)

endif(SEACAS_EXODIFF)
