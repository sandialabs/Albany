
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surfaceDiffusion_cse.g
               ${CMAKE_CURRENT_BINARY_DIR}/surfaceDiffusion_cse.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surfaceDiffusion.xml
               ${CMAKE_CURRENT_BINARY_DIR}/surfaceDiffusion.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surfaceDiffusionMaterials.xml
               ${CMAKE_CURRENT_BINARY_DIR}/surfaceDiffusionMaterials.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surfaceDiffusion.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/surfaceDiffusion.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/surfaceDiffusion.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/surfaceDiffusion.gold.e COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(MechWithHydrogenFastPath_surfaceDiffusion python ../run_exodiff_test.py "${SerialAlbanyT.exe}" surfaceDiffusion.xml)
endif()
