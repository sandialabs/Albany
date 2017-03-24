# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/channelDiffusion.g
               ${CMAKE_CURRENT_BINARY_DIR}/channelDiffusion.g COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/channelDiffusion.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/channelDiffusion.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/channelDiffusionMaterials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/channelDiffusionMaterials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/channelDiffusion.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/channelDiffusion.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/channelDiffusion.gold.e
               ${CMAKE_CURRENT_BINARY_DIR}/channelDiffusion.gold.e COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
add_test(MechWithHydrogenFastPath_channelDiffusion python ../run_exodiff_test.py "${SerialAlbanyT.exe}" channelDiffusion.yaml)
endif()
