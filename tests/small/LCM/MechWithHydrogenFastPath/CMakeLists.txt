configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_exodiff_test.py
               ${CMAKE_CURRENT_BINARY_DIR}/run_exodiff_test.py COPYONLY)

add_subdirectory(channel_diffusion)
# Disable "MechWithHydrogenFastPath_surfaceDiffusion problem until it is reworked to run properly
#add_subdirectory(surface_diffusion)
