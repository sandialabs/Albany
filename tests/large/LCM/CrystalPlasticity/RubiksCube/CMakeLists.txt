
# Create a symbolic link for exodiff
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EXODIFF} ${CMAKE_CURRENT_BINARY_DIR}/exodiff)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${SEACAS_EPU} ${CMAKE_CURRENT_BINARY_DIR}/epu)

# Copy Input files from source to binary dir

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube_Implicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube_Implicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube_Implicit_noScale.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube_Implicit_noScale.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube_Implicit_constScale.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube_Implicit_constScale.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube_Implicit_diagScale.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube_Implicit_diagScale.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube_Implicit_absRowSumScale.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube_Implicit_absRowSumScale.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube_Implicit_Materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube_Implicit_Materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube_Explicit.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube_Explicit.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube_Explicit_Materials.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube_Explicit_Materials.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.g
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.g COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.g.4.0
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.g.4.0 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.g.4.1
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.g.4.1 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.g.4.2
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.g.4.2 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.g.4.3
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.g.4.3 COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.exodiff
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.exodiff COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PostProcess.py
               ${CMAKE_CURRENT_BINARY_DIR}/PostProcess.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/PlotHardImplicit.py
               ${CMAKE_CURRENT_BINARY_DIR}/PlotHardImplicit.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RubiksCube.gnuplot
               ${CMAKE_CURRENT_BINARY_DIR}/RubiksCube.gnuplot COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CreateMaterialsFile.py
               ${CMAKE_CURRENT_BINARY_DIR}/CreateMaterialsFile.py COPYONLY)

# Add the tests
if(ALBANY_IFPACK2)
#add_test(CrystalPlasticity_RubiksCube_np1_Tpetra python ../run_exodiff_test.py "${SerialAlbanyT.exe}" RubiksCube.yaml)
#add_test(CrystalPlasticity_RubiksCube_np4_Tpetra python ../run_exodiff_test.py "${AlbanyT.exe}" RubiksCube.yaml)
endif()
