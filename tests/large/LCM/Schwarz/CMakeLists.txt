##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

add_subdirectory(Cubes)
add_subdirectory(CuboidsDebug)
if(ALBANY_DTK)
add_subdirectory(ParallelCubes)
endif()
add_subdirectory(NotchedCylinder)
add_subdirectory(LaserWeld)

