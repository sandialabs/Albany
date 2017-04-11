##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

add_subdirectory(Cubes)
if(ALBANY_DTK)
add_subdirectory(ParallelCubes)
endif()
#
# These are big examples. Any problems in the Schwarz implementation
# will be caught by the cubes problem above, so we disable them.
# 
#add_subdirectory(Overlap)
#add_subdirectory(NotchedCylinder)
#add_subdirectory(LaserWeld)

