#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

# SCOREC PUMI

  SET(SOURCES
    Albany_PUMIMeshStruct.cpp
    Albany_PUMIDiscretization.cpp
    Albany_PUMIQPData.cpp
    Albany_PUMINodeData.cpp
    Albany_PUMIVtk.cpp
    Albany_PUMIExodus.cpp
    Albany_PUMIOutput.cpp
    )
  SET(HEADERS
    Albany_AbstractPUMIDiscretization.hpp
    Albany_PUMIMeshStruct.hpp
    Albany_PUMIDiscretization.hpp
    Albany_PUMIQPData.hpp
    Albany_PUMIQPData_Def.hpp
    Albany_PUMINodeData.hpp
    Albany_PUMINodeData_Def.hpp
    )

include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src
  )

add_library(pumi ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

IF (INSTALL_ALBANY)
  install(TARGETS pumi EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
