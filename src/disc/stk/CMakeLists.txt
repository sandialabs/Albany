#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

#stk
SET(SOURCES 
  Albany_STKDiscretization.cpp
  Albany_GenericSTKMeshStruct.cpp
  Albany_TmplSTKMeshStruct.cpp
  Albany_FromCubitSTKMeshStruct.cpp
  Albany_IossSTKMeshStruct.cpp
  Albany_AsciiSTKMeshStruct.cpp
  )
SET(HEADERS 
  Albany_STKDiscretization.hpp
  Albany_AbstractSTKMeshStruct.hpp
  Albany_GenericSTKMeshStruct.hpp
  Albany_TmplSTKMeshStruct.hpp
  Albany_FromCubitSTKMeshStruct.hpp
  Albany_IossSTKMeshStruct.hpp
  Albany_AsciiSTKMeshStruct.hpp
  )

include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src 
  )

add_library(albanySTK ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

set_target_properties(albanySTK PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF (INSTALL_ALBANY)
  install(TARGETS albanySTK EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
