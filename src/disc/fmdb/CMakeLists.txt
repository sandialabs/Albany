#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

# SCOREC FMDB

SET(SOURCES 
  Albany_FMDBMeshStruct.cpp
  Albany_FMDBDiscretization.cpp
  )
SET(HEADERS 
  Albany_FMDBMeshStruct.hpp
  Albany_FMDBDiscretization.hpp
  )

include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
  ${FMDB_INCLUDE_DIR}
  ${Albany_SOURCE_DIR}/src 
  )

add_library(fmdb ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

IF (INSTALL_ALBANY)
  install(TARGETS fmdb EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()