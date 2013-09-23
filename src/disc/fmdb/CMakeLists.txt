#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

# SCOREC FMDB

  SET(SOURCES
    AlbPUMI_FMDBMeshStruct.cpp
    AlbPUMI_FMDBDiscretization.cpp
    AlbPUMI_QPData.cpp
    AlbPUMI_FMDBVtk.cpp
    AlbPUMI_FMDBExodus.cpp
    )
  SET(HEADERS
    AlbPUMI_AbstractPUMIDiscretization.hpp
    AlbPUMI_FMDBMeshStruct.hpp
    AlbPUMI_FMDBDiscretization.hpp
    AlbPUMI_FMDBDiscretization_Def.hpp
    AlbPUMI_QPData.hpp
    AlbPUMI_QPData_Def.hpp
    AlbPUMI_FMDBVtk.hpp
    AlbPUMI_FMDBExodus.hpp
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
