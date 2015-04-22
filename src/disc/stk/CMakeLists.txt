#*****************************************************************//
#    Albany 2.0:  Copyright 2012 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

#stk
SET(SOURCES 
  Albany_STKDiscretization.cpp
  Albany_GenericSTKMeshStruct.cpp
  Albany_GenericSTKFieldContainer.cpp
  Albany_OrdinarySTKFieldContainer.cpp
  Albany_MultiSTKFieldContainer.cpp
  Albany_STKNodeFieldContainer.cpp
  Albany_TmplSTKMeshStruct.cpp
  Albany_FromCubitSTKMeshStruct.cpp
  Albany_IossSTKMeshStruct.cpp
  Albany_AsciiSTKMesh2D.cpp
  Albany_AsciiSTKMeshStruct.cpp
  Albany_STKNodeSharing.cpp
  )
SET(HEADERS 
  Albany_NodalGraphUtils.hpp
  Albany_STKDiscretization.hpp
  Albany_AbstractSTKMeshStruct.hpp
  Albany_AbstractSTKFieldContainer.hpp
  Albany_GenericSTKMeshStruct.hpp
  Albany_GenericSTKFieldContainer.hpp
  Albany_GenericSTKFieldContainer_Def.hpp
  Albany_OrdinarySTKFieldContainer.hpp
  Albany_OrdinarySTKFieldContainer_Def.hpp
  Albany_MultiSTKFieldContainer.hpp
  Albany_MultiSTKFieldContainer_Def.hpp
  Albany_STKNodeFieldContainer.hpp
  Albany_STKNodeFieldContainer_Def.hpp
  Albany_TmplSTKMeshStruct.hpp
  Albany_TmplSTKMeshStruct_Def.hpp
  Albany_FromCubitSTKMeshStruct.hpp
  Albany_IossSTKMeshStruct.hpp
  Albany_AsciiSTKMesh2D.hpp
  Albany_AsciiSTKMeshStruct.hpp
  Albany_STKNodeSharing.hpp
  )

IF (ALBANY_FELIX) 
  SET(SOURCES ${SOURCES} Albany_ExtrudedSTKMeshStruct.cpp)
  SET(HEADERS ${HEADERS} Albany_ExtrudedSTKMeshStruct.hpp)
ENDIF()

IF (ALBANY_AERAS)
  SET(SOURCES ${SOURCES} Aeras_SpectralDiscretization.cpp
                         Aeras_SpectralOutputSTKMeshStruct.cpp)
  SET(HEADERS ${HEADERS} Aeras_SpectralDiscretization.hpp
                         Aeras_SpectralOutputSTKMeshStruct.hpp)
ENDIF() 

include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src 
  )

add_library(albanySTK ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
target_link_libraries(albanySTK ${Trilinos_LIBRARIES})

set_target_properties(albanySTK PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF (INSTALL_ALBANY)
  install(TARGETS albanySTK EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()

# Can remove following once stk_rebalance is in trilinos again
IF (ALBANY_STK_REBALANCE)
  add_subdirectory(percept)
ENDIF()
