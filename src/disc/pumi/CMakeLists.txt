#*****************************************************************//
#    Albany 3.0:  Copyright 2016 Sandia Corporation               //
#    This Software is released under the BSD license detailed     //
#    in the file "license.txt" in the top-level Albany directory  //
#*****************************************************************//

# SCOREC PUMI

SET(SOURCES
  Albany_APFMeshStruct.cpp
  Albany_APFDiscretization.cpp
  Albany_PUMIQPData.cpp
  Albany_PUMINodeData.cpp
  Albany_PUMIVtk.cpp
  Albany_PUMIOutput.cpp
  Albany_APFStabilizedStress.cpp
  )

SET(HEADERS
  Albany_APFMeshStruct.hpp
  Albany_APFDiscretization.hpp
  Albany_PUMIQPData.hpp
  Albany_PUMIQPData_Def.hpp
  Albany_PUMINodeData.hpp
  Albany_PUMINodeData_Def.hpp
  )

IF(ALBANY_SCOREC)
  SET(SOURCES
      ${SOURCES}
      Albany_PUMIMeshStruct.cpp
      Albany_PUMIDiscretization.cpp
      Albany_PUMIBuildBox.cpp
     )
  SET(HEADERS
      ${HEADERS}
      Albany_PUMIMeshStruct.hpp
      Albany_PUMIDiscretization.hpp
     )
ENDIF()

IF(ALBANY_AMP)
  SET(SOURCES
      ${SOURCES}
      Albany_SimMeshStruct.cpp
      Albany_SimDiscretization.cpp
      Albany_SimOutput.cpp
     )
  SET(HEADERS
      ${HEADERS}
      Albany_SimMeshStruct.hpp
      Albany_SimDiscretization.hpp
      Albany_SimOutput.hpp
     )
ENDIF()

IF(ALBANY_HAVE_STK)
  SET(SOURCES ${SOURCES} Albany_PUMIExodus.cpp)
  SET(HEADERS ${HEADERS} Albany_PUMIExodus.hpp)
ENDIF()


include_directories(
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src
  )

add_library(pumi ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

IF (INSTALL_ALBANY)
  install(TARGETS pumi EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
