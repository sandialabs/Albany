find_package(ParaView REQUIRED COMPONENTS vtkPVPythonCatalyst)
include("${PARAVIEW_USE_FILE}")

set(SOURCES
  Albany_Catalyst_Adapter.cpp
  Albany_Catalyst_Decorator.cpp
  Albany_Catalyst_EpetraDataArray.cpp
  Albany_Catalyst_Grid.cpp
)
set(HEADERS
  Albany_Catalyst_Adapter.hpp
  Albany_Catalyst_Decorator.hpp
  Albany_Catalyst_EpetraDataArray.hpp
  Albany_Catalyst_Grid.hpp
  Albany_Catalyst_TeuchosArrayRCPDataArray.hpp
  Albany_Catalyst_TeuchosArrayRCPDataArrayDecl.hpp
)

add_library(albanyCatalyst ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
target_link_libraries(albanyCatalyst LINK_PRIVATE vtkPVPythonCatalyst)

if(INSTALL_ALBANY)
  install(
    TARGETS albanyCatalyst
    EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}"
  )
endif()
