set(SOURCES_MPAS
  Albany_MpasSTKMeshStruct.cpp
  Interface.cpp
)
set(HEADERS_MPAS
  Albany_MpasSTKMeshStruct.hpp
)

add_library(mpasInterface ${SOURCES_MPAS})
target_link_libraries(mpasInterface PUBLIC albanyLib LandIce)
set_target_properties(mpasInterface PROPERTIES PUBLIC_HEADER "${HEADERS_MPAS}")

if (ALBANY_SUPPRESS_TRILINOS_WARNINGS)
  target_include_directories(mpasInterface SYSTEM PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
else()
  target_include_directories(mpasInterface PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
endif()

set (MPAS_INTERFACE_INCLUDE_DIRS
     ${LANDICE_INCLUDE_DIRS}
     ${CMAKE_CURRENT_SOURCE_DIR}/interface_with_mpas)
target_include_directories(mpasInterface PUBLIC
                           "$<BUILD_INTERFACE:${MPAS_INTERFACE_INCLUDE_DIRS}>"
                            $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>)
if (INSTALL_ALBANY)
  install(TARGETS mpasInterface
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
endif()
