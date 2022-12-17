set(HEADERS_CISM
  ali_driver.H
  ${CISM_INCLUDE_DIR}/DyCoreToGlimmer.H
)

set(SOURCES_CISM
  Albany_CismSTKMeshStruct.cpp
  ali_driver.cpp
  ${CISM_INCLUDE_DIR}/DyCoreToGlimmer.cpp
)
add_library(cismInterface ${Albany_LIBRARY_TYPE} ${SOURCES_CISM})

set_target_properties(cismInterface PROPERTIES PUBLIC_HEADER "${HEADERS_CISM}")
target_link_libraries(cismInterface PUBLIC albanyLib landIce)
if (ALBANY_SUPPRESS_TRILINOS_WARNINGS)
  target_include_directories(cismInterface SYSTEM PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
else()
  target_include_directories(cismInterface PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
endif()
target_include_directories(cismInterface SYSTEM PUBLIC ${CISM_INCLUDE_DIR})

set (CISM_INTERFACE_INCLUDE_DIRS
     ${LANDICE_INCLUDE_DIRS}
     ${CMAKE_CURRENT_SOURCE_DIR}/interface_with_cism)
target_include_directories(cismInterface PUBLIC
                           "$<BUILD_INTERFACE:${CISM_INTERFACE_INCLUDE_DIRS}>"
                            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
install(TARGETS cismInterface EXPORT albany-export
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/"
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
