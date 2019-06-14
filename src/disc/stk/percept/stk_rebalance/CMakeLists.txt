
SET(SOURCES GeomDecomp.cpp Partition.cpp Rebalance.cpp RebalanceUtils.cpp ZoltanPartition.cpp)
SET(HEADERS GeomDecomp.hpp Partition.hpp Rebalance.hpp RebalanceUtils.hpp ZoltanPartition.hpp)

include_directories(
  ${ALBANY_INCLUDE_DIRS}
  ${Albany_SOURCE_DIR}/src/disc/stk
)
include_directories(SYSTEM
        ${Trilinos_INCLUDE_DIRS}
        ${Trilinos_TPL_INCLUDE_DIRS})
add_library(albanySTKRebalance ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

target_link_libraries(albanySTKRebalance ${Trilinos_LIBRARIES})

IF (INSTALL_ALBANY)
  install(TARGETS albanySTKRebalance EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()

