
SET(SOURCES GeomDecomp.cpp Partition.cpp Rebalance.cpp RebalanceUtils.cpp ZoltanPartition.cpp)
SET(HEADERS GeomDecomp.hpp Partition.hpp Rebalance.hpp RebalanceUtils.hpp ZoltanPartition.hpp)

add_library(albanySTKRebalance ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
target_link_libraries(albanySTKRebalance ${Trilinos_LIBRARIES})

IF (INSTALL_ALBANY)
  install(TARGETS albanySTKRebalance EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()

