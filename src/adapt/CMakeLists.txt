##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//


# Adaptation

SET(SOURCES
    AAdapt_AbstractAdapter.cpp
    AAdapt_AdaptiveSolutionManager.cpp
    AAdapt_InitialCondition.cpp
    AAdapt_AnalyticFunction.cpp
    AAdapt_SolutionObserver.cpp
   )

SET(HEADERS
    AAdapt_AbstractAdapter.hpp
    AAdapt_AdaptiveSolutionManager.hpp
    AAdapt_InitialCondition.hpp
    AAdapt_AnalyticFunction.hpp
    AAdapt_SolutionObserver.hpp
   )

include_directories(${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
                    ${Albany_SOURCE_DIR} / src ${Albany_SOURCE_DIR} / src / evaluators
                    ${Albany_SOURCE_DIR} / src / problems ${Albany_SOURCE_DIR} / src / responses
                    ${Albany_SOURCE_DIR} / src / disc
                    ${Albany_SOURCE_DIR} / src / utility 
                    ${Albany_SOURCE_DIR} / src / disc / stk 
                   )

add_library(albanyAdapt ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

set_target_properties(albanyAdapt PROPERTIES PUBLIC_HEADER "${HEADERS}")

target_link_libraries(albanyAdapt)

IF(INSTALL_ALBANY)
install(TARGETS albanyAdapt EXPORT albany-export
        LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
        ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
        PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
