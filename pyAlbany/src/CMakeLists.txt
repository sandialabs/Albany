##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

set(ALB_TRILINOS_LIBS ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

set (ALBANY_INCLUDE_DIRS
     ${CMAKE_CURRENT_BINARY_DIR}
     ${Albany_SOURCE_DIR}/src
     ${Albany_SOURCE_DIR}/src/evaluators
     ${Albany_SOURCE_DIR}/src/evaluators/bc
     ${Albany_SOURCE_DIR}/src/evaluators/gather
     ${Albany_SOURCE_DIR}/src/evaluators/interpolation
     ${Albany_SOURCE_DIR}/src/evaluators/pde
     ${Albany_SOURCE_DIR}/src/evaluators/response
     ${Albany_SOURCE_DIR}/src/evaluators/scatter
     ${Albany_SOURCE_DIR}/src/evaluators/state
     ${Albany_SOURCE_DIR}/src/evaluators/utility
     ${Albany_SOURCE_DIR}/src/problems
     ${Albany_SOURCE_DIR}/src/responses
     ${Albany_SOURCE_DIR}/src/disc/stk
     ${Albany_SOURCE_DIR}/src/disc
     ${Albany_SOURCE_DIR}/src/utility
     ${CMAKE_BINARY_DIR}/src
)

set(PyAlbany_SRCS
  Albany_PyUtils.cpp
)
set(PyAlbany_HDRS
  Albany_PyUtils.hpp
)

set(PybindAlbany_SRCS
  Albany_Pybind11.cpp
  Albany_Pybind11_Comm.cpp
  Albany_Pybind11_ParallelEnv.cpp
  Albany_Pybind11_ParameterList.cpp
  Albany_Pybind11_Timer.cpp
  Albany_Pybind11_Tpetra.cpp
)
set(PybindAlbany_HDRS
  Albany_Pybind11_Comm.hpp
)

set(AI_SRCS
  Albany_Interface.cpp
)
set(AI_HDRS
  Albany_Interface.hpp
)

set(ALBANY_LIBRARIES 
  albanyLib
)

list(APPEND ALBANY_LIBRARIES corePDEs)

if(ENABLE_DEMO_PDES)
  list(APPEND ALBANY_LIBRARIES demoPDEs)
endif()

if (ENABLE_LANDICE)
  list (APPEND ALBANY_INCLUDE_DIRS
    ${Albany_SOURCE_DIR}/src/landIce
    ${Albany_SOURCE_DIR}/src/landIce/evaluators
    ${Albany_SOURCE_DIR}/src/landIce/problems
  )

  list (APPEND ALBANY_LIBRARIES landIce)

  if (ENABLE_MPAS_INTERFACE)
    list (APPEND ALBANY_LIBRARIES mpasInterface)
  endif()

  if (ENABLE_CISM_INTERFACE)
    list (APPEND ALBANY_LIBRARIES cismInterface)
  endif()
endif()

# Note: Albany_EXTRA_LIBRARIES is to allow users to specify
# libraries that one cannot get out of Trilinos directly.
set(TPL_LIBRARIES
  ${ALB_TRILINOS_LIBS}
  ${Trilinos_EXTRA_LD_FLAGS}
  ${Albany_EXTRA_LIBRARIES}
  ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}
)

add_library(pyalbanyLib ${Albany_LIBRARY_TYPE} ${PyAlbany_SRCS})
set_target_properties(pyalbanyLib PROPERTIES HEADER "${PyAlbany_HDRS}")
target_link_libraries(pyalbanyLib ${Trilinos_LIBRARIES})
if (ALBANY_SUPPRESS_TRILINOS_WARNINGS)
  target_include_directories(pyalbanyLib SYSTEM PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
else()
  target_include_directories(pyalbanyLib PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
endif()
target_include_directories(pyalbanyLib PUBLIC
                            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                           "$<BUILD_INTERFACE:${ALBANY_INCLUDE_DIRS}>")
list(APPEND ALBANY_LIBRARIES pyalbanyLib)

include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.10.4
	GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(pybind11)

EXECUTE_PROCESS(COMMAND
  ${PYTHON_EXECUTABLE} -c "import mpi4py; print(mpi4py.get_include())"
  OUTPUT_VARIABLE Mpi4Py_INCLUDE_DIR
  ERROR_VARIABLE  Mpi4Py_INCLUDE_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

add_library(albanyPyInterface ${Albany_LIBRARY_TYPE} ${AI_SRCS})
set_target_properties(albanyPyInterface PROPERTIES HEADER "${AI_HDRS}")
target_link_libraries(albanyPyInterface ${ALBANY_LIBRARIES} ${TPL_LIBRARIES})
if (ALBANY_SUPPRESS_TRILINOS_WARNINGS)
  target_include_directories(albanyPyInterface SYSTEM PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
else()
  target_include_directories(albanyPyInterface PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
endif()

target_include_directories(albanyPyInterface PUBLIC
                            "$<BUILD_INTERFACE:${ALBANY_INCLUDE_DIRS}>"
                            $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>)

pybind11_add_module(AlbanyInterface ${PybindAlbany_SRCS})
target_include_directories(AlbanyInterface PUBLIC ${Trilinos_INCLUDE_DIRS} ${Mpi4Py_INCLUDE_DIR})
target_compile_features(AlbanyInterface PUBLIC cxx_std_11)
#target_link_libraries(AlbanyInterface PRIVATE ${ALBANY_LIBRARIES} ${TPL_LIBRARIES})
target_link_libraries(AlbanyInterface PRIVATE ${ALBANY_LIBRARIES} albanyPyInterface)
set_target_properties(AlbanyInterface PROPERTIES SUFFIX ".so")

install(TARGETS pyalbanyLib
  LIBRARY DESTINATION ${LIB_INSTALL_DIR}
  ARCHIVE DESTINATION ${LIB_INSTALL_DIR})
install(TARGETS albanyPyInterface
  LIBRARY DESTINATION ${LIB_INSTALL_DIR}
  ARCHIVE DESTINATION ${LIB_INSTALL_DIR})
install(TARGETS AlbanyInterface
  LIBRARY DESTINATION ${PyAlbany_INSTALL_DIR}
  ARCHIVE DESTINATION ${PyAlbany_INSTALL_DIR})

IF (ALBANY_PYTHON_TESTS)
  add_custom_command(TARGET AlbanyInterface POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/AlbanyInterface.so ${CMAKE_CURRENT_BINARY_DIR}/../PyAlbany/.
      COMMENT "Copy ${PROJECT_BINARY_DIR}/src/AlbanyInterface.so"
  )
ENDIF()
