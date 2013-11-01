##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
# AERAS

  SET(SOURCES 
       evaluators/AERAS_EulerResid.cpp
       problems/AERAS_EulerProblem.cpp
       evaluators/AERAS_XZScalarAdvectionResid.cpp
       problems/AERAS_XZScalarAdvectionProblem.cpp
   )
   
  SET(HEADERS
       evaluators/AERAS_EulerResid_Def.hpp
       evaluators/AERAS_EulerResid.hpp
       problems/AERAS_EulerProblem.hpp
       evaluators/AERAS_XZScalarAdvectionResid_Def.hpp
       evaluators/AERAS_XZScalarAdvectionResid.hpp
       problems/AERAS_XZScalarAdvectionProblem.hpp
  )
  
include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
                     ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators 
                     ${Albany_SOURCE_DIR}/src/problems ${Albany_SOURCE_DIR}/src/responses 
                     ./problems ./evaluators
                    )

add_library(AERAS ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
