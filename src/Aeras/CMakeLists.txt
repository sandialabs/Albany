##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
# Aeras

  SET(SOURCES 
       evaluators/Aeras_ShallowWaterResid.cpp
       problems/Aeras_ShallowWaterProblem.cpp
       evaluators/Aeras_XZScalarAdvectionResid.cpp
       problems/Aeras_XZScalarAdvectionProblem.cpp
   )
   
  SET(HEADERS
       evaluators/Aeras_ShallowWaterResid_Def.hpp
       evaluators/Aeras_ShallowWaterResid.hpp
       problems/Aeras_ShallowWaterProblem.hpp
       evaluators/Aeras_XZScalarAdvectionResid_Def.hpp
       evaluators/Aeras_XZScalarAdvectionResid.hpp
       problems/Aeras_XZScalarAdvectionProblem.hpp
  )
  
include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
                     ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators 
                     ${Albany_SOURCE_DIR}/src/problems ${Albany_SOURCE_DIR}/src/responses 
                     ./problems ./evaluators
                    )

add_library(Aeras ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})
