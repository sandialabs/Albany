##*****************************************************************//
##    Albany 2.0:  Copyright 2012 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//
# Aeras

  SET(SOURCES 
       evaluators/Aeras_ComputeBasisFunctions.cpp
       evaluators/Aeras_ShallowWaterResid.cpp
       evaluators/Aeras_Atmosphere.cpp
       evaluators/Aeras_Atmosphere_Moisture.cpp
       evaluators/Aeras_GatherCoordinateVector.cpp
       evaluators/Aeras_SurfaceHeight.cpp
       problems/Aeras_ShallowWaterProblem.cpp
       evaluators/Aeras_XZScalarAdvectionResid.cpp
       problems/Aeras_XZScalarAdvectionProblem.cpp
       evaluators/Aeras_ScalarAdvectionResid.cpp
       evaluators/Aeras_XScalarAdvectionResid.cpp
       problems/Aeras_XScalarAdvectionProblem.cpp
       evaluators/Aeras_XZHydrostatic_VelResid.cpp
       evaluators/Aeras_XZHydrostatic_TemperatureResid.cpp
       evaluators/Aeras_XZHydrostatic_SPressureResid.cpp
       evaluators/Aeras_XZHydrostatic_TracerResid.cpp
       evaluators/Aeras_XZHydrostatic_Density.cpp
       evaluators/Aeras_XZHydrostatic_EtaDotPi.cpp
       evaluators/Aeras_XZHydrostatic_DensityWeightedVelx.cpp
       evaluators/Aeras_XZHydrostatic_Pressure.cpp
       evaluators/Aeras_XZHydrostatic_KineticEnergy.cpp
       evaluators/Aeras_XZHydrostatic_UTracer.cpp
       problems/Aeras_XZHydrostaticProblem.cpp
       problems/Aeras_Layouts.cpp
       evaluators/Aeras_GatherSolution.cpp
       evaluators/Aeras_ScatterResidual.cpp
       evaluators/Aeras_DOFInterpolation.cpp
       evaluators/Aeras_DOFGradInterpolation.cpp
       evaluators/Aeras_Atmosphere_Moisture.cpp
   )
   
  SET(HEADERS
       evaluators/Aeras_ComputeBasisFunctions_Def.hpp
       evaluators/Aeras_ComputeBasisFunctions.hpp
       evaluators/Aeras_Atmosphere_Def.hpp
       evaluators/Aeras_Atmosphere.hpp
       evaluators/Aeras_Atmosphere_Moisture_Def.hpp
       evaluators/Aeras_Atmosphere_Moisture.hpp
       evaluators/Aeras_ShallowWaterConstants.hpp
       evaluators/Aeras_SurfaceHeight.hpp
       evaluators/Aeras_SurfaceHeight_Def.hpp
       evaluators/Aeras_GatherCoordinateVector_Def.hpp
       evaluators/Aeras_GatherCoordinateVector.hpp
       problems/Aeras_ShallowWaterProblem.hpp
       evaluators/Aeras_XZScalarAdvectionResid_Def.hpp
       evaluators/Aeras_XZScalarAdvectionResid.hpp
       problems/Aeras_XZScalarAdvectionProblem.hpp
       evaluators/Aeras_ScalarAdvectionResid_Def.hpp
       evaluators/Aeras_ScalarAdvectionResid.hpp
       evaluators/Aeras_XScalarAdvectionResid_Def.hpp
       evaluators/Aeras_XScalarAdvectionResid.hpp
       problems/Aeras_XScalarAdvectionProblem.hpp
       problems/Aeras_XZHydrostaticProblem.hpp
       evaluators/Aeras_XZHydrostatic_VelResid_Def.hpp
       evaluators/Aeras_XZHydrostatic_VelResid.hpp
       evaluators/Aeras_XZHydrostatic_TemperatureResid_Def.hpp
       evaluators/Aeras_XZHydrostatic_TemperatureResid.hpp
       evaluators/Aeras_XZHydrostatic_SPressureResid_Def.hpp
       evaluators/Aeras_XZHydrostatic_SPressureResid.hpp
       evaluators/Aeras_XZHydrostatic_TracerResid_Def.hpp
       evaluators/Aeras_XZHydrostatic_TracerResid.hpp
       evaluators/Aeras_XZHydrostatic_Pressure_Def.hpp
       evaluators/Aeras_XZHydrostatic_Pressure.hpp
       evaluators/Aeras_XZHydrostatic_Density_Def.hpp
       evaluators/Aeras_XZHydrostatic_Density.hpp
       evaluators/Aeras_XZHydrostatic_EtaDotPi_Def.hpp
       evaluators/Aeras_XZHydrostatic_EtaDotPi.hpp
       evaluators/Aeras_XZHydrostatic_DensityWeightedVelx_Def.hpp
       evaluators/Aeras_XZHydrostatic_DensityWeightedVelx.hpp
       evaluators/Aeras_XZHydrostatic_KineticEnergy_Def.hpp
       evaluators/Aeras_XZHydrostatic_KineticEnergy.hpp
       evaluators/Aeras_XZHydrostatic_UTracer_Def.hpp
       evaluators/Aeras_XZHydrostatic_UTracer.hpp
       problems/Aeras_Layouts.hpp
       evaluators/Aeras_GatherSolution.hpp
       evaluators/Aeras_GatherSolution_Def.hpp
       evaluators/Aeras_ScatterResidual.hpp
       evaluators/Aeras_ScatterResidual_Def.hpp
       evaluators/Aeras_DOFInterpolation.hpp
       evaluators/Aeras_DOFInterpolation_Def.hpp
       evaluators/Aeras_DOFGradInterpolation.hpp
       evaluators/Aeras_DOFGradInterpolation_Def.hpp
       evaluators/Aeras_Atmosphere_Moisture.hpp
       evaluators/Aeras_Atmosphere_Moisture_Def.hpp
  )
  
include_directories (${Trilinos_INCLUDE_DIRS}  ${Trilinos_TPL_INCLUDE_DIRS}
                     ${Albany_SOURCE_DIR}/src ${Albany_SOURCE_DIR}/src/evaluators 
                     ${Albany_SOURCE_DIR}/src/problems ${Albany_SOURCE_DIR}/src/responses 
                     ./problems ./evaluators
                    )

add_library(Aeras ${Albany_LIBRARY_TYPE} ${SOURCES} ${HEADERS})

set_target_properties(Aeras PROPERTIES PUBLIC_HEADER "${HEADERS}")

IF (INSTALL_ALBANY)
  install(TARGETS Aeras EXPORT albany-export
    LIBRARY DESTINATION "${LIB_INSTALL_DIR}/"
    ARCHIVE DESTINATION "${LIB_INSTALL_DIR}/"
    PUBLIC_HEADER DESTINATION "${INCLUDE_INSTALL_DIR}")
ENDIF()
