//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_ProblemFactory.hpp"

#include "LandIce_SchoofFit.hpp"
#include "LandIce_Stokes.hpp"
#include "LandIce_StokesFO.hpp"
#include "LandIce_StokesL1L2.hpp"
#include "LandIce_Hydrology.hpp"
#include "LandIce_Enthalpy.hpp"
#include "LandIce_StokesFOThermoCoupled.hpp"
#include "LandIce_LaplacianSampling.hpp"
#include "LandIce_StokesFOThickness.hpp"

namespace LandIce
{

ProblemFactory::ProblemFactory (const Teuchos::RCP<Teuchos::ParameterList>& problemParams_,
                                const Teuchos::RCP<Teuchos::ParameterList>& discretizationParams_,
                                const Teuchos::RCP<ParamLib>& paramLib_) :
  problemParams(problemParams_),
  discretizationParams(discretizationParams_),
  paramLib(paramLib_)
{
  // Nothing to be done here
}

bool ProblemFactory::hasProblem (const std::string& problemName)
{
  if (problemName == "LandIce Stokes" ||
      problemName == "LandIce Stokes 3D" ||
      problemName == "LandIce Stokes 2D" ||
      problemName == "LandIce Stokes First Order 2D" ||
      problemName == "LandIce Stokes FO 2D" ||
      problemName == "LandIce Stokes First Order 2D XZ" ||
      problemName == "LandIce Stokes FO 2D XZ" ||
      problemName == "LandIce Stokes First Order 3D" ||
      problemName == "LandIce Stokes FO 3D" ||
      problemName == "LandIce Coupled FO H 3D" ||
      problemName == "LandIce Stokes L1L2 2D" ||
      problemName == "LandIce Hydrology 2D" ||
      problemName == "LandIce Enthalpy 3D" ||
      problemName == "LandIce Stokes FO Thermo Coupled 3D" ||
      problemName == "LandIce Schoof Fit" ||
      problemName == "LandIce Laplacian Sampling")
  {
    return true;
  }

  return false;
}

Teuchos::RCP<Albany::AbstractProblem>
ProblemFactory::create() const
{
  Teuchos::RCP<Albany::AbstractProblem> problem;
  using Teuchos::rcp;

  std::string& method = problemParams->get("Name", "");

  if (method == "LandIce Stokes" || method == "LandIce Stokes 3D" ) {
    problem = rcp(new LandIce::Stokes(problemParams, paramLib, 3));
  }
  else if (method == "LandIce Stokes 2D" ) {
    problem = rcp(new LandIce::Stokes(problemParams, paramLib, 2));
  }
  else if (method == "LandIce Stokes First Order 2D" || method == "LandIce Stokes FO 2D" ||
           method == "LandIce Stokes First Order 2D XZ" || method == "LandIce Stokes FO 2D XZ") {
    problem = rcp(new LandIce::StokesFO(problemParams, discretizationParams, paramLib, 2));
  }
  else if (method == "LandIce Stokes First Order 3D" || method == "LandIce Stokes FO 3D" ) {
    problem = rcp(new LandIce::StokesFO(problemParams, discretizationParams, paramLib, 3));
  }
  else if (method == "LandIce Coupled FO H 3D" ) {
    problem = rcp(new LandIce::StokesFOThickness(problemParams, discretizationParams, paramLib, 3));
  }
  else if (method == "LandIce Stokes L1L2 2D") {
    problem = rcp(new LandIce::StokesL1L2(problemParams, paramLib, 2));
  }
  else if (method == "LandIce Hydrology 2D") {
    problem = rcp(new LandIce::Hydrology(problemParams, discretizationParams, paramLib, 2));
  }
  else if (method == "LandIce Enthalpy 3D") {
    problem = rcp(new LandIce::Enthalpy(problemParams, discretizationParams, paramLib, 3));
  }
  else if (method == "LandIce Stokes FO Thermo Coupled 3D") {
    problem = rcp(new LandIce::StokesFOThermoCoupled(problemParams, discretizationParams, paramLib, 3));
  }
  else if (method == "LandIce Schoof Fit") {
    problem = rcp(new LandIce::SchoofFit(problemParams, paramLib, 2));
  }
  else if (method == "LandIce Laplacian Sampling") {
    problem = rcp(new LandIce::LaplacianSampling(problemParams, discretizationParams, paramLib, 2));
  }

  return problem;
}

} // Namespace LandIce
