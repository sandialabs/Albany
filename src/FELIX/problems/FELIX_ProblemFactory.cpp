//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "FELIX_ProblemFactory.hpp"

#include "FELIX_SchoofFit.hpp"
#include "FELIX_Stokes.hpp"
#include "FELIX_StokesFO.hpp"
#include "FELIX_StokesL1L2.hpp"
#include "FELIX_Hydrology.hpp"
#include "FELIX_Elliptic2D.hpp"
#include "FELIX_Enthalpy.hpp"
#include "FELIX_PopulateMesh.hpp"
#include "FELIX_StokesFOThermoCoupled.hpp"
#include "FELIX_LaplacianSampling.hpp"

#ifdef ALBANY_EPETRA
#include "FELIX_StokesFOHydrology.hpp"
#include "FELIX_StokesFOThickness.hpp"
#endif

namespace FELIX
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
  if (problemName == "FELIX Stokes" ||
      problemName == "FELIX Stokes 3D" ||
      problemName == "FELIX Stokes 2D" ||
      problemName == "FELIX Stokes First Order 2D" ||
      problemName == "FELIX Stokes FO 2D" ||
      problemName == "FELIX Stokes First Order 2D XZ" ||
      problemName == "FELIX Stokes FO 2D XZ" ||
      problemName == "FELIX Stokes First Order 3D" ||
      problemName == "FELIX Stokes FO 3D" ||
      problemName == "FELIX Coupled FO H 3D" ||
      problemName == "FELIX Coupled FO Hydrology 3D" ||
      problemName == "FELIX Stokes L1L2 2D" ||
      problemName == "FELIX Hydrology 2D" ||
      problemName == "FELIX Elliptic 2D" ||
      problemName == "FELIX Enthalpy 3D" ||
      problemName == "FELIX Populate Mesh" ||
      problemName == "FELIX Stokes FO Thermo Coupled 3D" ||
      problemName == "FELIX Schoof Fit" ||
      problemName == "FELIX Laplacian Sampling")
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

  if (method == "FELIX Stokes" || method == "FELIX Stokes 3D" ) {
    problem = rcp(new FELIX::Stokes(problemParams, paramLib, 3));
  }
  else if (method == "FELIX Stokes 2D" ) {
    problem = rcp(new FELIX::Stokes(problemParams, paramLib, 2));
  }
  else if (method == "FELIX Stokes First Order 2D" || method == "FELIX Stokes FO 2D" ||
           method == "FELIX Stokes First Order 2D XZ" || method == "FELIX Stokes FO 2D XZ") {
    problem = rcp(new FELIX::StokesFO(problemParams, discretizationParams, paramLib, 2));
  }
  else if (method == "FELIX Stokes First Order 3D" || method == "FELIX Stokes FO 3D" ) {
    problem = rcp(new FELIX::StokesFO(problemParams, discretizationParams, paramLib, 3));
  }
  else if (method == "FELIX Coupled FO H 3D" ) {
#ifdef ALBANY_EPETRA
    problem = rcp(new FELIX::StokesFOThickness(problemParams, paramLib, 3));
#else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, " **** FELIX Coupled FO H requires Epetra, recompile with -DENABLE_ALBANY_EPETRA_EXE ****\n");
#endif
  }
  else if (method == "FELIX Coupled FO Hydrology 3D" ) {
#ifdef ALBANY_EPETRA
    problem = rcp(new FELIX::StokesFOHydrology(problemParams, paramLib, 3));
#else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, " **** FELIX Coupled FO Hydrology requires Epetra, recompile with -DENABLE_ALBANY_EPETRA_EXE ****\n");
#endif
  }
  else if (method == "FELIX Stokes L1L2 2D") {
    problem = rcp(new FELIX::StokesL1L2(problemParams, paramLib, 2));
  }
  else if (method == "FELIX Hydrology 2D") {
    problem = rcp(new FELIX::Hydrology(problemParams, paramLib, 2));
  }
  else if (method == "FELIX Elliptic 2D") {
    problem = rcp(new FELIX::Elliptic2D(problemParams, paramLib, 1));
  }
  else if (method == "FELIX Enthalpy 3D") {
    problem = rcp(new FELIX::Enthalpy(problemParams, paramLib, 3));
  }
  else if (method == "FELIX Populate Mesh") {
    problem = rcp(new FELIX::PopulateMesh(problemParams, discretizationParams, paramLib));
  }
  else if (method == "FELIX Stokes FO Thermo Coupled 3D") {
    problem = rcp(new FELIX::StokesFOThermoCoupled(problemParams, paramLib, 3));
  }
  else if (method == "FELIX Schoof Fit") {
    problem = rcp(new FELIX::SchoofFit(problemParams, paramLib, 2));
  }
  else if (method == "FELIX Laplacian Sampling") {
    problem = rcp(new FELIX::LaplacianSampling(problemParams, discretizationParams, paramLib, 2));
  }
  return problem;
}

} // Namespace FELIX
