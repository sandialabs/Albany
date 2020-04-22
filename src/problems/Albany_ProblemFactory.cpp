//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Albany_ProblemFactory.hpp"

// Always enable HeatProblem and SideLaplacianProblem
#include "Albany_HeatProblem.hpp"
#include "Albany_ThermalProblem.hpp"
#include "Albany_PopulateMesh.hpp"
#include "Albany_SideLaplacianProblem.hpp"

#ifdef ALBANY_DEMO_PDES
#include "Albany_CahnHillProblem.hpp"
#include "Albany_Helmholtz2DProblem.hpp"
#include "Albany_NavierStokes.hpp"
#include "Albany_LinComprNSProblem.hpp"
#include "Albany_AdvDiffProblem.hpp"
#include "Albany_ReactDiffSystem.hpp"
#include "Albany_ComprNSProblem.hpp"
#include "Albany_ODEProblem.hpp"
#include "Albany_PNPProblem.hpp"
#include "Albany_ThermoElectrostaticsProblem.hpp"
#endif

#ifdef ALBANY_LANDICE
#include "LandIce/problems/LandIce_ProblemFactory.hpp"
#endif

Albany::ProblemFactory::ProblemFactory(
       const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
       const Teuchos::RCP<ParamLib>& paramLib_,
       const Teuchos::RCP<const Teuchos::Comm<int> >& commT_) :
  problemParams(Teuchos::sublist(topLevelParams, "Problem", true)),
  discretizationParams(Teuchos::sublist(topLevelParams, "Discretization")),
  paramLib(paramLib_),
  commT(commT_)
{
}

namespace {
// In "Mechanics 3D", extract "Mechanics".
inline std::string getName (const std::string& method) {
  if (method.size() < 3) return method;
  return method.substr(0, method.size() - 3);
}
// In "Mechanics 3D", extract 3.
inline int getNumDim (const std::string& method) {
  if (method.size() < 3) return -1;
  return static_cast<int>(method[method.size() - 2] - '0');
}
} // namespace

Teuchos::RCP<Albany::AbstractProblem>
Albany::ProblemFactory::create()
{
  Teuchos::RCP<Albany::AbstractProblem> strategy;
  using Teuchos::rcp;

  std::string& method = problemParams->get("Name", "Heat 1D");

  if (method == "Heat 1D") {
    strategy = rcp(new Albany::HeatProblem(problemParams, paramLib, 1, commT));
  }
  else if (method == "Heat 2D") {
    strategy = rcp(new Albany::HeatProblem(problemParams, paramLib, 2, commT));
  }
  else if (method == "Heat 3D") {
    strategy = rcp(new Albany::HeatProblem(problemParams, paramLib, 3, commT));
  }
  else if (method == "Thermal 1D") {
    strategy = rcp(new Albany::ThermalProblem(problemParams, paramLib, 1, commT));
  }
  else if (method == "Thermal 2D") {
    strategy = rcp(new Albany::ThermalProblem(problemParams, paramLib, 2, commT));
  }
  else if (method == "Thermal 3D") {
    strategy = rcp(new Albany::ThermalProblem(problemParams, paramLib, 3, commT));
  }
  else if (method == "Populate Mesh") {
    strategy = rcp(new Albany::PopulateMesh(problemParams, discretizationParams, paramLib));
  }
  else if (method == "Side Laplacian 3D") {
    strategy = rcp(new Albany::SideLaplacian(problemParams, paramLib, 1));
  }
#ifdef ALBANY_DEMO_PDES
  else if (method == "CahnHill 2D") {
    strategy = rcp(new Albany::CahnHillProblem(problemParams, paramLib, 2, commT));
  }
  else if (method == "ODE") {
    strategy = rcp(new Albany::ODEProblem(problemParams, paramLib, 0));
  }
  else if (method == "Helmholtz 2D") {
    strategy = rcp(new Albany::Helmholtz2DProblem(problemParams, paramLib));
  }
  else if (method == "NavierStokes 1D") {
    strategy = rcp(new Albany::NavierStokes(problemParams, paramLib, 1));
  }
  else if (method == "NavierStokes 2D") {
    strategy = rcp(new Albany::NavierStokes(problemParams, paramLib, 2));
  }
  else if (method == "NavierStokes 3D") {
    strategy = rcp(new Albany::NavierStokes(problemParams, paramLib, 3));
  }
  else if (method == "LinComprNS 1D") {
    strategy = rcp(new Albany::LinComprNSProblem(problemParams, paramLib, 1));
  }
  else if (method == "AdvDiff 1D") {
    strategy = rcp(new Albany::AdvDiffProblem(problemParams, paramLib, 1));
  }
  else if (method == "AdvDiff 2D") {
    strategy = rcp(new Albany::AdvDiffProblem(problemParams, paramLib, 2));
  }
  else if ((method == "Reaction-Diffusion System 3D") || (method == "Reaction-Diffusion System")) {
    strategy = rcp(new Albany::ReactDiffSystem(problemParams, paramLib, 3));
  }
  else if (method == "LinComprNS 2D") {
    strategy = rcp(new Albany::LinComprNSProblem(problemParams, paramLib, 2));
  }
  else if (method == "LinComprNS 3D") {
    strategy = rcp(new Albany::LinComprNSProblem(problemParams, paramLib, 3));
  }
  else if (method == "ComprNS 2D") {
    strategy = rcp(new Albany::ComprNSProblem(problemParams, paramLib, 2));
  }
  else if (method == "ComprNS 3D") {
    strategy = rcp(new Albany::ComprNSProblem(problemParams, paramLib, 3));
  }
  else if (method == "PNP 1D") {
    strategy = rcp(new Albany::PNPProblem(problemParams, paramLib, 1));
  }
  else if (method == "PNP 2D") {
    strategy = rcp(new Albany::PNPProblem(problemParams, paramLib, 2));
  }
  else if (method == "PNP 3D") {
    strategy = rcp(new Albany::PNPProblem(problemParams, paramLib, 3));
  }
  else if (method == "ThermoElectrostatics 1D") {
    strategy = rcp(new Albany::ThermoElectrostaticsProblem(problemParams, paramLib, 1));
  }
  else if (method == "ThermoElectrostatics 2D") {
    strategy = rcp(new Albany::ThermoElectrostaticsProblem(problemParams, paramLib, 2));
  }
  else if (method == "ThermoElectrostatics 3D") {
    strategy = rcp(new Albany::ThermoElectrostaticsProblem(problemParams, paramLib, 3));
  }
#endif
#ifdef ALBANY_LANDICE
  else if (LandIce::ProblemFactory::hasProblem(method)) {
    LandIce::ProblemFactory felix_factory(problemParams,discretizationParams,paramLib);
    strategy = felix_factory.create();
  }
#endif
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                       std::endl <<
                       "Error!  Unknown problem " << method <<
                       "!" << std::endl << "Supplied parameter list is " <<
                       std::endl << *problemParams);
  }

  return strategy;
}

void Albany::ProblemFactory::setReferenceConfigurationManager(
  const Teuchos::RCP<AAdapt::rc::Manager>& rc_mgr_)
{
  rc_mgr = rc_mgr_;
}
