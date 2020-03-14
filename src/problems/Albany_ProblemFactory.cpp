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

#ifdef ALBANY_ATO
#include "ATO/problems/LinearElasticityProblem.hpp"
#include "ATO/problems/LinearElasticityModalProblem.hpp"
#include "ATO/problems/PoissonsEquation.hpp"
#endif

#if defined(ALBANY_LCM)
#include "LCM/problems/MechanicsProblem.hpp"
#include "LCM/problems/ElasticityProblem.hpp"
#include "LCM/problems/ThermoElasticityProblem.hpp"
#include "LCM/problems/ConstitutiveDriverProblem.hpp"
#include "LCM/problems/HMCProblem.hpp"
#include "LCM/problems/ElectroMechanicsProblem.hpp"
#if defined(ALBANY_LAME) || defined(ALBANY_LAMENT)
#include "LCM/problems/lame/LameProblem.hpp"
#endif
#endif

#ifdef ALBANY_AERAS
#include "Aeras/problems/Aeras_ShallowWaterProblem.hpp"
#include "Aeras/problems/Aeras_ShallowWaterProblemNoAD.hpp"
#include "Aeras/problems/Aeras_XZScalarAdvectionProblem.hpp"
#include "Aeras/problems/Aeras_XScalarAdvectionProblem.hpp"
#include "Aeras/problems/Aeras_XZHydrostaticProblem.hpp"
#include "Aeras/problems/Aeras_HydrostaticProblem.hpp"
#endif

#ifdef ALBANY_LANDICE
#include "LandIce/problems/LandIce_ProblemFactory.hpp"
#endif

#ifdef ALBANY_TSUNAMI
#include "Tsunami/problems/Tsunami_NavierStokes.hpp"
#include "Tsunami/problems/Tsunami_Boussinesq.hpp"
#endif

#ifdef ALBANY_TDM
#include "TDM/problems/TDManufacturing.hpp"
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
#if defined(ALBANY_LCM)
  else if (method == "LAME" || method == "Lame" || method == "lame") {
#if defined(ALBANY_LAME) || defined(ALBANY_LAMENT)
    strategy = rcp(new Albany::LameProblem(problemParams, paramLib, 3, commT));
#else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, " **** LAME materials not enabled, recompile with -DENABLE_LAME or -DENABLE_LAMENT ****\n");
#endif
  }
  else if (getName(method) == "Mechanics") {
    strategy = rcp(new Albany::MechanicsProblem(problemParams, paramLib, getNumDim(method), rc_mgr, commT));
  }
  else if (getName(method) == "Elasticity") {
    strategy = rcp(new Albany::ElasticityProblem(problemParams, paramLib, getNumDim(method), rc_mgr));
  }
  else if (method == "Constitutive Model Driver") {
    strategy = rcp(new Albany::ConstitutiveDriverProblem(problemParams, paramLib, 3, commT));
  }
  else if (method == "ThermoElasticity 1D") {
    strategy = rcp(new Albany::ThermoElasticityProblem(problemParams, paramLib, 1));
  }
  else if (method == "ThermoElasticity 2D") {
    strategy = rcp(new Albany::ThermoElasticityProblem(problemParams, paramLib, 2));
  }
  else if (method == "ThermoElasticity 3D") {
    strategy = rcp(new Albany::ThermoElasticityProblem(problemParams, paramLib, 3));
  }
  else if (method == "HMC 1D") {
    strategy = rcp(new Albany::HMCProblem(problemParams, paramLib, 1, commT));
  }
  else if (method == "HMC 2D") {
    strategy = rcp(new Albany::HMCProblem(problemParams, paramLib, 2, commT));
  }
  else if (method == "HMC 3D") {
    strategy = rcp(new Albany::HMCProblem(problemParams, paramLib, 3, commT));
  }
  else if (method == "Electromechanics 1D") {
    strategy = rcp(new Albany::ElectroMechanicsProblem(problemParams, paramLib, 1, commT));
  }
  else if (method == "Electromechanics 2D") {
    strategy = rcp(new Albany::ElectroMechanicsProblem(problemParams, paramLib, 2, commT));
  }
  else if (method == "Electromechanics 3D") {
    strategy = rcp(new Albany::ElectroMechanicsProblem(problemParams, paramLib, 3, commT));
  }
#endif
#ifdef ALBANY_ATO
  else if (method == "LinearElasticity 1D") {
    strategy = rcp(new Albany::LinearElasticityProblem(problemParams, paramLib, 1));
  }
  else if (method == "LinearElasticity 2D") {
    strategy = rcp(new Albany::LinearElasticityProblem(problemParams, paramLib, 2));
  }
  else if (method == "LinearElasticity 3D") {
    strategy = rcp(new Albany::LinearElasticityProblem(problemParams, paramLib, 3));
  }
  else if (method == "Poissons Equation 1D") {
    strategy = rcp(new Albany::PoissonsEquationProblem(problemParams, paramLib, 1));
  }
  else if (method == "Poissons Equation 2D") {
    strategy = rcp(new Albany::PoissonsEquationProblem(problemParams, paramLib, 2));
  }
  else if (method == "Poissons Equation 3D") {
    strategy = rcp(new Albany::PoissonsEquationProblem(problemParams, paramLib, 3));
  }
  else if (method == "LinearElasticityModal 1D") {
    strategy = rcp(new Albany::LinearElasticityModalProblem(problemParams, paramLib, 1));
  }
  else if (method == "LinearElasticityModal 2D") {
    strategy = rcp(new Albany::LinearElasticityModalProblem(problemParams, paramLib, 2));
  }
  else if (method == "LinearElasticityModal 3D") {
    strategy = rcp(new Albany::LinearElasticityModalProblem(problemParams, paramLib, 3));
  }
#endif
#ifdef ALBANY_TDM
  else if (method == "TDManufacturing 1D") {
    strategy = rcp(new Albany::TDManufacturing(problemParams, paramLib, 1, commT));
  }
  else if (method == "TDManufacturing 2D") {
    strategy = rcp(new Albany::TDManufacturing(problemParams, paramLib, 2, commT));
  }
  else if (method == "TDManufacturing 3D") {
    strategy = rcp(new Albany::TDManufacturing(problemParams, paramLib, 3, commT));
  }
#endif
#ifdef ALBANY_LANDICE
  else if (LandIce::ProblemFactory::hasProblem(method)) {
    LandIce::ProblemFactory felix_factory(problemParams,discretizationParams,paramLib);
    strategy = felix_factory.create();
  }
#endif
#ifdef ALBANY_AERAS
  else if (method == "Aeras Shallow Water" ) {
    strategy = rcp(new Aeras::ShallowWaterProblem(problemParams, paramLib, 2));
  }
  else if (method == "Aeras Shallow Water 3D" ) {
    strategy = rcp(new Aeras::ShallowWaterProblem(problemParams, paramLib, 3));
  }
  else if (method == "Aeras Shallow Water No AD 3D" ) {
    strategy = rcp(new Aeras::ShallowWaterProblemNoAD(problemParams, paramLib, 3));
  }
  else if (method == "Aeras XZ Scalar Advection" ) {
    strategy = rcp(new Aeras::XZScalarAdvectionProblem(problemParams, paramLib, 2));
  }
  else if (method == "Aeras X Scalar Advection" ) {
    strategy = rcp(new Aeras::XScalarAdvectionProblem(problemParams, paramLib, 1));
  }
  else if (method == "Aeras XZ Hydrostatic" ) {
    strategy = rcp(new Aeras::XZHydrostaticProblem(problemParams, paramLib, 1));
  }
  else if (method == "Aeras Hydrostatic" ) {
    strategy = rcp(new Aeras::HydrostaticProblem(problemParams, paramLib, 2));
  }
#endif
#ifdef ALBANY_TSUNAMI
  else if (method == "Tsunami Steady Stokes 2D" ) {
    strategy = rcp(new Tsunami::NavierStokes(problemParams, paramLib, 2, false, false));
  }
  else if (method == "Tsunami Navier Stokes 2D" ) {
    strategy = rcp(new Tsunami::NavierStokes(problemParams, paramLib, 2));
  }
  else if (method == "Tsunami Navier Stokes 3D" ) {
    strategy = rcp(new Tsunami::NavierStokes(problemParams, paramLib, 3));
  }
  else if (method == "Tsunami Boussinesq 1D" ) {
    strategy = rcp(new Tsunami::Boussinesq(problemParams, paramLib, 1));
  }
  else if (method == "Tsunami Boussinesq 2D" ) {
    strategy = rcp(new Tsunami::Boussinesq(problemParams, paramLib, 2));
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
