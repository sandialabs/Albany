//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Albany_ProblemFactory.hpp"

// Always enable HeatProblem
#include "Albany_HeatProblem.hpp"

#ifdef ALBANY_DEMO_PDES
#include "Albany_CahnHillProblem.hpp"
#include "Albany_Helmholtz2DProblem.hpp"
#include "Albany_NavierStokes.hpp"
#include "Albany_GPAMProblem.hpp"
#include "Albany_LinComprNSProblem.hpp"
#include "Albany_AdvDiffProblem.hpp"
#include "Albany_ComprNSProblem.hpp"
#include "Albany_ODEProblem.hpp"
#include "Albany_PNPProblem.hpp"
#endif

#ifdef ALBANY_QCAD
#include "QCAD_PoissonProblem.hpp"
#include "QCAD_SchrodingerProblem.hpp"
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
#include "LCM/problems/PoroElasticityProblem.hpp"
#include "LCM/problems/UnSatPoroElasticityProblem.hpp"
#include "LCM/problems/TLPoroPlasticityProblem.hpp"
#include "LCM/problems/ThermoPoroPlasticityProblem.hpp"
#include "LCM/problems/GradientDamageProblem.hpp"
#include "LCM/problems/ThermoMechanicalProblem.hpp"
#include "LCM/problems/ProjectionProblem.hpp"
#include "LCM/problems/ConstitutiveDriverProblem.hpp"
#include "LCM/problems/HMCProblem.hpp"
#include "LCM/problems/ElectroMechanicsProblem.hpp"
#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
#include "LCM/problems/PeridigmProblem.hpp"
#endif
#endif
#if defined(ALBANY_LAME) || defined(ALBANY_LAMENT)
#include "LCM/problems/lame/LameProblem.hpp"
#endif
#endif

#ifdef ALBANY_HYDRIDE
#include "Hydride/problems/HydrideProblem.hpp"
#include "Hydride/problems/HydMorphProblem.hpp"
#include "Hydride/problems/MesoScaleLinkProblem.hpp"
#include "Hydride/problems/LaplaceBeltramiProblem.hpp"
#endif

#ifdef ALBANY_AMP
#include "AMP/problems/PhaseProblem.hpp"
#endif

#ifdef ALBANY_GOAL
#include "GOAL/problems/GOAL_MechanicsProblem.hpp"
#endif

#ifdef ALBANY_FELIX
#include "FELIX/problems/FELIX_Stokes.hpp"
#include "FELIX/problems/FELIX_StokesFO.hpp"
#include "FELIX/problems/FELIX_StokesL1L2.hpp"
#ifdef ALBANY_EPETRA
#include "FELIX/problems/FELIX_Hydrology.hpp"
#include "FELIX/problems/FELIX_Elliptic2D.hpp"
#include "FELIX/problems/FELIX_StokesFOThickness.hpp"
#endif
#endif

#ifdef ALBANY_AERAS
#include "Aeras/problems/Aeras_ShallowWaterProblem.hpp"
#include "Aeras/problems/Aeras_XZScalarAdvectionProblem.hpp"
#include "Aeras/problems/Aeras_XScalarAdvectionProblem.hpp"
#include "Aeras/problems/Aeras_XZHydrostaticProblem.hpp"
#include "Aeras/problems/Aeras_HydrostaticProblem.hpp"
#endif

Albany::ProblemFactory::ProblemFactory(
       const Teuchos::RCP<Teuchos::ParameterList>& problemParams_,
       const Teuchos::RCP<ParamLib>& paramLib_,
       const Teuchos::RCP<const Teuchos::Comm<int> >& commT_) :
  problemParams(problemParams_),
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
  else if (method == "GPAM 1D") {
    strategy = rcp(new Albany::GPAMProblem(problemParams, paramLib, 1));
  }
  else if (method == "GPAM 2D") {
    strategy = rcp(new Albany::GPAMProblem(problemParams, paramLib, 2));
  }
  else if (method == "GPAM 3D") {
    strategy = rcp(new Albany::GPAMProblem(problemParams, paramLib, 3));
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
#endif
#ifdef ALBANY_QCAD
  else if (method == "Poisson 1D") {
    strategy = rcp(new QCAD::PoissonProblem(problemParams, paramLib, 1, commT));
  }
  else if (method == "Poisson 2D") {
    strategy = rcp(new QCAD::PoissonProblem(problemParams, paramLib, 2, commT));
  }
  else if (method == "Poisson 3D") {
    strategy = rcp(new QCAD::PoissonProblem(problemParams, paramLib, 3, commT));
  }
  else if (method == "Schrodinger 1D") {
    strategy = rcp(new QCAD::SchrodingerProblem(problemParams, paramLib, 1, commT));
  }
  else if (method == "Schrodinger 2D") {
    strategy = rcp(new QCAD::SchrodingerProblem(problemParams, paramLib, 2, commT));
  }
  else if (method == "Schrodinger 3D") {
    strategy = rcp(new QCAD::SchrodingerProblem(problemParams, paramLib, 3, commT));
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
  else if (method == "PoroElasticity 1D") {
    strategy = rcp(new Albany::PoroElasticityProblem(problemParams, paramLib, 1));
  }
  else if (method == "PoroElasticity 2D") {
    strategy = rcp(new Albany::PoroElasticityProblem(problemParams, paramLib, 2));
  }
  else if (method == "PoroElasticity 3D") {
    strategy = rcp(new Albany::PoroElasticityProblem(problemParams, paramLib, 3));
  }
  else if (method == "UnSaturated PoroElasticity 1D") {
    strategy = rcp(new Albany::UnSatPoroElasticityProblem(problemParams, paramLib, 1));
  }
  else if (method == "UnSaturated PoroElasticity 2D") {
    strategy = rcp(new Albany::UnSatPoroElasticityProblem(problemParams, paramLib, 2));
  }
  else if (method == "UnSaturated PoroElasticity 3D") {
    strategy = rcp(new Albany::UnSatPoroElasticityProblem(problemParams, paramLib, 3));
  }
  else if (method == "Total Lagrangian PoroPlasticity 1D") {
    strategy = rcp(new Albany::TLPoroPlasticityProblem(problemParams, paramLib, 1));
  }
  else if (method == "Total Lagrangian PoroPlasticity 2D") {
    strategy = rcp(new Albany::TLPoroPlasticityProblem(problemParams, paramLib, 2));
  }
  else if (method == "Total Lagrangian PoroPlasticity 3D") {
    strategy = rcp(new Albany::TLPoroPlasticityProblem(problemParams, paramLib, 3));
  }
  else if (method == "Total Lagrangian ThermoPoroPlasticity 1D") {
    strategy = rcp(new Albany::ThermoPoroPlasticityProblem(problemParams, paramLib, 1));
  }
  else if (method == "Total Lagrangian ThermoPoroPlasticity 2D") {
    strategy = rcp(new Albany::ThermoPoroPlasticityProblem(problemParams, paramLib, 2));
  }
  else if (method == "Total Lagrangian ThermoPoroPlasticity 3D") {
    strategy =   rcp(new Albany::ThermoPoroPlasticityProblem(problemParams, paramLib, 3));
  }
  else if (method == "Total Lagrangian Plasticity with Projection 1D") {
    strategy = rcp(new Albany::ProjectionProblem(problemParams, paramLib, 1));
  }
  else if (method == "Total Lagrangian Plasticity with Projection 2D") {
    strategy = rcp(new Albany::ProjectionProblem(problemParams, paramLib, 2));
  }
  else if (method == "Total Lagrangian Plasticity with Projection 3D") {
    strategy =   rcp(new Albany::ProjectionProblem(problemParams, paramLib, 3));
  }
  else if (method == "GradientDamage") {
    strategy = rcp(new Albany::GradientDamageProblem(problemParams, paramLib, 3));
  }
  else if (method == "ThermoMechanical") {
    strategy = rcp(new Albany::ThermoMechanicalProblem(problemParams, paramLib, 3));
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
#ifdef ALBANY_AMP
  else if (method == "Phase 1D") {
    strategy = rcp(new Albany::PhaseProblem(problemParams, paramLib, 1, commT));
  }
  else if (method == "Phase 2D") {
    strategy = rcp(new Albany::PhaseProblem(problemParams, paramLib, 2, commT));
  }
  else if (method == "Phase 3D") {
    strategy = rcp(new Albany::PhaseProblem(problemParams, paramLib, 3, commT));
  }
#endif
#ifdef ALBANY_GOAL
  else if (method == "GOAL Mechanics 3D") {
    strategy = rcp(new Albany::GOALMechanicsProblem(problemParams, paramLib, 3, commT));
  }
#endif
#ifdef ALBANY_HYDRIDE
  else if (method == "Hydride 2D") {
    strategy = rcp(new Albany::HydrideProblem(problemParams, paramLib, 2, commT));
  }
  else if (method == "HydMorph 2D") {
    strategy = rcp(new Albany::HydMorphProblem(problemParams, paramLib, 2, commT));
  }
  else if (method == "MesoScaleLink 1D") {
    strategy = rcp(new Albany::MesoScaleLinkProblem(problemParams, paramLib, 1, commT));
  }
  else if (method == "MesoScaleLink 2D") {
    strategy = rcp(new Albany::MesoScaleLinkProblem(problemParams, paramLib, 2, commT));
  }
  else if (method == "MesoScaleLink 3D") {
    strategy = rcp(new Albany::MesoScaleLinkProblem(problemParams, paramLib, 3, commT));
  }
  else if (method == "LaplaceBeltrami 2D") {
    strategy = rcp(new Albany::LaplaceBeltramiProblem(problemParams, paramLib, 2, commT));
  }
  else if (method == "LaplaceBeltrami 3D") {
    strategy = rcp(new Albany::LaplaceBeltramiProblem(problemParams, paramLib, 3, commT));
  }
#endif
#ifdef ALBANY_FELIX
  else if (method == "FELIX Stokes" || method == "FELIX Stokes 3D" ) {
    strategy = rcp(new FELIX::Stokes(problemParams, paramLib, 3));
  }
  else if (method == "FELIX Stokes 2D" ) {
    strategy = rcp(new FELIX::Stokes(problemParams, paramLib, 2));
  }
  else if (method == "FELIX Stokes First Order 2D" || method == "FELIX Stokes FO 2D" ||
           method == "FELIX Stokes First Order 2D XZ" || method == "FELIX Stokes FO 2D XZ") {
    strategy = rcp(new FELIX::StokesFO(problemParams, paramLib, 2));
  }
  else if (method == "FELIX Stokes First Order 3D" || method == "FELIX Stokes FO 3D" ) {
    strategy = rcp(new FELIX::StokesFO(problemParams, paramLib, 3));
  }
  else if (method == "FELIX Coupled FO H 3D" ) {
#ifdef ALBANY_EPETRA
      strategy = rcp(new FELIX::StokesFOThickness(problemParams, paramLib, 3));
#else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, " **** FELIX Coupled FO H requires Epetra, recompile with -DENABLE_ALBANY_EPETRA_EXE ****\n");
#endif
    }
  else if (method == "FELIX Stokes L1L2 2D") {
    strategy = rcp(new FELIX::StokesL1L2(problemParams, paramLib, 2));
  }
  else if (method == "FELIX Hydrology 2D") {
    strategy = rcp(new FELIX::Hydrology(problemParams, paramLib, 2));
  }
  else if (method == "FELIX Elliptic 2D") {
    strategy = rcp(new FELIX::Elliptic2D(problemParams, paramLib, 1));
  }
#endif
#ifdef ALBANY_AERAS
  else if (method == "Aeras Shallow Water" ) {
    strategy = rcp(new Aeras::ShallowWaterProblem(problemParams, paramLib, 2));
  }
  else if (method == "Aeras Shallow Water 3D" ) {
    strategy = rcp(new Aeras::ShallowWaterProblem(problemParams, paramLib, 3));
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
  else if (method == "Peridigm Code Coupling" ) {
#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
    strategy = rcp(new Albany::PeridigmProblem(problemParams, paramLib, 3, commT));
#else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, " **** Peridigm code coupling requires epetra and Peridigm, recompile with -DENABLE_ALBANY_EPETRA_EXE and -DENABLE_PERIDIGM ****\n");
#endif
#else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, " **** Peridigm code coupling not enabled, recompile with -DENABLE_PERIDIGM ****\n");
#endif
  }
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
