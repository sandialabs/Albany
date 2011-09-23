/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Albany_ProblemFactory.hpp"
#include "Albany_Helmholtz2DProblem.hpp"
#include "Albany_HeatProblem.hpp"
#include "Albany_NavierStokes.hpp"
#include "Albany_ODEProblem.hpp"
#include "Albany_ThermoElectrostaticsProblem.hpp"
#include "QCAD_PoissonProblem.hpp"
#include "QCAD_SchrodingerProblem.hpp"

#ifdef ALBANY_LCM
#include "LCM/problems/ElasticityProblem.hpp"
#include "LCM/problems/NonlinearElasticityProblem.hpp"
#include "LCM/problems/ThermoElasticityProblem.hpp"
#include "LCM/problems/PoroElasticityProblem.hpp"
#include "LCM/problems/GradientDamageProblem.hpp"
#ifdef ALBANY_LAME
#include "LCM/problems/LameProblem.hpp"
#endif
#endif

Albany::ProblemFactory::ProblemFactory(
       const Teuchos::RCP<Teuchos::ParameterList>& problemParams_,
       const Teuchos::RCP<ParamLib>& paramLib_,
       const Teuchos::RCP<const Epetra_Comm>& comm_) :
  problemParams(problemParams_),
  paramLib(paramLib_),
  comm(comm_)
{
}

Teuchos::RCP<Albany::AbstractProblem>
Albany::ProblemFactory::create()
{
  Teuchos::RCP<Albany::AbstractProblem> strategy;
  using Teuchos::rcp;

  std::string& method = problemParams->get("Name", "Heat 1D");

  if (method == "Heat 1D") {
    strategy = rcp(new Albany::HeatProblem(problemParams, paramLib, 1));
  }
  else if (method == "Heat 2D") {
    strategy = rcp(new Albany::HeatProblem(problemParams, paramLib, 2));
  }
  else if (method == "Heat 3D") {
    strategy = rcp(new Albany::HeatProblem(problemParams, paramLib, 3));
  }
  else if (method == "ODE") {
    strategy = rcp(new Albany::ODEProblem(problemParams, paramLib, 0));
  }
  else if (method == "Helmholtz 2D") {
    strategy = rcp(new Albany::Helmholtz2DProblem(problemParams, paramLib));
  }
  else if (method == "Poisson 1D") {
    strategy = rcp(new QCAD::PoissonProblem(problemParams, paramLib, 1, comm));
  }
  else if (method == "Poisson 2D") {
    strategy = rcp(new QCAD::PoissonProblem(problemParams, paramLib, 2, comm));
  }
  else if (method == "Poisson 3D") {
    strategy = rcp(new QCAD::PoissonProblem(problemParams, paramLib, 3, comm));
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
  else if (method == "Schrodinger 1D") {
    strategy = rcp(new QCAD::SchrodingerProblem(problemParams, paramLib, 1, comm));
  }
  else if (method == "Schrodinger 2D") {
    strategy = rcp(new QCAD::SchrodingerProblem(problemParams, paramLib, 2, comm));
  }
  else if (method == "Schrodinger 3D") {
    strategy = rcp(new QCAD::SchrodingerProblem(problemParams, paramLib, 3, comm));
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
#ifdef ALBANY_LCM
  else if (method == "LAME" || method == "Lame" || method == "lame") {
#ifdef ALBANY_LAME
    strategy = rcp(new Albany::LameProblem(problemParams, paramLib, 3));
#else
    TEST_FOR_EXCEPTION(true, std::runtime_error, " **** LAME materials not enabled, recompile with -DENABLE_LAME ****\n");
#endif
  }
  else if (method == "Elasticity 1D") {
    strategy = rcp(new Albany::ElasticityProblem(problemParams, paramLib, 1));
  }
  else if (method == "Elasticity 2D") {
    strategy = rcp(new Albany::ElasticityProblem(problemParams, paramLib, 2));
  }
  else if (method == "Elasticity 3D") {
    strategy = rcp(new Albany::ElasticityProblem(problemParams, paramLib, 3));
  }
  else if (method == "NonlinearElasticity 1D") {
    strategy = rcp(new Albany::NonlinearElasticityProblem(problemParams, paramLib, 1));
  }
  else if (method == "NonlinearElasticity 2D") {
    strategy = rcp(new Albany::NonlinearElasticityProblem(problemParams, paramLib, 2));
  }
  else if (method == "NonlinearElasticity 3D") {
    strategy = rcp(new Albany::NonlinearElasticityProblem(problemParams, paramLib, 3));
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
  else if (method == "GradientDamage") {
    strategy = rcp(new Albany::GradientDamageProblem(problemParams, paramLib, 3));
  }
#endif
  else {
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                       std::endl << 
                       "Error!  Unknown problem " << method << 
                       "!" << std::endl << "Supplied parameter list is " << 
                       std::endl << *problemParams);
  }

  return strategy;
}
