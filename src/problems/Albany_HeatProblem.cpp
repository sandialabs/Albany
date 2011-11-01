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


#include "Albany_HeatProblem.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_InitialCondition.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"


Albany::HeatProblem::
HeatProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_),
  haveSource(false),
  haveAbsorption(false),
  numDim(numDim_)
{
  this->setNumEquations(1);

  if (numDim==1) periodic = params->get("Periodic BC", false);
  else           periodic = false;
  if (periodic) *out <<" Periodic Boundary Conditions being used." <<std::endl;

  haveSource =  params->isSublist("Source Functions");
  haveAbsorption =  params->isSublist("Absorption");

}

Albany::HeatProblem::
~HeatProblem()
{
}

void
Albany::HeatProblem::
buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
    Albany::StateManager& stateMgr,
    Teuchos::ArrayRCP< Teuchos::RCP<Albany::AbstractResponseFunction> >& responses)
{
  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1); rfm.resize(1);
  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  rfm[0] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);

  constructResidEvaluators<PHAL::AlbanyTraits::Residual  >(*fm[0], *meshSpecs[0], stateMgr);
  constructResidEvaluators<PHAL::AlbanyTraits::Jacobian  >(*fm[0], *meshSpecs[0], stateMgr);
  constructResidEvaluators<PHAL::AlbanyTraits::Tangent   >(*fm[0], *meshSpecs[0], stateMgr);
  constructResidEvaluators<PHAL::AlbanyTraits::SGResidual>(*fm[0], *meshSpecs[0], stateMgr);
  constructResidEvaluators<PHAL::AlbanyTraits::SGJacobian>(*fm[0], *meshSpecs[0], stateMgr);
  constructResidEvaluators<PHAL::AlbanyTraits::SGTangent >(*fm[0], *meshSpecs[0], stateMgr);
  constructResidEvaluators<PHAL::AlbanyTraits::MPResidual>(*fm[0], *meshSpecs[0], stateMgr);
  constructResidEvaluators<PHAL::AlbanyTraits::MPJacobian>(*fm[0], *meshSpecs[0], stateMgr);
  constructResidEvaluators<PHAL::AlbanyTraits::MPTangent >(*fm[0], *meshSpecs[0], stateMgr);
  constructResponseEvaluators<PHAL::AlbanyTraits::Residual  >(*rfm[0], *meshSpecs[0], stateMgr, responses);
  constructResponseEvaluators<PHAL::AlbanyTraits::Jacobian  >(*rfm[0], *meshSpecs[0], stateMgr);
  constructResponseEvaluators<PHAL::AlbanyTraits::Tangent   >(*rfm[0], *meshSpecs[0], stateMgr);
  constructResponseEvaluators<PHAL::AlbanyTraits::SGResidual>(*rfm[0], *meshSpecs[0], stateMgr);
  constructResponseEvaluators<PHAL::AlbanyTraits::SGJacobian>(*rfm[0], *meshSpecs[0], stateMgr);
  constructResponseEvaluators<PHAL::AlbanyTraits::SGTangent >(*rfm[0], *meshSpecs[0], stateMgr);
  constructResponseEvaluators<PHAL::AlbanyTraits::MPResidual>(*rfm[0], *meshSpecs[0], stateMgr);
  constructResponseEvaluators<PHAL::AlbanyTraits::MPJacobian>(*rfm[0], *meshSpecs[0], stateMgr);
  constructResponseEvaluators<PHAL::AlbanyTraits::MPTangent >(*rfm[0], *meshSpecs[0], stateMgr);

  constructDirichletEvaluators(*meshSpecs[0]);

}

void
Albany::HeatProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   vector<string> dirichletNames(neq);
   dirichletNames[0] = "T";
   Albany::DirichletUtils dirUtils;
   dfm = dirUtils.constructDirichletEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::HeatProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidHeatProblemParams");

  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->set("Convection Velocity", "{0,0,0}", "");
  validPL->set<bool>("Have Rho Cp", false, "Flag to indicate if rhoCp is used");

  return validPL;
}

