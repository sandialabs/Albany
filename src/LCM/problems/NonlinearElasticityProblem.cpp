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

#include "NonlinearElasticityProblem.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"


Albany::NonlinearElasticityProblem::
NonlinearElasticityProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
			   const Teuchos::RCP<ParamLib>& paramLib_,
			   const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_),
  haveSource(false),
  numDim(numDim_)
{
 
  std::string& method = params->get("Name", "NonlinearElasticity ");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

  matModel = params->sublist("Material Model").get("Model Name","NeoHookean");
}

Albany::NonlinearElasticityProblem::
~NonlinearElasticityProblem()
{
}

void
Albany::NonlinearElasticityProblem::
buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
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
Albany::NonlinearElasticityProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{

   // Construct Dirichlet evaluators for all nodesets and names
   vector<string> dirichletNames(neq);
   dirichletNames[0] = "X";
   if (neq>1) dirichletNames[1] = "Y";
   if (neq>2) dirichletNames[2] = "Z";
   Albany::DirichletUtils dirUtils;
   dfm = dirUtils.constructDirichletEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::NonlinearElasticityProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidNonlinearElasticityProblemParams");

  validPL->sublist("Elastic Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");
  validPL->sublist("Shear Modulus", false, "");
  validPL->sublist("Material Model", false, "");
  validPL->set<bool>("avgJ", false, "Flag to indicate the J should be averaged");
  validPL->set<bool>("volavgJ", false, "Flag to indicate the J should be volume averaged");
  validPL->set<bool>("Use Composite Tet 10", false, "Flag to use the compostie tet 10 basis in Intrepid");
  if (matModel == "J2")
  {
    validPL->set<bool>("Compute Dislocation Density Tensor", false, "Flag to compute the dislocaiton density tensor (only for 3D)");
    validPL->sublist("Hardening Modulus", false, "");
    validPL->sublist("Yield Strength", false, "");
    validPL->sublist("Saturation Modulus", false, "");
    validPL->sublist("Saturation Exponent", false, "");
  }

  return validPL;
}

void
Albany::NonlinearElasticityProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}

