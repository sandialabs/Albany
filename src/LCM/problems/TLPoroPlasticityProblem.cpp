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

#include "TLPoroPlasticityProblem.hpp"
#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_SolutionMaxValueResponseFunction.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"


Albany::TLPoroPlasticityProblem::
TLPoroPlasticityProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
			const Teuchos::RCP<ParamLib>& paramLib_,
			const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_ + 1), // additional DOF for pore pressure
  haveSource(false),
  numDim(numDim_)
{
 
  std::string& method = params->get("Name", "Total Lagrangian PoroPlasticity ");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

  matModel = params->sublist("Material Model").get("Model Name", "NeoHookean");

// Changing this ifdef changes ordering from  (X,Y,T) to (T,X,Y)
//#define NUMBER_T_FIRST
#ifdef NUMBER_T_FIRST
  T_offset=0;
  X_offset=1;
#else
  X_offset=0;
  T_offset=numDim;
#endif
}

Albany::TLPoroPlasticityProblem::
~TLPoroPlasticityProblem()
{
}

void
Albany::TLPoroPlasticityProblem::
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
Albany::TLPoroPlasticityProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   vector<string> dirichletNames(neq);
   dirichletNames[X_offset] = "X";
   if (numDim>1) dirichletNames[X_offset+1] = "Y";
   if (numDim>2) dirichletNames[X_offset+2] = "Z";
   dirichletNames[T_offset] = "T";
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::TLPoroPlasticityProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidTLPoroPlasticityProblemParams");
  validPL->sublist("Material Model", false, "");
  validPL->sublist("Porosity", false, "");
  validPL->sublist("Biot Coefficient", false, "");
  validPL->sublist("Biot Modulus", false, "");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->sublist("Kozeny-Carman Permeability", false, "");
  validPL->sublist("Elastic Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");
  validPL->sublist("Stabilization Parameter", false, "");
  if (matModel=="J2"){
   validPL->sublist("Hardening Modulus", false, "");
   validPL->sublist("Saturation Modulus", false, "");
   validPL->sublist("Saturation Exponent", false, "");
   validPL->sublist("Yield Strength", false, "");
  }

  return validPL;
}

void
Albany::TLPoroPlasticityProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}

