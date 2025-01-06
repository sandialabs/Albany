//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_AdvDiffSystemProblem.hpp"
#include "InitialCondition.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"


Albany::AdvDiffSystemProblem::
AdvDiffSystemProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_),
  numDim(numDim_),
  params(params_),
  use_sdbcs_(false)
{
  if (numDim==1) periodic = params->get("Periodic BC", false);
  else           periodic = false;
  if (periodic) *out <<" Periodic Boundary Conditions being used." <<std::endl;

  this->setNumEquations(1);

}

Albany::AdvDiffSystemProblem::
~AdvDiffSystemProblem()
{
}

void
Albany::AdvDiffSystemProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

 /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");

  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, BUILD_RESID_FM,
		  Teuchos::null);

  if (meshSpecs[0]->nsNames.size() > 0) { // Build a nodeset evaluator if nodesets are present
     constructDirichletEvaluators(meshSpecs[0]->nsNames);
  }


}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
Albany::AdvDiffSystemProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<AdvDiffSystemProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
Albany::AdvDiffSystemProblem::constructDirichletEvaluators(
        const std::vector<std::string>& nodeSetIDs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
   dirichletNames[0] = "rhop";
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(nodeSetIDs, dirichletNames,
                                          this->params, this->paramLib);
   use_sdbcs_ = dirUtils.useSDBCs();
   offsets_ = dirUtils.getOffsets();
   nodeSetIDs_ = dirUtils.getNodeSetIDs();
}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::AdvDiffSystemProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidAdvDiffSystemProblemParams");

  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->sublist("Options", false, "");

  return validPL;
}

