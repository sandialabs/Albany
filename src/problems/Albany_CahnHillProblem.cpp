//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_CahnHillProblem.hpp"

#include "Intrepid2_FieldContainer.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"


Albany::CahnHillProblem::
CahnHillProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_,
             Teuchos::RCP<const Teuchos::Comm<int> >& commT_) :
  Albany::AbstractProblem(params_, paramLib_, 2),
  numDim(numDim_),
  haveNoise(false),
  commT(commT_)
{}

Albany::CahnHillProblem::
~CahnHillProblem()
{
}

void
Albany::CahnHillProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");

  fm.resize(1);
  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, BUILD_RESID_FM, 
		  Teuchos::null);

  if(meshSpecs[0]->nsNames.size() > 0) // Build a nodeset evaluator if nodesets are present

    constructDirichletEvaluators(meshSpecs[0]->nsNames);

}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::CahnHillProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<CahnHillProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

// Dirichlet BCs
void
Albany::CahnHillProblem::constructDirichletEvaluators(const std::vector<std::string>& nodeSetIDs)
{
   // Construct BC evaluators for all node sets and names
   std::vector<std::string> bcNames(neq);
   bcNames[0] = "rho";
   Albany::BCUtils<Albany::DirichletTraits> bcUtils;
   dfm = bcUtils.constructBCEvaluators(nodeSetIDs, bcNames,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::CahnHillProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidCahnHillProblemParams");

  Teuchos::Array<int> defaultPeriod;

  validPL->set<double>("b", 0.0, "b value in equation 1.1");
  validPL->set<double>("gamma", 0.0, "gamma value in equation 2.2");
  validPL->set<double>("Langevin Noise SD", 0.0, "Standard deviation of the Langevin noise to apply");
  validPL->set<Teuchos::Array<int> >("Langevin Noise Time Period", defaultPeriod, 
    "Time period to apply Langevin noise");
  validPL->set<bool>("Lump Mass", true, "Lump mass matrix in time derivative term");

  return validPL;
}

