//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LaplaceBeltramiProblem.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"


Albany::LaplaceBeltramiProblem::
LaplaceBeltramiProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                       const Teuchos::RCP<ParamLib>& paramLib_,
                       const int numDim_,
                       const Teuchos::RCP<const Epetra_Comm>& comm_) :
  Albany::AbstractProblem(params_, paramLib_),
  numDim(numDim_),
  comm(comm_) {

  std::string& method = params_->get("Method", "Laplace");

  if(method == "LaplaceBeltrami" ||
         method == "DispLaplaceBeltrami") // Two DOF vectors per node - solution and target solution

     this->setNumEquations(2 * numDim_);

  else

     this->setNumEquations(numDim_);

  // Ask the discretization to initialize the problem by copying the mesh coordinates into the initial guess
  //this->requirements.push_back("Initial Guess Coords");

}

Albany::LaplaceBeltramiProblem::
~LaplaceBeltramiProblem() {
}

void
Albany::LaplaceBeltramiProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr) {

  /* Construct All Phalanx Evaluators */
  int physSets = meshSpecs.size();
  std::cout << "LaplaceBeltrami Problem Num MeshSpecs: " << physSets << std::endl;
  fm.resize(physSets);

  for(int ps = 0; ps < physSets; ps++) {
    fm[ps]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM,
                    Teuchos::null);
  }



  if(meshSpecs[0]->nsNames.size() > 0) // Build a nodeset evaluator if nodesets are present

    constructDirichletEvaluators(meshSpecs[0]->nsNames);


}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::LaplaceBeltramiProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList) {

  ConstructEvaluatorsOp<LaplaceBeltramiProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

// Dirichlet BCs
void
Albany::LaplaceBeltramiProblem::constructDirichletEvaluators(const std::vector<std::string>& nodeSetIDs) {
  // Construct BC evaluators for all node sets and names
  std::vector<std::string> bcNames(numDim);
//  bcNames[0] = "Identity";
  bcNames[0] = "X";
  if(numDim > 1)
    bcNames[1] = "Y";
  if(numDim > 2)
    bcNames[2] = "Z";

  Albany::BCUtils<Albany::DirichletTraits> bcUtils;
  dfm = bcUtils.constructBCEvaluators(nodeSetIDs, bcNames,
                                      this->params, this->paramLib, numDim);
}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::LaplaceBeltramiProblem::getValidProblemParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidLaplaceBeltramiProblemParams");

  validPL->set<std::string>("Method", "", "Smoothing method to use");

  return validPL;
}

