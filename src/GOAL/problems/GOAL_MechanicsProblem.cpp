//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ProblemUtils.hpp"
#include "Albany_Utils.hpp"
#include "LCM_Utils.h"
#include "PHAL_AlbanyTraits.hpp"
#include "GOAL_MechanicsProblem.hpp"

namespace Albany {

/*****************************************************************************/
GOALMechanicsProblem::GOALMechanicsProblem(
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    const Teuchos::RCP<ParamLib>& paramLib,
    const int numDim,
    Teuchos::RCP<const Teuchos::Comm<int> >& commT) :
  Albany::AbstractProblem(params, paramLib),
  numDims(numDim)
{
  // compute number of equations
  int numEq = 0;
  numEq += numDims;
  this->setNumEquations(numEq);

  // create the material database
  materialDB = LCM::createMaterialDatabase(params, commT);

  // print a summary of the problem
  *out << "GOAL Mechanics Problem" << std::endl;
  *out << "Number of spatial dimensions: " << numDims << std::endl;

  // fill in the dof names
  offsets["X"] = 0;
  if (numDims > 1)
    offsets["Y"] = 1;
  if (numDims > 2)
    offsets["Z"] = 2;
}

/*****************************************************************************/
GOALMechanicsProblem::~GOALMechanicsProblem()
{
}

/*****************************************************************************/
int GOALMechanicsProblem::getOffset(std::string const& var)
{
  assert(offsets.count(var) == 1);
  return offsets[var];
}

/*****************************************************************************/
void GOALMechanicsProblem::buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct> > meshSpecs,
    StateManager& stateMgr)
{
  *out << "Building primal problem pde instantiations\n";

  // get the number of physics sets
  int physSets = meshSpecs.size();
  fm.resize(physSets);

  // build evaluators for each physics set
  for (int ps=0; ps < physSets; ++ps)
  {
    fm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(
        *fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM, Teuchos::null);
  }

  // construct dirichlet bc evaluators
  constructDirichletEvaluators(*meshSpecs[0], this->params);
}

/*****************************************************************************/
Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> > GOALMechanicsProblem::
buildEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    const MeshSpecsStruct& meshSpecs,
    StateManager& stateMgr,
    FieldManagerChoice fmChoice,
    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // calls constructEvaluators<EvalT> for all EvalT
  ConstructEvaluatorsOp<GOALMechanicsProblem> op(
      *this, fm0, meshSpecs, stateMgr, fmChoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

/*****************************************************************************/
void GOALMechanicsProblem::constructDirichletEvaluators(
    const Albany::MeshSpecsStruct& meshSpecs,
    Teuchos::RCP<Teuchos::ParameterList>& bcs)
{
  dfm = Teuchos::null;
}

/*****************************************************************************/
void GOALMechanicsProblem::constructNeumannEvaluators(
    const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
  nfm = Teuchos::null;
}

/*****************************************************************************/
Teuchos::RCP<const Teuchos::ParameterList> GOALMechanicsProblem::
getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> pl =
      this->getGenericProblemParams("ValidGOALMechanicsProblemParams");
  pl->set<std::string>("MaterialDB Filename", "materials.xml", "");
  pl->sublist("Hierarchic Boundary Conditions", false, "");
  return pl;
}

/*****************************************************************************/
void GOALMechanicsProblem::getAllocatedStates(
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP
      <Intrepid::FieldContainer<RealType> > > > oldSt,
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP
      <Intrepid::FieldContainer<RealType> > > > newSt) const
{
  oldSt = oldState;
  newSt = newState;
}

}
