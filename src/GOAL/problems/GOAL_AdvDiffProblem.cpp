//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ProblemUtils.hpp"
#include "Albany_Utils.hpp"
#include "LCM_Utils.h"
#include "PHAL_AlbanyTraits.hpp"
#include "GOAL_AdvDiffProblem.hpp"

namespace Albany {

/*****************************************************************************/
GOALAdvDiffProblem::GOALAdvDiffProblem(
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    const Teuchos::RCP<ParamLib>& paramLib,
    const int numDim,
    Teuchos::RCP<const Teuchos::Comm<int> >& commT) :
  Albany::AbstractProblem(params, paramLib),
  numDims(numDim)
{
  // compute number of equations
  this->setNumEquations(1);

  // print a summary of the problem
  *out << "GOAL AdvDiff Problem" << std::endl;
  *out << "Number of spatial dimensions: " << numDims << std::endl;

  // get the diffusivity coefficient
  k = params->get<double>("Diffusivity Coefficient", 1.0);

  // get the advection vector
  Teuchos::Array<double> dummy;
  a = params->get<Teuchos::Array<double> >("Advection Vector", dummy);

  // determine if supg stabilization should be used
  useSUPG = params->get<bool>("SUPG", false);

  // if solving the adjoint problem, should we use an enriched basis?
  if (params->isParameter("Enrich Adjoint"))
    enrichAdjoint = params->get<bool>("Enrich Adjoint", false);

  // if solving the adjoint problem, we need a quantity of interest
  if (params->isSublist("Quantity of Interest")) {
    Teuchos::ParameterList& p = params->sublist("Quantity of Interest", false);
    qoiParams = Teuchos::rcpFromRef(p);
  }

  // fill in the dof names
  offsets["U"] = 0;
}

/*****************************************************************************/
GOALAdvDiffProblem::~GOALAdvDiffProblem()
{
}

/*****************************************************************************/
int GOALAdvDiffProblem::getOffset(std::string const& var)
{
  assert(offsets.count(var) == 1);
  return offsets[var];
}

/*****************************************************************************/
void GOALAdvDiffProblem::buildProblem(
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
}

/*****************************************************************************/
Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> > GOALAdvDiffProblem::
buildEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    const MeshSpecsStruct& meshSpecs,
    StateManager& stateMgr,
    FieldManagerChoice fmChoice,
    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // calls constructEvaluators<EvalT> for all EvalT
  ConstructEvaluatorsOp<GOALAdvDiffProblem> op(
      *this, fm0, meshSpecs, stateMgr, fmChoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

/*****************************************************************************/
Teuchos::RCP<const Teuchos::ParameterList> GOALAdvDiffProblem::
getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> pl =
      this->getGenericProblemParams("ValidGOALAdvDiffProblemParams");
  Teuchos::Array<double> dummy;
  pl->sublist("Hierarchic Boundary Conditions", false, "");
  pl->set<bool>("Enrich Adjoint", false, "should the adjoint solve be enriched");
  pl->sublist("Quantity of Interest", false, "QoI used for adjoint solve");
  pl->set<Teuchos::Array<double> >("Advection Vector", dummy, "");
  pl->set<double>("Diffusivity Coefficient", 1.0, "");
  pl->set<bool>("SUPG", false, "");
  return pl;
}

}
