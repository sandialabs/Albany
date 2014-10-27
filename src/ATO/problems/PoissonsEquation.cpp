//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "PoissonsEquation.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "ATO_TopoTools.hpp"

Albany::PoissonsEquationProblem::
PoissonsEquationProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
		        const Teuchos::RCP<ParamLib>& paramLib_,
		        const int numDim_) :
  ATO::OptimizationProblem(params_, paramLib_, /*neq=*/ 1),
  Albany::AbstractProblem(params_, paramLib_, /*neq=*/ 1),
  numDim(numDim_)
{
  std::string& method = params->get("Name", "Poissons Equation ");
  *out << "Problem Name = " << method << std::endl;
}

Albany::PoissonsEquationProblem::
~PoissonsEquationProblem()
{
}

void
Albany::PoissonsEquationProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{

  int physSets = meshSpecs.size();
  *out << "Num MeshSpecs: " << physSets << '\n';
  fm.resize(physSets);
  bool haveSidesets = false;

  *out << "Calling PoissonsEquationProblem::buildEvaluators" << '\n';
  for (int ps = 0; ps < physSets; ++ps) {
    fm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM,
        Teuchos::null);
    if (meshSpecs[ps]->ssNames.size() > 0) haveSidesets = true;
  }
  constructDirichletEvaluators(*meshSpecs[0]);

  if( haveSidesets )
    constructNeumannEvaluators(meshSpecs[0]);

   setupTopOpt(meshSpecs,stateMgr);

}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::PoissonsEquationProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<PoissonsEquationProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

// Dirichlet BCs
void
Albany::PoissonsEquationProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  dirichletNames[0] = "P";
  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
}

// Neumann BCs
void
Albany::PoissonsEquationProblem::constructNeumannEvaluators(
        const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
   // Note: we only enter this function if sidesets are defined in the mesh file
   // i.e. meshSpecs.ssNames.size() > 0

   Albany::BCUtils<Albany::NeumannTraits> neuUtils;

   // Check to make sure that Neumann BCs are given in the input file

   if(!neuUtils.haveBCSpecified(this->params))

      return;

   // Construct BC evaluators for all side sets and names
   // Note that the string index sets up the equation offset, so ordering is important
   std::vector<std::string> neumannNames(neq);
   Teuchos::Array<Teuchos::Array<int> > offsets;
   offsets.resize(neq);

   neumannNames[0] = "P";
   offsets[0].resize(1);
   offsets[0][0] = 0;

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both
   std::vector<std::string> condNames(2); //dudx, dudy, dudz, dudn, P
   Teuchos::ArrayRCP<std::string> dof_names(1);
     dof_names[0] = "Phi";

   // Note that sidesets are only supported for two and 3D currently
   if(numDim == 2)
    condNames[0] = "(dudx, dudy)";
   else if(numDim == 3)
    condNames[0] = "(dudx, dudy, dudz)";
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

   condNames[1] = "dudn";

   nfm = neuUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, true, 0,
                                          condNames, offsets, dl,
                                          this->params, this->paramLib);

}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::PoissonsEquationProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidPoissonsEquationProblemParams");

  validPL->set<double>("Isotropic Modulus", 0.0);

  Teuchos::RCP<ATO::Topology> emptyTopo;
  emptyTopo = Teuchos::null;
  validPL->set<Teuchos::RCP<ATO::Topology> >("Topology", emptyTopo);

  validPL->sublist("Objective Aggregator", false, "");

  return validPL;
}

void
Albany::PoissonsEquationProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}
