//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "ElasticityProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"

Albany::ElasticityProblem::
ElasticityProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
		  const Teuchos::RCP<ParamLib>& paramLib_,
		  const int numDim_,
                  const Teuchos::RCP<AAdapt::rc::Manager>& rc_mgr_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_),
  haveSource(false),
  numDim(numDim_),
  use_sdbcs_(false),
  rc_mgr(rc_mgr_)
{
  std::string& method = params->get("Name", "Elasticity ");
  *out << "Problem Name = " << method << std::endl;

  haveSource =  params->isSublist("Source Functions");

  matModel = params->sublist("Material Model").get("Model Name", "LinearElasticity");

  computeError = params->get<bool>("Compute Error", false);

  if (computeError)
    this->setNumEquations(2*neq);

// the following function returns the problem information required for setting the rigid body modes (RBMs) for elasticity problems
//written by IK, Feb. 2012

  int numScalar = 0;
  int nullSpaceDim = 0;
  if (numDim == 1) {nullSpaceDim = 0; }
  else {
    if (numDim == 2) {nullSpaceDim = 3; }
    if (numDim == 3) {nullSpaceDim = 6; }
  }

  if (computeError)
    rigidBodyModes->setParameters(2*numDim, numDim, numScalar, nullSpaceDim);
  else
    rigidBodyModes->setParameters(numDim, numDim, numScalar, nullSpaceDim);

}

Albany::ElasticityProblem::
~ElasticityProblem()
{
}

void
Albany::ElasticityProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>  meshSpecs,
  Albany::StateManager& stateMgr)
{
  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");

  fm.resize(1);

  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, BUILD_RESID_FM,
		  Teuchos::null);

  if(meshSpecs[0]->nsNames.size() > 0) // Build a nodeset evaluator if nodesets are present

    constructDirichletEvaluators(*meshSpecs[0]);

  if(meshSpecs[0]->ssNames.size() > 0) // Build a sideset evaluator if sidesets are present

    constructNeumannEvaluators(meshSpecs[0]);

}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
Albany::ElasticityProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<ElasticityProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

// Dirichlet BCs
void
Albany::ElasticityProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  dirichletNames[0] = "X";
  if (neq>1) dirichletNames[1] = "Y";
  if (neq>2) dirichletNames[2] = "Z";
  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
  use_sdbcs_ = dirUtils.useSDBCs();
  offsets_ = dirUtils.getOffsets();
  nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

// Neumann BCs
void
Albany::ElasticityProblem::constructNeumannEvaluators(
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
   std::vector<std::string> neumannNames(neq + 1);
   Teuchos::Array<Teuchos::Array<int>> offsets;
   offsets.resize(neq + 1);

   neumannNames[0] = "sig_x";
   offsets[0].resize(1);
   offsets[0][0] = 0;
   offsets[neq].resize(neq);
   offsets[neq][0] = 0;

   if (neq>1){
      neumannNames[1] = "sig_y";
      offsets[1].resize(1);
      offsets[1][0] = 1;
      offsets[neq][1] = 1;
   }

   if (neq>2){
     neumannNames[2] = "sig_z";
      offsets[2].resize(1);
      offsets[2][0] = 2;
      offsets[neq][2] = 2;
   }

   neumannNames[neq] = "all";

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both
   std::vector<std::string> condNames(3); //dudx, dudy, dudz, dudn, P
   Teuchos::ArrayRCP<std::string> dof_names(1);
     dof_names[0] = "Displacement";

   // Note that sidesets are only supported for two and 3D currently
   if(numDim == 2)
    condNames[0] = "(t_x, t_y)";
   else if(numDim == 3)
    condNames[0] = "(t_x, t_y, t_z)";
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

   condNames[1] = "dudn";
   condNames[2] = "P";

   nfm.resize(1); // Elasticity problem only has one element block

   nfm[0] = neuUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, true, 0,
                                          condNames, offsets, dl,
                                          this->params, this->paramLib);

}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::ElasticityProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidElasticityProblemParams");

  validPL->sublist("Density", false, "");
  validPL->sublist("Elastic Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");
  validPL->sublist("Material Model", false, "");

  validPL->set<bool>("Compute Error", false, "");

#ifdef ALBANY_ATO
  // Add additional parameters now for Topological Optimization.
  // ... these in an evaluator rather that in getValidProblemParameters()
  // ... as (apparently) they arn't parsable but used later
  validPL->set<bool>("avgJ", false, "");
  validPL->set<bool>("volavgJ", false, "");
  validPL->set<bool>("weighted_Volume_Averaged_J", false, "");
#endif //ALBANY_ATO

  if (matModel == "CapExplicit"|| matModel == "CapImplicit")
  {
	validPL->set<double>("A", false, "");
	validPL->set<double>("B", false, "");
	validPL->set<double>("C", false, "");
	validPL->set<double>("theta", false, "");
	validPL->set<double>("R", false, "");
	validPL->set<double>("kappa0", false, "");
	validPL->set<double>("W", false, "");
	validPL->set<double>("D1", false, "");
	validPL->set<double>("D2", false, "");
	validPL->set<double>("calpha", false, "");
	validPL->set<double>("psi", false, "");
	validPL->set<double>("N", false, "");
	validPL->set<double>("L", false, "");
	validPL->set<double>("phi", false, "");
	validPL->set<double>("Q", false, "");
  }

  if (matModel == "GursonSD")
  {
	validPL->set<double>("f0", false, "");
	validPL->set<double>("Y0", false, "");
	validPL->set<double>("kw", false, "");
	validPL->set<double>("N", false, "");
	validPL->set<double>("q1", false, "");
	validPL->set<double>("q2", false, "");
	validPL->set<double>("q3", false, "");
	validPL->set<double>("eN", false, "");
	validPL->set<double>("sN", false, "");
	validPL->set<double>("fN", false, "");
	validPL->set<double>("fc", false, "");
	validPL->set<double>("ff", false, "");
	validPL->set<double>("flag", false, "");
  }

  return validPL;
}

void
Albany::ElasticityProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}
