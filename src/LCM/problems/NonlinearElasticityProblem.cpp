//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "NonlinearElasticityProblem.hpp"
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

//the following function returns the problem information required for setting the rigid body modes (RBMs) for elasticity problems (in src/Albany_SolverFactory.cpp)
//written by IK, Feb. 2012 
void
Albany::NonlinearElasticityProblem::getRBMInfoForML(
   int& numPDEs, int& numElasticityDim, int& numScalar,  int& nullSpaceDim)
{
  numPDEs = numDim;
  numElasticityDim = numDim;
  numScalar = 0;
  if (numDim == 1) {nullSpaceDim = 0; }
  else {
    if (numDim == 2) {nullSpaceDim = 3; }
    if (numDim == 3) {nullSpaceDim = 6; }
  }
}


void
Albany::NonlinearElasticityProblem::
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

    constructDirichletEvaluators(*meshSpecs[0]);

  if(meshSpecs[0]->ssNames.size() > 0) // Build a sideset evaluator if sidesets are present

    constructNeumannEvaluators(meshSpecs[0]);

}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::NonlinearElasticityProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<NonlinearElasticityProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

void
Albany::NonlinearElasticityProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{

  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<string> dirichletNames(neq);
  dirichletNames[0] = "X";
  if (neq>1) dirichletNames[1] = "Y";
  if (neq>2) dirichletNames[2] = "Z";
  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
}

// Neumann BCs
void
Albany::NonlinearElasticityProblem::constructNeumannEvaluators(
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
   std::vector<string> neumannNames(neq + 1);
   Teuchos::Array<Teuchos::Array<int> > offsets;
   offsets.resize(neq + 1);

   neumannNames[0] = "Tx";
   offsets[0].resize(1);
   offsets[0][0] = 0;
   offsets[neq].resize(neq);
   offsets[neq][0] = 0;

   if (neq>1){ 
      neumannNames[1] = "Ty";
      offsets[1].resize(1);
      offsets[1][0] = 1;
      offsets[neq][1] = 1;
   }

   if (neq>2){
     neumannNames[2] = "Tz";
      offsets[2].resize(1);
      offsets[2][0] = 2;
      offsets[neq][2] = 2;
   }

   neumannNames[neq] = "all";

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both

   std::vector<string> condNames(3); //dudx, dudy, dudz, dudn, P
   Teuchos::ArrayRCP<string> dof_names(1);
     dof_names[0] = "Displacement";

   // Note that sidesets are only supported for two and 3D currently
   if(numDim == 2)
    condNames[0] = "(dudx, dudy)";
   else if(numDim == 3)
    condNames[0] = "(dudx, dudy, dudz)";
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
  validPL->set<bool>("weighted_Volume_Averaged_J", false, "Flag to indicate the J should be volume averaged with stabilization");
  validPL->set<bool>("Use Composite Tet 10", false, "Flag to use the compostie tet 10 basis in Intrepid");


  if (matModel == "J2"|| matModel == "J2Fiber" || matModel == "GursonFD" || matModel == "RIHMR")
  {
    validPL->set<bool>("Compute Dislocation Density Tensor", false, "Flag to compute the dislocaiton density tensor (only for 3D)");
    validPL->sublist("Hardening Modulus", false, "");
    validPL->sublist("Yield Strength", false, "");
    validPL->sublist("Saturation Modulus", false, "");
    validPL->sublist("Saturation Exponent", false, "");
  }

  if (matModel == "J2Fiber")
  {
	validPL->set<RealType>("xiinf_J2",false,"");
	validPL->set<RealType>("tau_J2",false,"");
	validPL->set<RealType>("k_f1",false,"");
	validPL->set<RealType>("q_f1",false,"");
	validPL->set<RealType>("vol_f1",false,"");
	validPL->set<RealType>("xiinf_f1",false,"");
	validPL->set<RealType>("tau_f1",false,"");
	validPL->set<RealType>("k_f2",false,"");
	validPL->set<RealType>("q_f2",false,"");
	validPL->set<RealType>("vol_f2",false,"");
	validPL->set<RealType>("xiinf_f2",false,"");
	validPL->set<RealType>("tau_f2",false,"");
	validPL->set<RealType>("X0",false,"");
	validPL->set<RealType>("Y0",false,"");
	validPL->set<RealType>("Z0",false,"");
	validPL->sublist("direction_f1",false,"");
	validPL->sublist("direction_f2",false,"");
	validPL->sublist("Ring Center",false,"");
	validPL->set<bool>("isLocalCoord",false,"");
  }

  if (matModel == "GursonFD")
  {
	validPL->set<RealType>("N",false,"");
	validPL->set<RealType>("eq0",false,"");
	validPL->set<RealType>("f0",false,"");
	validPL->set<RealType>("kw",false,"");
	validPL->set<RealType>("eN",false,"");
	validPL->set<RealType>("sN",false,"");
	validPL->set<RealType>("fN",false,"");
	validPL->set<RealType>("fc",false,"");
	validPL->set<RealType>("ff",false,"");
	validPL->set<RealType>("q1",false,"");
	validPL->set<RealType>("q2",false,"");
	validPL->set<RealType>("q3",false,"");
	validPL->set<bool>("isSaturationH",false,"");
	validPL->set<bool>("isHyper",false,"");
  }

  if (matModel == "MooneyRivlin")
  {
	 validPL->set<RealType>("c1",false,"");
	 validPL->set<RealType>("c2",false,"");
	 validPL->set<RealType>("c",false,"");
  }

  if (matModel == "MooneyRivlinDamage")
  {
	 validPL->set<RealType>("c1",false,"");
	 validPL->set<RealType>("c2",false,"");
	 validPL->set<RealType>("c",false,"");
	 validPL->set<RealType>("zeta_inf",false,"");
	 validPL->set<RealType>("iota",false,"");
  }

  if (matModel == "MooneyRivlinIncompressible")
  {
	 validPL->set<RealType>("c1",false,"");
	 validPL->set<RealType>("c2",false,"");
	 validPL->set<RealType>("mult",false,"");
  }

  if (matModel == "MooneyRivlinIncompDamage")
  {
	 validPL->set<RealType>("c1",false,"");
	 validPL->set<RealType>("c2",false,"");
	 validPL->set<RealType>("mult",false,"");
	 validPL->set<RealType>("zeta_inf",false,"");
	 validPL->set<RealType>("iota",false,"");
  }

  if (matModel == "AAA")
  {
	 validPL->set<RealType>("alpha",false,"");
	 validPL->set<RealType>("beta",false,"");
	 validPL->set<RealType>("mult",false,"");
  }

  if (matModel == "RIHMR")
  {
	validPL->sublist("Recovery Modulus", false, "");
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

