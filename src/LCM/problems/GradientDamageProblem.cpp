//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "GradientDamageProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"

Albany::GradientDamageProblem::
GradientDamageProblem(
		      const Teuchos::RCP<Teuchos::ParameterList>& params_,
		      const Teuchos::RCP<ParamLib>& paramLib_,
		      const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_ + 1),
  haveSource(false),
  numDim(numDim_)
{
 
  std::string& method = params->get("Name", "GradientDamage ");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

  // Problem supports transient
//  params->set<int>("Number Of Solution Vectors", 2);

// Changing this ifdef changes ordering from  (X,Y,D) to (D,X,Y)
//#define NUMBER_D_FIRST
#ifdef NUMBER_D_FIRST
  D_offset=0;
  X_offset=1;
#else
  X_offset=0;
  D_offset=numDim;
#endif

// the following function returns the problem information required for setting the rigid body modes (RBMs) for elasticity problems
//written by IK, Feb. 2012

  int numScalar = 1;
  int nullSpaceDim = 0;
  if (numDim == 1) {nullSpaceDim = 0; }
  else {
    if (numDim == 2) {nullSpaceDim = 3; }
    if (numDim == 3) {nullSpaceDim = 6; }
  }

  rigidBodyModes->setParameters(numDim + 1, numDim, numScalar, nullSpaceDim);

}

Albany::GradientDamageProblem::
~GradientDamageProblem()
{
}

void
Albany::GradientDamageProblem::
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
  constructDirichletEvaluators(*meshSpecs[0]);
}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
Albany::GradientDamageProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<GradientDamageProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
Albany::GradientDamageProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  dirichletNames[X_offset] = "X";
  if (numDim>1) dirichletNames[X_offset+1] = "Y";
  if (numDim>2) dirichletNames[X_offset+2] = "Z";
  dirichletNames[D_offset] = "D";
  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::GradientDamageProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidGradientDamageProblemParams");

  validPL->sublist("Bulk Modulus", false, "");
  validPL->sublist("Shear Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");
  validPL->sublist("Damage Length Scale", false, "");
  validPL->sublist("Hardening Modulus", false, "");
  validPL->sublist("Yield Strength", false, "");
  validPL->sublist("Saturation Modulus", false, "");
  validPL->sublist("Saturation Exponent", false, "");
  validPL->set<double>("gc", false, "");
  validPL->set<bool>("avgJ", false, "Flag to indicate the J should be averaged");
  validPL->set<bool>("volavgJ", false, "Flag to indicate the J should be volume averaged");
  validPL->set<bool>("Use Composite Tet 10", false, "Flag to use the compostie tet 10 basis in Intrepid2");

  return validPL;
}

void
Albany::GradientDamageProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>>> oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>>> newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}
