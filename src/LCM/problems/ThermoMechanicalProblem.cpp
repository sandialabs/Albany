//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "ThermoMechanicalProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"

Albany::ThermoMechanicalProblem::
ThermoMechanicalProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
			const Teuchos::RCP<ParamLib>& paramLib_,
			const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_ + 1),
  haveSource(false),
  numDim(numDim_)
{
 
  std::string& method = params->get("Name", "ThermoMechanical");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

// Changing this ifdef changes ordering from  (X,Y,T) to (T,X,Y)
//#define NUMBER_T_FIRST
#ifdef NUMBER_T_FIRST
  T_offset=0;
  X_offset=1;
#else
  X_offset=0;
  T_offset=numDim;
#endif

  model = params->sublist("Material Model").get("Model Name","ThermoMechanical");

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

Albany::ThermoMechanicalProblem::
~ThermoMechanicalProblem()
{
}

void
Albany::ThermoMechanicalProblem::
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
Albany::ThermoMechanicalProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<ThermoMechanicalProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
Albany::ThermoMechanicalProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  dirichletNames[X_offset] = "X";
  if (numDim>1) dirichletNames[X_offset+1] = "Y";
  if (numDim>2) dirichletNames[X_offset+2] = "Z";
  dirichletNames[T_offset] = "T";
  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::ThermoMechanicalProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidThermoMechanicalProblemParams");

  validPL->sublist("Thermal Conductivity", false, "");
  validPL->sublist("Bulk Modulus", false, "");
  validPL->sublist("Shear Modulus", false, "");
  validPL->sublist("Hardening Modulus", false, "");
  validPL->sublist("Saturation Modulus", false, "");
  validPL->sublist("Saturation Exponent", false, "");
  validPL->sublist("Yield Strength", false, "");
  validPL->set<RealType>("Reference Temperature", false, "");
  validPL->set<RealType>("Thermal Expansion Coefficient", false, "");
  validPL->set<RealType>("Density", false, "");
  validPL->set<RealType>("Heat Capacity", false, "");
  validPL->sublist("Material Model", false, "");
  validPL->set<bool>("volavgJ", false, "Flag to indicate the J should be volume averaged");
  validPL->set<bool>("weighted_Volume_Averaged_J", false, "Flag to indicate the J should be volume averaged with stabilization");

  return validPL;
}

void
Albany::ThermoMechanicalProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>>> oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>>> newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}

