//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "ThermoPoroPlasticityProblem.hpp"

#include "Intrepid2_FieldContainer.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"

#include "Albany_EvaluatorUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"


Albany::ThermoPoroPlasticityProblem::
ThermoPoroPlasticityProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
			const Teuchos::RCP<ParamLib>& paramLib_,
			const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_ + 2), // additional DOFs
                                                            // one for pore pressure
                                                            // one for temperature
  haveSource(false),
  numDim(numDim_)
{
 
  std::string& method = params->get("Name", "Total Lagrangian ThermoPoroPlasticity ");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

  matModel = params->sublist("Material Model").get("Model Name", "Neohookean");

// Changing this ifdef changes ordering from  (X,Y,T) to (T,X,Y)
//#define NUMBER_T_FIRST
#ifdef NUMBER_T_FIRST
  T_offset=0;
  TEMP_offset = 1;
  X_offset=2;
#else
  X_offset=0;
  T_offset=numDim;
  TEMP_offset = numDim + 1;
#endif

// the following function returns the problem information required for setting the rigid body modes (RBMs) for elasticity problems
//written by IK, Feb. 2012

  int numScalar = 2;
  int nullSpaceDim = 0;
  if (numDim == 1) {nullSpaceDim = 0; }
  else {
    if (numDim == 2) {nullSpaceDim = 3; }
    if (numDim == 3) {nullSpaceDim = 6; }
  }

  rigidBodyModes->setParameters(numDim + 2, numDim, numScalar, nullSpaceDim);

}

Albany::ThermoPoroPlasticityProblem::
~ThermoPoroPlasticityProblem()
{
}

void
Albany::ThermoPoroPlasticityProblem::
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
Albany::ThermoPoroPlasticityProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<ThermoPoroPlasticityProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
Albany::ThermoPoroPlasticityProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  dirichletNames[X_offset] = "X";
  if (numDim>1) dirichletNames[X_offset+1] = "Y";
  if (numDim>2) dirichletNames[X_offset+2] = "Z";
  dirichletNames[T_offset] = "T";
  dirichletNames[TEMP_offset] = "TEMP";
  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::ThermoPoroPlasticityProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidThermoPoroPlasticityProblemParams");
  validPL->sublist("Material Model", false, "");
  validPL->set<bool>("avgJ", false, "Flag to indicate the J should be averaged");
  validPL->set<bool>("volavgJ", false, "Flag to indicate the J should be volume averaged");
  validPL->set<bool>("weighted_Volume_Averaged_J", false, "Flag to indicate the J should be volume averaged with stabilization");
  validPL->sublist("Porosity", false, "");
  validPL->sublist("Biot Coefficient", false, "");
  validPL->sublist("Biot Modulus", false, "");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->sublist("Kozeny-Carman Permeability", false, "");
  validPL->sublist("Shear Modulus", false, "");
  validPL->sublist("Elastic Modulus", false, "");
  validPL->sublist("Bulk Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");
  validPL->sublist("Stabilization Parameter", false, "");
  validPL->sublist("Reference Temperature", false, "");
  validPL->sublist("Skeleton Thermal Expansion", false, "");
  validPL->sublist("Pore-Fluid Thermal Expansion", false, "");
  validPL->sublist("Skeleton Density", false, "");
  validPL->sublist("Pore-Fluid Density", false, "");
  validPL->sublist("Skeleton Specific Heat", false, "");
  validPL->sublist("Pore-Fluid Specific Heat", false, "");
 // validPL->sublist("Mixture Thermal Expansion", false, "");
 // validPL->sublist("Mixture Specific Heat", false, "");
  if (matModel == "J2"|| matModel == "J2Fiber" || matModel == "GursonFD")
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
 	validPL->set<RealType>("Mx_f1",false,"");
 	validPL->set<RealType>("My_f1",false,"");
 	validPL->set<RealType>("Mz_f1",false,"");
 	validPL->set<RealType>("k_f2",false,"");
 	validPL->set<RealType>("q_f2",false,"");
 	validPL->set<RealType>("vol_f2",false,"");
 	validPL->set<RealType>("xiinf_f2",false,"");
 	validPL->set<RealType>("tau_f2",false,"");
 	validPL->set<RealType>("Mx_f2",false,"");
 	validPL->set<RealType>("My_f2",false,"");
 	validPL->set<RealType>("Mz_f2",false,"");
   }

   if (matModel == "GursonFD")
   {
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
   }

  return validPL;
}

void
Albany::ThermoPoroPlasticityProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>>> oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>>> newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}

