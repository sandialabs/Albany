//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "HDiffusionDeformationProblem.hpp"
#include "Albany_InitialCondition.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_EvaluatorUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"

Albany::HDiffusionDeformationProblem::
HDiffusionDeformationProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
			const Teuchos::RCP<ParamLib>& paramLib_,
			const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_ + 2),
  haveSource(false),



  numDim(numDim_)
{
 
  std::string& method = params->get("Name", "Hydrogen Diffusion-Deformation");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

  matModel = params->sublist("Material Model").get("Model Name","J2");

// Changing this ifdef changes ordering from  (X,Y,T) to (T,X,Y)
//#define NUMBER_T_FIRST
#ifdef NUMBER_T_FIRST
  T_offset=0;
  Thydro_offset= 1;
  X_offset=2;
#else
  X_offset=0;
  T_offset=numDim;
  Thydro_offset= numDim+1;
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

Albany::HDiffusionDeformationProblem::
~HDiffusionDeformationProblem()
{
}

void
Albany::HDiffusionDeformationProblem::
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
  constructDirichletEvaluators(*meshSpecs[0]);
}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::HDiffusionDeformationProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<HDiffusionDeformationProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

void
Albany::HDiffusionDeformationProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<string> dirichletNames(neq);
  dirichletNames[X_offset] = "X";
  if (numDim>1) dirichletNames[X_offset+1] = "Y";
  if (numDim>2) dirichletNames[X_offset+2] = "Z";
  dirichletNames[T_offset] = "T";
  dirichletNames[Thydro_offset] = "Thydro";
  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::HDiffusionDeformationProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidHDiffusionDeformationProblemParams");

  validPL->sublist("Material Model", false, "");
  validPL->set<bool>("avgJ", false, "Flag to indicate the J should be averaged");
  validPL->set<bool>("volavgJ", false, "Flag to indicate the J should be volume averaged");
  validPL->set<bool>("weighted_Volume_Averaged_J", false, "Flag to indicate the J should be volume averaged with stabilization");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->sublist("Elastic Modulus", false, "");
  validPL->sublist("Bulk Modulus", false, "");
  validPL->sublist("Shear Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");
  validPL->sublist("Hardening Modulus", false, "");
  validPL->sublist("Saturation Modulus", false, "");
  validPL->sublist("Saturation Exponent", false, "");
  validPL->sublist("Yield Strength", false, "");
  validPL->set<RealType>("Reference Temperature", false, "");
  validPL->set<RealType>("Thermal Expansion Coefficient", false, "");
  validPL->set<RealType>("Density", false, "");
  validPL->set<RealType>("Heat Capacity", false, "");
  validPL->sublist("Temperature", false, "");
  validPL->set<RealType>("Avogadro Number", false, "");
  validPL->sublist("Trap Binding Energy", false, "");
  validPL->set<RealType>("Ideal Gas Constant", false, "");
  validPL->sublist("Diffusion Activation Enthalpy", false, "");
  validPL->sublist("Pre Exponential Factor", false, "");
  validPL->sublist("Diffusion Coefficient", false, "");
  validPL->sublist("Equilibrium Constant", false, "");
  validPL->sublist("Trapped Solvent", false, "");
  validPL->sublist("Trapped Concentration", false, "");
  validPL->sublist("Total Concentration", false, "");
  validPL->sublist("Molar Volume", false, "");
  validPL->sublist("Partial Molar Volume", false, "");
  validPL->sublist("Stress Free Total Concentration", false, "");
  validPL->sublist("Effective Diffusivity", false, "");
  validPL->sublist("Strain Rate Factor", false, "");
  validPL->sublist("Tau Contribution", false, "");
  validPL->sublist("CL Unit Gradient", false, "");
  validPL->sublist("Lattice Deformation Gradient", false, "");
  validPL->sublist("Element Length", false, "");
  validPL->sublist("Stabilization Parameter", false, "");
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
Albany::HDiffusionDeformationProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}

