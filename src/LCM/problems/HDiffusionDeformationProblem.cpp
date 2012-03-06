/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

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

// Changing this ifdef changes ordering from  (X,Y,T) to (T,X,Y)
//#define NUMBER_T_FIRST
#ifdef NUMBER_T_FIRST
  T_offset=0;
  X_offset=1;
#else
  X_offset=0;
  T_offset=numDim;
  Thydro_offset= numDim+1;
#endif
}

Albany::HDiffusionDeformationProblem::
~HDiffusionDeformationProblem()
{
}

//the following function returns the problem information required for setting the rigid body modes (RBMs) for elasticity problems (in src/Albany_SolverFactory.cpp)
//written by IK, Feb. 2012 
void Albany::HDiffusionDeformationProblem::getRBMInfoForML(
   int& numPDEs, int& numElasticityDim, int& numScalar,  int& nullSpaceDim)
{
  numPDEs = numDim + 2;
  numElasticityDim = numDim;
  numScalar = 2;
  if (numDim == 1) {nullSpaceDim = 0; }
  else {
    if (numDim == 2) {nullSpaceDim = 3; }
    if (numDim == 3) {nullSpaceDim = 6; }
  }
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

  validPL->set<bool>("avgJ", false, "Flag to indicate the J should be averaged");
  validPL->set<bool>("volavgJ", false, "Flag to indicate the J should be volume averaged");
  validPL->sublist("Thermal Conductivity", false, "");
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
  validPL->sublist("Avogadro Number", false, "");
  validPL->sublist("Trap Binding Energy", false, "");
  validPL->sublist("Ideal Gas Constant", false, "");
  validPL->sublist("Diffusion Activation Enthalpy", false, "");
  validPL->sublist("Pre Exponential Factor", false, "");
  validPL->sublist("Diffusion Coefficient", false, "");
  validPL->sublist("Equilibrium Constant", false, "");
  validPL->sublist("Trapped Solvent", false, "");
  validPL->sublist("TrappedConcentration", false, "");
  validPL->sublist("Molar Volume", false, "");
  validPL->sublist("Effective Diffusivity", false, "");
  validPL->sublist("Strain Rate Factor", false, "");
  validPL->sublist("Tau Contribution", false, "");

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

