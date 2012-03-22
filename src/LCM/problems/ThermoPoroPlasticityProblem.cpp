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

#include "ThermoPoroPlasticityProblem.hpp"
#include "Albany_InitialCondition.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"


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

  matModel = params->sublist("Material Model").get("Model Name", "NeoHookean");

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
}

Albany::ThermoPoroPlasticityProblem::
~ThermoPoroPlasticityProblem()
{
}

//the following function returns the problem information required for setting the rigid body modes (RBMs) for elasticity problems (in src/Albany_SolverFactory.cpp)
//written by IK, Feb. 2012 
void Albany::ThermoPoroPlasticityProblem::getRBMInfoForML(
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
Albany::ThermoPoroPlasticityProblem::
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
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

void
Albany::ThermoPoroPlasticityProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<string> dirichletNames(neq);
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
  validPL->sublist("Poissons Ratio", false, "");
  validPL->sublist("Stabilization Parameter", false, "");
  validPL->sublist("Reference Temperature", false, "");
  validPL->sublist("Skeleton Thermal Expansion", false, "");
  validPL->sublist("Pore-Fluid Thermal Expansion", false, "");
  validPL->sublist("Skeleton Density", false, "");
  validPL->sublist("Pore-Fluid Density", false, "");
  validPL->sublist("Skeleton Specific Heat", false, "");
  validPL->sublist("Pore-Fluid Specific Heat", false, "");
  validPL->sublist("Bulk Modulus", false, "");
  validPL->sublist("Mixture Thermal Expansion", false, "");
  validPL->sublist("Mixture Specific Heat", false, "");
  if (matModel=="J2"){
   validPL->sublist("Hardening Modulus", false, "");
   validPL->sublist("Saturation Modulus", false, "");
   validPL->sublist("Saturation Exponent", false, "");
   validPL->sublist("Yield Strength", false, "");
  }

  return validPL;
}

void
Albany::ThermoPoroPlasticityProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}

