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

#include "ProjectionProblem.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"


Albany::ProjectionProblem::
ProjectionProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
			const Teuchos::RCP<ParamLib>& paramLib_,
			const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_ + 9), // additional DOF for pore pressure
  haveSource(false),
  numDim(numDim_),
  projection(params->sublist("Projection").get("Projection Variable", ""),params->sublist("Projection").get("Projection Rank",0),params->sublist("Projection").get("Projection Comp",0),numDim)
{
 
  std::string& method = params->get("Name", "Total Lagrangian Plasticity with Projection ");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

  matModel = params->sublist("Material Model").get("Model Name", "NeoHookean");
  projectionVariable = params->sublist("Projection").get("Projection Variable", "");
  projectionRank = params->sublist("Projection").get("Projection Rank",0);
  *out << "Projection Variable: " << projectionVariable << std::endl;
  *out << "Projection Variable Rank: " << projectionRank << std::endl;

  // Only run if there is a projection variable defined
  if(projection.isProjected())
  {
 	  // For debug purposes
 	  *out << "Will variable be projected? " << projection.isProjected() << std::endl;
 	  *out << "Number of components: " << projection.getProjectedComponents() << std::endl;
 	  *out << "Rank of variable: " << projection.getProjectedRank() << std::endl;

 	  /* the evaluator constructor requires information on the size of the
 	   * projected variable as boolean flags in the argument list. Allowed
 	   * variable types are vector, (rank 2) tensor, or scalar (default).
 	   */
 	  switch(projection.getProjectedRank()){
 	  case 1:
 		 isProjectedVarVector = true;
 		 isProjectedVarTensor = false;
 		 break;

 	  case 2:
 		  //isProjectedVarVector = false;
 		  //isProjectedVarTensor = true;
 		  isProjectedVarVector = true;
 		  isProjectedVarTensor = false;
 		  break;

 	  default:
 		  isProjectedVarVector = false;
 		  isProjectedVarTensor = false;
 		  break;
 	  }
  }

// Changing this ifdef changes ordering from  (X,Y,T) to (T,X,Y)
//#define NUMBER_T_FIRST
#ifdef NUMBER_T_FIRST
  T_offset=0;
  X_offset=projection.getProjectedComponents();
#else
  X_offset=0;
  T_offset=numDim;
#endif
}

Albany::ProjectionProblem::
~ProjectionProblem()
{
}

//the following function returns the problem information required for setting the rigid body modes (RBMs) for elasticity problems (in src/Albany_SolverFactory.cpp)
//written by IK, Feb. 2012 
void Albany::ProjectionProblem::getRBMInfoForML(
   int& numPDEs, int& numElasticityDim, int& numScalar,  int& nullSpaceDim)
{
  numPDEs = numDim + projection.getProjectedComponents();
  numElasticityDim = numDim;
  numScalar = projection.getProjectedComponents();
  if (numDim == 1) {nullSpaceDim = 0; }
  else {
    if (numDim == 2) {nullSpaceDim = 3; }
    if (numDim == 3) {nullSpaceDim = 6; }
  }
}


void
Albany::ProjectionProblem::
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
Albany::ProjectionProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<ProjectionProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

void
Albany::ProjectionProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<string> dirichletNames(neq);
  dirichletNames[X_offset] = "X";
  if (numDim>1) dirichletNames[X_offset+1] = "Y";
  if (numDim>2) dirichletNames[X_offset+2] = "Z";
  dirichletNames[T_offset] = "T";
  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::ProjectionProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidProjectionProblemParams");

  validPL->sublist("Material Model", false, "");
  validPL->set<bool>("avgJ", false, "Flag to indicate the J should be averaged");
  validPL->set<bool>("volavgJ", false, "Flag to indicate the J should be volume averaged");
  validPL->set<bool>("weighted_Volume_Averaged_J", false, "Flag to indicate the J should be volume averaged with stabilization");
  validPL->sublist("Elastic Modulus", false, "");
  validPL->sublist("Shear Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");
  validPL->sublist("Projection", false, "");

  if (matModel=="J2"|| matModel=="J2Fiber"){
   validPL->set<bool>("Compute Dislocation Density Tensor", false, "Flag to compute the dislocaiton density tensor (only for 3D)");
   validPL->sublist("Hardening Modulus", false, "");
   validPL->sublist("Saturation Modulus", false, "");
   validPL->sublist("Saturation Exponent", false, "");
   validPL->sublist("Yield Strength", false, "");

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
  }

  return validPL;
}

void
Albany::ProjectionProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}
