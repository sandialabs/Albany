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
  constructDirichletEvaluators(*meshSpecs[0]);
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
Albany::NonlinearElasticityProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}

