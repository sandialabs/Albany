//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "PoroElasticityProblem.hpp"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"


Albany::PoroElasticityProblem::
PoroElasticityProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
			const Teuchos::RCP<ParamLib>& paramLib_,
			const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_ + 1), // additional DOF for pore pressure
  haveSource(false),
  numDim(numDim_)
{
 
  std::string& method = params->get("Name", "PoroElasticity ");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

  matModel = params->sublist("Material Model").get("Model Name", "LinearElasticity");

// Changing this ifdef changes ordering from  (X,Y,T) to (T,X,Y)
//#define NUMBER_T_FIRST
#ifdef NUMBER_T_FIRST
  T_offset=0;
  X_offset=1;
#else
  X_offset=0;
  T_offset=numDim;
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

Albany::PoroElasticityProblem::
~PoroElasticityProblem()
{
}

void
Albany::PoroElasticityProblem::
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
Albany::PoroElasticityProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<PoroElasticityProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

void
Albany::PoroElasticityProblem::constructDirichletEvaluators(
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
Albany::PoroElasticityProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidPoroElasticityProblemParams");
  validPL->set<bool>("avgJ", false, "Flag to indicate the J should be averaged");
  validPL->set<bool>("volavgJ", false, "Flag to indicate the J should be volume averaged");
  validPL->set<bool>("weighted_Volume_Averaged_J", false, "Flag to indicate the J should be volume averaged with stabilization");
  validPL->sublist("Material Model", false, "");
  validPL->sublist("Porosity", false, "");
  validPL->sublist("Biot Coefficient", false, "");
  validPL->sublist("Biot Modulus", false, "");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->sublist("Kozeny-Carman Permeability", false, "");
  validPL->sublist("Elastic Modulus", false, "");
  validPL->sublist("Shear Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");
  validPL->sublist("Stabilization Parameter", false, "");


  if (matModel == "CapExplicit" || matModel == "CapImplicit")
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
Albany::PoroElasticityProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType>>>> oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType>>>> newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}

