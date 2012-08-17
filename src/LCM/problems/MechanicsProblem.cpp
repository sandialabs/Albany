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

#include "MechanicsProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"


Albany::MechanicsProblem::
MechanicsProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                 const Teuchos::RCP<ParamLib>& paramLib_,
                 const int numDim_,
                 const Teuchos::RCP<const Epetra_Comm>& comm) :
  Albany::AbstractProblem(params_, paramLib_, numDim_),
  haveSource(false),
  numDim(numDim_),
  haveMatDB(false)
{
 
  std::string& method = params->get("Name", "Mechanics ");
  *out << "Problem Name = " << method << std::endl;
  
  haveSource =  params->isSublist("Source Functions");

  if(params->isType<string>("MaterialDB Filename")){
    haveMatDB = true;
    mtrlDbFilename = params->get<string>("MaterialDB Filename");
    materialDB = Teuchos::rcp(new QCAD::MaterialDatabase(mtrlDbFilename, comm));
  }
  TEUCHOS_TEST_FOR_EXCEPTION(!haveMatDB, std::logic_error,
                             "Mechanics Problem Requires a Material Database");

}

Albany::MechanicsProblem::
~MechanicsProblem()
{
}

//the following function returns the problem information required for setting the rigid body modes (RBMs) for elasticity problems (in src/Albany_SolverFactory.cpp)
//written by IK, Feb. 2012 
void
Albany::MechanicsProblem::getRBMInfoForML(
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
Albany::MechanicsProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  /* Construct All Phalanx Evaluators */
  int physSets = meshSpecs.size();
  cout << "Num MeshSpecs: " << physSets << endl;
  fm.resize(physSets);

  cout << "Calling MechanicsProblem::buildEvaluators" << endl;
  for (int ps=0; ps<physSets; ps++) {
    fm[ps]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM, 
		    Teuchos::null);
  }
  constructDirichletEvaluators(*meshSpecs[0]);
}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::MechanicsProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<MechanicsProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

void
Albany::MechanicsProblem::constructDirichletEvaluators(
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
Albany::MechanicsProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidMechanicsProblemParams");

  // validPL->sublist("Elastic Modulus", false, "");
  // validPL->sublist("Poissons Ratio", false, "");
  // validPL->sublist("Bulk Modulus", false, "");
  // validPL->sublist("Shear Modulus", false, "");

  validPL->set<string>("MaterialDB Filename","materials.xml","Filename of material database xml file");
  
  // validPL->sublist("Material Model", false, "");
  // validPL->set<bool>("avgJ", false, "Flag to indicate the J should be averaged");
  // validPL->set<bool>("volavgJ", false, "Flag to indicate the J should be volume averaged");
  // validPL->set<bool>("weighted_Volume_Averaged_J", false, "Flag to indicate the J should be volume averaged with stabilization");
  // validPL->set<bool>("Use Composite Tet 10", false, "Flag to use the compostie tet 10 basis in Intrepid");


  // if (matModel == "J2"|| matModel == "J2Fiber" || matModel == "GursonFD" || matModel == "RIHMR")
  // {
  //   validPL->set<bool>("Compute Dislocation Density Tensor", false, "Flag to compute the dislocaiton density tensor (only for 3D)");
  //   validPL->sublist("Hardening Modulus", false, "");
  //   validPL->sublist("Yield Strength", false, "");
  //   validPL->sublist("Saturation Modulus", false, "");
  //   validPL->sublist("Saturation Exponent", false, "");
  // }

  // if (matModel == "J2Fiber")
  // {
  //       validPL->set<RealType>("xiinf_J2",false,"");
  //       validPL->set<RealType>("tau_J2",false,"");
  //       validPL->set<RealType>("k_f1",false,"");
  //       validPL->set<RealType>("q_f1",false,"");
  //       validPL->set<RealType>("vol_f1",false,"");
  //       validPL->set<RealType>("xiinf_f1",false,"");
  //       validPL->set<RealType>("tau_f1",false,"");
  //       validPL->set<RealType>("k_f2",false,"");
  //       validPL->set<RealType>("q_f2",false,"");
  //       validPL->set<RealType>("vol_f2",false,"");
  //       validPL->set<RealType>("xiinf_f2",false,"");
  //       validPL->set<RealType>("tau_f2",false,"");
  //       validPL->set<RealType>("X0",false,"");
  //       validPL->set<RealType>("Y0",false,"");
  //       validPL->set<RealType>("Z0",false,"");
  //       validPL->sublist("direction_f1",false,"");
  //       validPL->sublist("direction_f2",false,"");
  //       validPL->sublist("Ring Center",false,"");
  //       validPL->set<bool>("isLocalCoord",false,"");
  // }

  // if (matModel == "GursonFD")
  // {
  //       validPL->set<RealType>("N",false,"");
  //       validPL->set<RealType>("eq0",false,"");
  //       validPL->set<RealType>("f0",false,"");
  //       validPL->set<RealType>("kw",false,"");
  //       validPL->set<RealType>("eN",false,"");
  //       validPL->set<RealType>("sN",false,"");
  //       validPL->set<RealType>("fN",false,"");
  //       validPL->set<RealType>("fc",false,"");
  //       validPL->set<RealType>("ff",false,"");
  //       validPL->set<RealType>("q1",false,"");
  //       validPL->set<RealType>("q2",false,"");
  //       validPL->set<RealType>("q3",false,"");
  //       validPL->set<bool>("isSaturationH",false,"");
  //       validPL->set<bool>("isHyper",false,"");
  // }

  // if (matModel == "MooneyRivlin")
  // {
  //        validPL->set<RealType>("c1",false,"");
  //        validPL->set<RealType>("c2",false,"");
  //        validPL->set<RealType>("c",false,"");
  // }

  // if (matModel == "MooneyRivlinDamage")
  // {
  //        validPL->set<RealType>("c1",false,"");
  //        validPL->set<RealType>("c2",false,"");
  //        validPL->set<RealType>("c",false,"");
  //        validPL->set<RealType>("zeta_inf",false,"");
  //        validPL->set<RealType>("iota",false,"");
  // }

  // if (matModel == "MooneyRivlinIncompressible")
  // {
  //        validPL->set<RealType>("c1",false,"");
  //        validPL->set<RealType>("c2",false,"");
  //        validPL->set<RealType>("mu",false,"");
  // }

  // if (matModel == "RIHMR")
  // {
  //       validPL->sublist("Recovery Modulus", false, "");
  // }

  return validPL;
}

void
Albany::MechanicsProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}

