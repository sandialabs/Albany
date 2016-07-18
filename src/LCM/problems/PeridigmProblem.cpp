//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "PeridigmProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"

Albany::PeridigmProblem::
PeridigmProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                const Teuchos::RCP<ParamLib>& paramLib_,
                const int numDim_,
                Teuchos::RCP<const Teuchos::Comm<int>>& commT):
  Albany::AbstractProblem(params_, paramLib_, numDim_),
  haveSource(false), numDim(numDim_), haveMatDB(false), supportsTransient(false)
{
 
  std::string& method = params->get("Name", "Peridigm Code Coupling ");
  *out << "Problem Name = " << method << std::endl;
  peridigmParams = Teuchos::rcpFromRef(params->sublist("Peridigm Parameters", true));

  // Only support 3D analyses
  TEUCHOS_TEST_FOR_EXCEPTION(neq != 3,
                             Teuchos::Exceptions::InvalidParameter,
                             "\nOnly three-dimensional analyses are suppored when coupling with Peridigm.\n");

  // Read the material data base file, if any
  if(params->isType<std::string>("MaterialDB Filename")){
    std::string filename = params->get<std::string>("MaterialDB Filename");
    materialDataBase = Teuchos::rcp(new LCM::MaterialDatabase(filename, commT));
  }

  // Determine if transient analyses should be supported
  supportsTransient = false;
  if(params->isParameter("Solution Method")){
    std::string solutionMethod = params->get<std::string>("Solution Method");
    if(solutionMethod == "Transient"){
      supportsTransient = true;
    }
  }

  // "Sphere Volume" is a required field for peridynamic simulations that read an Exodus sphere mesh
  requirements.push_back("Sphere Volume");

  // The following function returns the problem information required for setting the rigid body modes (RBMs) for elasticity problems
  // written by IK, Feb. 2012
  int numScalar = 0;
  int nullSpaceDim = 0;
  if (numDim == 1) {nullSpaceDim = 0; }
  else {
    if (numDim == 2) {nullSpaceDim = 3; }
    if (numDim == 3) {nullSpaceDim = 6; }
  }
  rigidBodyModes->setParameters(numDim, numDim, numScalar, nullSpaceDim);
}

Albany::PeridigmProblem::
~PeridigmProblem()
{
}

void
Albany::PeridigmProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>  meshSpecs,
  Albany::StateManager& stateMgr)
{
  /* Construct All Phalanx Evaluators */
  int physSets = meshSpecs.size();
  fm.resize(physSets);

  for (int ps=0; ps<physSets; ps++) {
    fm[ps]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM, Teuchos::null);
  }
  constructDirichletEvaluators(*meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag>>
Albany::PeridigmProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], meshSpecs, stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<PeridigmProblem> op(*this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
Albany::PeridigmProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
   dirichletNames[0] = "X";
   if (neq>1) dirichletNames[1] = "Y";
   if (neq>2) dirichletNames[2] = "Z";
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames, this->params, this->paramLib);
   offsets_ = dirUtils.getOffsets(); 
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::PeridigmProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidPeridigmProblemParams");

  validPL->sublist("Peridigm Parameters", false, "The full parameter list that will be passed to Peridigm.");
  validPL->set<std::string>("MaterialDB Filename", "materials.xml", "Filename of material database xml file");
  validPL->set<bool>("Supports Transient", "false", "Flag for enabling transient analyses");

  return validPL;
}
