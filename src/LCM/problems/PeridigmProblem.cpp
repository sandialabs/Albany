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
  haveSource(false), numDim(numDim_), haveMatDB(false),
  use_sdbcs_(false),
  supportsTransient(false)
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
    materialDataBase = Teuchos::rcp(new Albany::MaterialDatabase(filename, commT));
  }

  // Determine if transient analyses should be supported
  supportsTransient = false;
  if(params->isParameter("Solution Method")){
    std::string solutionMethod = params->get<std::string>("Solution Method");
    if(solutionMethod == "Transient" || solutionMethod == "Transient Tempus"){
      supportsTransient = true;
    }
  }

  // "Sphere Volume" is a required field for peridynamic simulations that read an Exodus sphere mesh
  // requirements.push_back("Sphere Volume");

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
    if (meshSpecs[ps]->ssNames.size() > 0) {
      constructNeumannEvaluators(meshSpecs[ps]);
    }
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
   use_sdbcs_ = dirUtils.useSDBCs();
   offsets_ = dirUtils.getOffsets();
   nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

void
Albany::PeridigmProblem::constructNeumannEvaluators(
    const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
  Albany::BCUtils<Albany::NeumannTraits> neuUtils;

  // Check to make sure that Neumann BCs are given in the input file
  if (!neuUtils.haveBCSpecified(this->params)) {
    return;
  }

  // Construct BC evaluators for all side sets and names
  // Note that the string index sets up the equation offset,
  // so ordering is important

  std::vector<std::string> neumannNames(neq + 1);
  Teuchos::Array<Teuchos::Array<int>> offsets;
  offsets.resize(neq + 1);

  int index = 0;

  neumannNames[index] = "sig_x";
  offsets[index].resize(1);
  offsets[index++][0] = 0;

  // The Neumann BC code uses offsets[neq].size() as num dim, so use numDim
  // here rather than neq.
  offsets[neq].resize(numDim);
  offsets[neq][0] = 0;

  if (numDim > 1) {
    neumannNames[index] = "sig_y";
    offsets[index].resize(1);
    offsets[index++][0] = 1;
    offsets[neq][1] = 1;
  }

  if (numDim > 2) {
    neumannNames[index] = "sig_z";
    offsets[index].resize(1);
    offsets[index++][0] = 2;
    offsets[neq][2] = 2;
  }

  neumannNames[neq] = "all";

  // Construct BC evaluators for all possible names of conditions
  // Should only specify flux vector components (dudx, dudy, dudz),
  // or dudn, not both
  std::vector<std::string> condNames(3); //dudx, dudy, dudz, dudn, P
  Teuchos::ArrayRCP<std::string> dof_names(1);
  dof_names[0] = "Displacement";

  // Note that sidesets are only supported for two and 3D currently
  if (numDim == 2)
    condNames[0] = "(t_x, t_y)";
  else if (numDim == 3)
    condNames[0] = "(t_x, t_y, t_z)";
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        '\n' << "Error: Sidesets only supported in 2 and 3D." << '\n');

  condNames[1] = "dudn";
  condNames[2] = "P";

  // DJL jump through hoops because we don't have a stored data layout
  Teuchos::RCP<shards::CellTopology> cellType = Teuchos::rcp(new shards::CellTopology (&meshSpecs->ctd));

  // The datalayout will be invalid for particle/sphere elements
  if (cellType->getNodeCount() != 1){

    nfm.resize(1); // Albany has issues with NBC ... currently hard-coded to one block?
                   // I could see this failing if there are multiple element blocks with different element types

    Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis = Albany::getIntrepid2Basis(meshSpecs->ctd);
    const int numNodes = intrepidBasis->getCardinality();
    const int worksetSize = meshSpecs->worksetSize;
    Intrepid2::DefaultCubatureFactory cubFactory;
    Teuchos::RCP <Intrepid2::Cubature<PHX::Device>  > cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs->cubatureDegree);
    const int numDim = cubature->getDimension();
    const int numQPts = cubature->getNumPoints();
    const int numVertices = cellType->getNodeCount();
    Teuchos::RCP<Albany::Layouts> dataLayout = Teuchos::rcp(new Albany::Layouts(worksetSize, numVertices, numNodes, numQPts, numDim));

    nfm[0] = neuUtils.constructBCEvaluators(
        meshSpecs,
        neumannNames,
        dof_names,
        true,        // isVectorField
        0,           // offsetToFirstDOF
        condNames,
        offsets,
        dataLayout,
        this->params,
        this->paramLib);
    }
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
