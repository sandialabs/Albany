//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_StokesFOThickness.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include <string>
#include <string.hpp> // For util::upper_case (do not confuse this with <string>! string.hpp is an Albany file)

// Uncomment for some setup output
#define OUTPUT_TO_SCREEN

LandIce::StokesFOThickness::
StokesFOThickness( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_),
  discParams(discParams_),
  numDim(numDim_),
  use_sdbcs_(false)
{
  //Set # of PDEs per node.
  std::string eqnSet = params_->sublist("Equation Set").get<std::string>("Type", "LandIce");
  neq = 3; //LandIce FO Stokes system is a system of 2 PDEs

  // Set the num PDEs for the null space object to pass to ML
  this->rigidBodyModes->setNumPDEs(neq);

  // Need to allocate a fields in mesh database
  if (params->isParameter("Required Fields"))
  {
    // Need to allocate a fields in mesh database
    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Fields");
    for (int i(0); i<req.size(); ++i)
      this->requirements.push_back(req[i]);
  }

  // Parsing the LandIce boundary conditions sublist
  auto landice_bcs_params = Teuchos::sublist(params,"LandIce BCs");
  int num_bcs = landice_bcs_params->get<int>("Number",0);
  for (int i=0; i<num_bcs; ++i) {
    auto this_bc = Teuchos::sublist(landice_bcs_params,Albany::strint("BC",i));
    std::string type_str = util::upper_case(this_bc->get<std::string>("Type"));

    LandIceBC type;
    if (type_str=="BASAL FRICTION") {
      type = LandIceBC::BasalFriction;
    } else if (type_str=="LATERAL") {
      type = LandIceBC::Lateral;
    } else if (type_str=="SYNTETIC TEST") {
      type = LandIceBC::SynteticTest;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                  "Error! Unknown LandIce bc '" + type_str + "'.\n");
    }
    landice_bcs[type].push_back(this_bc);
  }


  // Surface side, where velocity diagnostics are computed (e.g., velocity mismatch)
  surfaceSideName = params->isParameter("Surface Side Name") ? params->get<std::string>("Surface Side Name") : "__INVALID__";

  // Basal side, where thickness-related diagnostics are computed (e.g., SMB)
  basalSideName   = params->isParameter("Basal Side Name")   ? params->get<std::string>("Basal Side Name")   : "__INVALID__";

  if (basalSideName!="__INVALID__")
  {
    // Defining the thickness equation only in 2D (basal side)
    sideSetEquations[2].push_back(basalSideName);
  }

  dof_names.resize(2);
  dof_names[0] = "Velocity";
  dof_names[1] = "Thickness";

  resid_names.resize(2);
  resid_names[0] = "StokesFO Residual";
  resid_names[1] = "Thickness Residual";

  scatter_names.resize(2);
  scatter_names[0] = "Scatter " + resid_names[0];
  scatter_names[1] = "Scatter " + resid_names[1];
}

void
LandIce::StokesFOThickness::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

  // Building cell basis and cubature
  const CellTopologyData * const cell_top = &meshSpecs[0]->ctd;
  cellBasis = Albany::getIntrepid2Basis(*cell_top);
  cellType = rcp(new shards::CellTopology (cell_top));

  Intrepid2::DefaultCubatureFactory cubFactory;
  cellCubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs[0]->cubatureDegree);

  const int worksetSize     = meshSpecs[0]->worksetSize;
  vecDimFO                  = std::min((int)neq,(int)2);
  const int numCellSides    = cellType->getFaceCount();
  const int numCellVertices = cellType->getNodeCount();
  const int numCellNodes    = cellBasis->getCardinality();
  const int numCellQPs      = cellCubature->getNumPoints();

  dl = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim,vecDimFO));
  // dl_full = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim, neq));

  int sideDim = numDim-1;
  for (auto it : landice_bcs) {
    for (auto pl: it.second) {
      std::string ssName = pl->get<std::string>("Side Set Name");
      TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(ssName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                  "Error! Either the side set name is wrong or something went wrong while building the side mesh specs.\n");
      const Albany::MeshSpecsStruct& sideMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(ssName)[0];

      // Building also side structures
      const CellTopologyData * const side_top = &sideMeshSpecs.ctd;
      sideBasis[ssName] = Albany::getIntrepid2Basis(*side_top);
      sideType[ssName] = rcp(new shards::CellTopology (side_top));

      // If there's no side discretiation, then sideMeshSpecs.cubatureDegree will be -1, and the user need to specify a cubature degree somewhere else
      int sideCubDegree = sideMeshSpecs.cubatureDegree;
      if (pl->isParameter("Cubature Degree")) {
        sideCubDegree = pl->get<int>("Cubature Degree");
      }
      TEUCHOS_TEST_FOR_EXCEPTION (sideCubDegree<0, std::runtime_error, "Error! Missing cubature degree information on side '" << ssName << ".\n"
                                                                       "       Either add a side discretization, or specify 'Cubature Degree' in sublist '" + pl->name() + "'.\n");
      sideCubature[ssName] = cubFactory.create<PHX::Device, RealType, RealType>(*sideType[ssName], sideCubDegree);

      int numSideVertices = sideType[ssName]->getNodeCount();
      int numSideNodes    = sideBasis[ssName]->getCardinality();
      int numSideQPs      = sideCubature[ssName]->getNumPoints();

      dl->side_layouts[ssName] = rcp(new Albany::Layouts(worksetSize,numSideVertices,numSideNodes,
                                                         numSideQPs,sideDim,numDim,numCellSides,vecDimFO));

      // dl_full->side_layouts[ssName] = rcp(new Albany::Layouts(worksetSize,numSideVertices,numSideNodes,
                                                              // numSideQPs,sideDim,numDim,numCellSides,neq));
    }
  }

  if (basalSideName!="__INVALID__" && dl->side_layouts.find(basalSideName)==dl->side_layouts.end())
  {
    TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(basalSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                "Error! Either 'Basal Side Name' is wrong or something went wrong while building the side mesh specs.\n");
    const Albany::MeshSpecsStruct& basalMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(basalSideName)[0];

    // Building also basal side structures
    const CellTopologyData * const side_top = &basalMeshSpecs.ctd;
    sideBasis[basalSideName] = Albany::getIntrepid2Basis(*side_top);
    sideType[basalSideName] = rcp(new shards::CellTopology (side_top));

    sideCubature[basalSideName] = cubFactory.create<PHX::Device, RealType, RealType>(*sideType.at(basalSideName), basalMeshSpecs.cubatureDegree);

    int numBasalSideVertices = sideType.at(basalSideName)->getNodeCount();
    int numBasalSideNodes    = sideBasis.at(basalSideName)->getCardinality();
    int numBasalSideQPs      = sideCubature.at(basalSideName)->getNumPoints();

    dl->side_layouts[basalSideName] = rcp(new Albany::Layouts(worksetSize,numBasalSideVertices,numBasalSideNodes,
                                                              numBasalSideQPs,sideDim,numDim,numCellSides,vecDimFO));
    // dl_full->side_layouts[basalSideName] = rcp(new Albany::Layouts(worksetSize,numBasalSideVertices,numBasalSideNodes,
                                                                   // numBasalSideQPs,sideDim,numDim,numCellSides,neq));
  }

  if (surfaceSideName!="INVALID")
  {
    TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(surfaceSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                  "Error! Either 'Surface Side Name' is wrong or something went wrong while building the side mesh specs.\n");

    const Albany::MeshSpecsStruct& surfaceMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(surfaceSideName)[0];

    // Building also surface side structures
    const CellTopologyData * const side_top = &surfaceMeshSpecs.ctd;
    sideBasis[surfaceSideName] = Albany::getIntrepid2Basis(*side_top);
    sideType[surfaceSideName] = rcp(new shards::CellTopology (side_top));

    sideCubature[surfaceSideName] = cubFactory.create<PHX::Device, RealType, RealType>(*sideType.at(basalSideName), surfaceMeshSpecs.cubatureDegree);

    int numSurfaceSideVertices = sideType.at(surfaceSideName)->getNodeCount();
    int numSurfaceSideNodes    = sideBasis.at(surfaceSideName)->getCardinality();
    int numSurfaceSideQPs      = sideCubature.at(surfaceSideName)->getNumPoints();

    dl->side_layouts[surfaceSideName] = rcp(new Albany::Layouts(worksetSize,numSurfaceSideVertices,numSurfaceSideNodes,
                                                                numSurfaceSideQPs,sideDim,numDim,numCellSides,vecDimFO));
    // dl_full->side_layouts[surfaceSideName] = rcp(new Albany::Layouts(worksetSize,numSurfaceSideVertices,numSurfaceSideNodes,
                                                                     // numSurfaceSideQPs,sideDim,numDim,numCellSides,neq));
  }

#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  int commRank = Teuchos::GlobalMPISession::getRank();
  int commSize = Teuchos::GlobalMPISession::getNProc();
  out->setProcRankAndSize(commRank, commSize);
  out->setOutputToRootOnly(0);

  *out << "=== Field Dimensions ===\n"
       << " Volume:\n"
       << "   Workset     = " << worksetSize << "\n"
       << "   Vertices    = " << numCellVertices << "\n"
       << "   CellNodes   = " << numCellNodes << "\n"
       << "   CellQuadPts = " << numCellQPs << "\n"
       << "   Dim         = " << numDim << "\n"
       << "   VecDim      = " << neq << "\n"
       << "   VecDimFO    = " << vecDimFO << "\n";
  for (auto it : dl->side_layouts) {
    *out << " Side Set '" << it.first << "':\n" 
         << "  Vertices   = " << it.second->vertices_vector->dimension(1) << "\n"
         << "  Nodes      = " << it.second->node_scalar->dimension(1) << "\n"
         << "  QuadPts    = " << it.second->qp_scalar->dimension(1) << "\n";
  }
#endif

 /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM, Teuchos::null);
  constructDirichletEvaluators(*meshSpecs[0]);

  if(meshSpecs[0]->ssNames.size() > 0) // Build a sideset evaluator if sidesets are present
     constructNeumannEvaluators(meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
LandIce::StokesFOThickness::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes

  Albany::ConstructEvaluatorsOp<StokesFOThickness> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);

  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);

  return *op.tags;
}

void
LandIce::StokesFOThickness::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
   for (int i=0; i<neq-1; i++) {
     std::stringstream s; s << "U" << i;
     dirichletNames[i] = s.str();
   }
   dirichletNames[neq-1] = "H";
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
   use_sdbcs_ = dirUtils.useSDBCs();
   offsets_ = dirUtils.getOffsets();
   nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

// Neumann BCs
void
LandIce::StokesFOThickness::constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{

   // Note: we only enter this function if sidesets are defined in the mesh file
   // i.e. meshSpecs.ssNames.size() > 0

   Albany::BCUtils<Albany::NeumannTraits> nbcUtils;

   // Check to make sure that Neumann BCs are given in the input file

   if(!nbcUtils.haveBCSpecified(this->params)) {
      return;
   }


   // Construct BC evaluators for all side sets and names
   // Note that the string index sets up the equation offset, so ordering is important

   std::vector<std::string> neumannNames(neq + 1);
   Teuchos::Array<Teuchos::Array<int> > offsets;
   offsets.resize(neq + 1);

   neumannNames[0] = "U0";
   offsets[0].resize(1);
   offsets[0][0] = 0;
   offsets[neq].resize(neq-1);
   offsets[neq][0] = 0;

   if (neq-1>1){
      neumannNames[1] = "U1";
      offsets[1].resize(1);
      offsets[1][0] = 1;
      offsets[neq][1] = 1;
   }

   if (neq-1>2){
     neumannNames[2] = "U2";
      offsets[2].resize(1);
      offsets[2][0] = 2;
      offsets[neq][2] = 2;
   }

   neumannNames[neq-1] = "H";
   offsets[neq-1].resize(1);
   offsets[neq-1][0] = neq-1;

   neumannNames[neq] = "all";

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dCdx, dCdy, dCdz), or dCdn, not both
   std::vector<std::string> condNames(3); //(dCdx, dCdy, dCdz), dCdn, P
   Teuchos::ArrayRCP<std::string> dof_names(2);
     dof_names[0] = "Velocity";
     dof_names[1] = "Thickness";

   // Note that sidesets are only supported for two and 3D currently
   if(numDim == 2)
    condNames[0] = "(dFluxdx, dFluxdy)";
   else if(numDim == 3)
    condNames[0] = "(dFluxdx, dFluxdy, dFluxdz)";
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

   condNames[1] = "dFluxdn";
   condNames[2] = "P";

   nfm.resize(1); // LandIce problem only has one element block

   nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, true, 0,
                                          condNames, offsets, dl,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
LandIce::StokesFOThickness::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidStokesFOThicknessProblemParams");

  validPL->sublist("LandIce Viscosity", false, "");
  validPL->sublist("LandIce Surface Gradient", false, "");
  validPL->sublist("Equation Set", false, "");
  validPL->sublist("Body Force", false, "");
  validPL->sublist("LandIce Physical Parameters", false, "");
  validPL->set<double>("Time Step", 1.0, "Time step for divergence flux ");
  validPL->set<Teuchos::RCP<double> >("Time Step Ptr", Teuchos::null, "Time step ptr for divergence flux ");
  validPL->sublist("LandIce BCs", false, "Specify boundary conditions specific to LandIce (bypass usual Neumann/Dirichlet classes)");
  validPL->sublist("Parameter Fields", false, "Parameter Fields to be registered");
  validPL->set<std::string> ("Basal Side Name", "", "Name of the basal side set");
  validPL->set<std::string> ("Surface Side Name", "", "Name of the surface side set");
  validPL->set<Teuchos::Array<std::string> > ("Required Fields", Teuchos::Array<std::string>(), "");
  validPL->set<Teuchos::Array<std::string> > ("Required Basal Fields", Teuchos::Array<std::string>(), "");
  validPL->set<Teuchos::Array<std::string> > ("Required Surface Fields", Teuchos::Array<std::string>(), "");
  validPL->set<int> ("Layered Data Length", 0, "Number of layers in input layered data files.");

  return validPL;
}

