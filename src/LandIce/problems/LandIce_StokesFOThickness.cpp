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


LandIce::StokesFOThickness::
StokesFOThickness( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_),
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

  basalSideName   = params->isParameter("Basal Side Name")   ? params->get<std::string>("Basal Side Name")   : "INVALID";
  surfaceSideName = params->isParameter("Surface Side Name") ? params->get<std::string>("Surface Side Name") : "INVALID";
  lateralSideName = params->isParameter("Lateral Side Name") ? params->get<std::string>("Lateral Side Name") : "INVALID";
  basalEBName = "INVALID";
  surfaceEBName = "INVALID";
  sliding = params->isSublist("LandIce Basal Friction Coefficient");
  lateral_resid = params->isSublist("LandIce Lateral BC");
  TEUCHOS_TEST_FOR_EXCEPTION (sliding && basalSideName=="INVALID", std::logic_error,
                              "Error! With sliding, you need to provide a valid 'Basal Side Name',\n" );
  TEUCHOS_TEST_FOR_EXCEPTION (lateral_resid && lateralSideName=="INVALID", std::logic_error,
                              "Error! With lateral BC, you need to provide a valid 'Lateral Side Name',\n" );

  // Loading basal requirements (if any)
  if (params->isParameter("Required Basal Fields"))
  {
    TEUCHOS_TEST_FOR_EXCEPTION (basalSideName=="INVALID", std::logic_error, "Error! In order to specify basal requirements, you must also specify a valid 'Basal Side Name'.\n");

    // Need to allocate a fields in basal mesh database
    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Basal Fields");
    this->ss_requirements[basalSideName].reserve(req.size()); // Note: this is not for performance, but to guarantee
    for (int i(0); i<req.size(); ++i)                         //       that ss_requirements.at(basalSideName) does not
      this->ss_requirements[basalSideName].push_back(req[i]); //       throw, even if it's empty...
  }

  // Loading surface requirements (if any)
  if (params->isParameter("Required Surface Fields"))
  {
    TEUCHOS_TEST_FOR_EXCEPTION (surfaceSideName=="INVALID", std::logic_error, "Error! In order to specify surface requirements, you must also specify a valid 'Surface Side Name'.\n");

    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Surface Fields");
    this->ss_requirements[surfaceSideName].reserve(req.size()); // Note: same motivation as for the basal side
    for (int i(0); i<req.size(); ++i)
      this->ss_requirements[surfaceSideName].push_back(req[i]);
  }

  if (basalSideName!="INVALID")
  {
    // Defining the thickness equation only in 2D (basal side)
    sideSetEquations[2].push_back(basalSideName);
  }
}

LandIce::StokesFOThickness::
~StokesFOThickness()
{
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

  elementBlockName = meshSpecs[0]->ebName;

  const int worksetSize     = meshSpecs[0]->worksetSize;
  vecDimFO                  = std::min((int)neq,(int)2);
  const int numCellSides    = cellType->getFaceCount();
  const int numCellVertices = cellType->getNodeCount();
  const int numCellNodes    = cellBasis->getCardinality();
  const int numCellQPs      = cellCubature->getNumPoints();

  dl = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim,vecDimFO));
  dl_full = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim, neq));

  int numSurfaceSideVertices = -1;
  int numSurfaceSideNodes    = -1;
  int numSurfaceSideQPs      = -1;
  int numBasalSideVertices   = -1;
  int numBasalSideNodes      = -1;
  int numBasalSideQPs        = -1;
  int numLateralSideVertices = -1;
  int numLateralSideNodes    = -1;
  int numLateralSideQPs      = -1;

  int sideDim = numDim-1;

  if (basalSideName!="INVALID")
  {
    TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(basalSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                "Error! Either 'Basal Side Name' is wrong or something went wrong while building the side mesh specs.\n");
    const Albany::MeshSpecsStruct& basalMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(basalSideName)[0];

    // Building also basal side structures
    const CellTopologyData * const side_top = &basalMeshSpecs.ctd;
    basalSideBasis = Albany::getIntrepid2Basis(*side_top);
    basalSideType = rcp(new shards::CellTopology (side_top));

    basalEBName   = basalMeshSpecs.ebName;
    basalCubature = cubFactory.create<PHX::Device, RealType, RealType>(*basalSideType, basalMeshSpecs.cubatureDegree);

    numBasalSideVertices = basalSideType->getNodeCount();
    numBasalSideNodes    = basalSideBasis->getCardinality();
    numBasalSideQPs      = basalCubature->getNumPoints();

    dl_basal = rcp(new Albany::Layouts(worksetSize,numBasalSideVertices,numBasalSideNodes,
                                       numBasalSideQPs,sideDim,numDim,numCellSides,neq));
    dl_full->side_layouts[basalSideName] = dl_basal;
  }

  if (surfaceSideName!="INVALID")
  {
    TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(surfaceSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                  "Error! Either 'Surface Side Name' is wrong or something went wrong while building the side mesh specs.\n");

    const Albany::MeshSpecsStruct& surfaceMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(surfaceSideName)[0];

    // Building also surface side structures
    const CellTopologyData * const side_top = &surfaceMeshSpecs.ctd;
    surfaceSideBasis = Albany::getIntrepid2Basis(*side_top);
    surfaceSideType = rcp(new shards::CellTopology (side_top));

    surfaceEBName   = surfaceMeshSpecs.ebName;
    surfaceCubature = cubFactory.create<PHX::Device, RealType, RealType>(*surfaceSideType, surfaceMeshSpecs.cubatureDegree);

    numSurfaceSideVertices = surfaceSideType->getNodeCount();
    numSurfaceSideNodes    = surfaceSideBasis->getCardinality();
    numSurfaceSideQPs      = surfaceCubature->getNumPoints();

    dl_surface = rcp(new Albany::Layouts(worksetSize,numSurfaceSideVertices,numSurfaceSideNodes,
                                         numSurfaceSideQPs,sideDim,numDim,numCellSides,vecDimFO));
    dl->side_layouts[surfaceSideName] = dl_surface;
  }

  if (lateralSideName!="INVALID") {
    // For lateral bc, we want to give the option of not creating a whole discretization, since
    // it would just be needed to setup the lateral bc. If the user does not specify a lateral
    // discretization, we build the information we need ourself. The user can specify the quadrature
    // degree for the lateral bc in the 'LandIce Lateral BC' sublist. This would also be used to
    // override the specs of the lateral discretization (if present).

    const CellTopologyData* side_top = nullptr;
    int lateralCubDegree = -1;

    auto ss_ms = meshSpecs[0]->sideSetMeshSpecs;
    auto it = ss_ms.find(lateralSideName);
    if (it!=ss_ms.end()) {
      // The user specified a lateral side discretization. Just get the topology from the lateral side mesh specs
      side_top = &it->second[0]->ctd;
      lateralCubDegree = it->second[0]->cubatureDegree;
    } else {
      // The user did not specify a lateral side discretization. We need to create a topology from
      // the cell topology. We need to check what's the cell topology: if tetra/tria/quad/hexa, then
      // all sides have the same topology, so any side of the cell topology works. But if we have wedges,
      // then lateral side is the quadrilateral. As of this writing, the quad sides of a wedge are the first
      // three. However, for safety, we loop through all sides and check their node count, and pick the first
      // side with 4 nodes.
      std::string cell_top_name (cell_top->base->name);
      if (cell_top_name=="Wedge_6") {
        for (int i=0; i<cellType->getSubcellCount(sideDim); ++i) {
          std::string tmp (cellType->getCellTopologyData(sideDim,i)->base->name);
          if (tmp=="Quadrilateral_4") {
            // Found the lateral side. Get topology and break
            side_top = cellType->getCellTopologyData(sideDim,i);
            break;
          }
        }
      } else {
        // All sides have the same topology. Just pick the first
        side_top = cellType->getCellTopologyData(sideDim,0);
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION (side_top==nullptr, std::runtime_error, "Error! Something went wrong while detecting lateral side topology.\n");

    if (params->sublist("LandIce Lateral BC").isParameter("Cubature Degree")) {
      lateralCubDegree = params->sublist("LandIce Lateral BC").get<int>("Cubature Degree");
    }
    TEUCHOS_TEST_FOR_EXCEPTION (lateralCubDegree<0, std::runtime_error, "Error! Missing cubature degree information on side '" << lateralSideName << ".\n");

    lateralSideBasis = Albany::getIntrepid2Basis(*side_top);
    lateralSideType = rcp(new shards::CellTopology (side_top));

    lateralCubature = cubFactory.create<PHX::Device, RealType, RealType>(*lateralSideType, lateralCubDegree);

    numLateralSideVertices = lateralSideType->getNodeCount();
    numLateralSideNodes    = lateralSideBasis->getCardinality();
    numLateralSideQPs      = lateralCubature->getNumPoints();

    dl_lateral = rcp(new Albany::Layouts(worksetSize,numLateralSideVertices,numLateralSideNodes,
                                         numLateralSideQPs,sideDim,numDim,numCellSides,neq));
    dl_full->side_layouts[lateralSideName] = dl_lateral;
  }

//#ifdef OUTPUT_TO_SCREEN
  *out << "Field Dimensions: \n"
       << "  Workset             = " << worksetSize << "\n"
       << "  Vertices            = " << numCellVertices << "\n"
       << "  CellNodes           = " << numCellNodes << "\n"
       << "  CellQuadPts         = " << numCellQPs << "\n"
       << "  Dim                 = " << numDim << "\n"
       << "  VecDim              = " << neq << "\n"
       << "  VecDimFO            = " << vecDimFO << "\n"
       << "  BasalSideVertices   = " << numBasalSideVertices << "\n"
       << "  BasalSideNodes      = " << numBasalSideNodes << "\n"
       << "  BasalSideQuadPts    = " << numBasalSideQPs << "\n"
       << "  SurfaceSideVertices = " << numSurfaceSideVertices << "\n"
       << "  SurfaceSideNodes    = " << numSurfaceSideNodes << "\n"
       << "  SurfaceSideQuadPts  = " << numSurfaceSideQPs << "\n"
       << "  LateralSideVertices = " << numLateralSideVertices << "\n"
       << "  LateralSideNodes    = " << numLateralSideNodes << "\n"
       << "  LateralSideQuadPts  = " << numLateralSideQPs << std::endl;
//#endif

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
   std::vector<std::string> condNames(6); //(dCdx, dCdy, dCdz), dCdn, basal, P, lateral, basal_scalar_field
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
   condNames[2] = "basal";
   condNames[3] = "P";
   condNames[4] = "lateral";
   condNames[5] = "basal_scalar_field";

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
  validPL->sublist("LandIce Basal Friction Coefficient", false, "Parameters needed to compute the basal friction coefficient");
  validPL->sublist("Parameter Fields", false, "Parameter Fields to be registered");
  validPL->set<std::string> ("Basal Side Name", "", "Name of the basal side set");
  validPL->set<std::string> ("Surface Side Name", "", "Name of the surface side set");
  validPL->set<std::string> ("Lateral Side Name", "", "Name of the lateral side set");
  validPL->sublist("LandIce Lateral BC", false, "Specify parameters to use for the lateral BC");
  validPL->set<Teuchos::Array<std::string> > ("Required Fields", Teuchos::Array<std::string>(), "");
  validPL->set<Teuchos::Array<std::string> > ("Required Basal Fields", Teuchos::Array<std::string>(), "");
  validPL->set<Teuchos::Array<std::string> > ("Required Surface Fields", Teuchos::Array<std::string>(), "");
  validPL->set<int> ("Layered Data Length", 0, "Number of layers in input layered data files.");

  return validPL;
}

