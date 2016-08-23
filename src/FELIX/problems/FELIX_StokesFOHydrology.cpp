//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <string>

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_FancyOStream.hpp"

#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"

#include "FELIX_StokesFOHydrology.hpp"

FELIX::StokesFOHydrology::
StokesFOHydrology (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                   const Teuchos::RCP<ParamLib>& paramLib_,
                   const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_),
  numDim(numDim_)
{
  basalSideName   = params->get<std::string>("Basal Side Name");
  surfaceSideName = params->isParameter("Surface Side Name") ? params->get<std::string>("Surface Side Name") : "INVALID";
  basalEBName = "INVALID";
  surfaceEBName = "INVALID";

  // Need to allocate a fields in mesh database
  Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Fields");
  for (int i(0); i<req.size(); ++i)
    this->requirements.push_back(req[i]);

  // Need to allocate a fields in basal mesh database
  Teuchos::Array<std::string> breq = params->get<Teuchos::Array<std::string> > ("Required Basal Fields");
  this->ss_requirements[basalSideName].reserve(breq.size()); // Note: this is not for performance, but to guarantee
  for (int i(0); i<breq.size(); ++i)                         //       that ss_requirements.at(basalSideName) does not
    this->ss_requirements[basalSideName].push_back(breq[i]); //       throw, even if it's empty...

  if (params->isParameter("Required Surface Fields"))
  {
    TEUCHOS_TEST_FOR_EXCEPTION (surfaceSideName=="INVALID", std::logic_error, "Error! In order to specify surface requirements, you must also specify a valid 'Surface Side Name'.\n");

    Teuchos::Array<std::string> sreq = params->get<Teuchos::Array<std::string> > ("Required Surface Fields");
    this->ss_requirements[surfaceSideName].reserve(sreq.size()); // Note: same motivation as for the basal side
    for (int i(0); i<sreq.size(); ++i)
      this->ss_requirements[surfaceSideName].push_back(sreq[i]);
  }

  stokes_dof_names.resize(1);
  stokes_resid_names.resize(1);
  stokes_dof_names[0] = "Velocity";
  stokes_resid_names[0] = "Residual Stokes";
  stokes_neq = 2;

  has_h_equation = params->sublist("FELIX Hydrology").get<bool>("Use Water Thickness Equation",false);
  std::string sol_method = params->get<std::string>("Solution Method");
  if (sol_method=="Unsteady")
    unsteady = true;
  else
    unsteady = false;

  TEUCHOS_TEST_FOR_EXCEPTION (unsteady && !has_h_equation, std::logic_error,
                              "Error! Unsteady case require to use the water thickness equation.\n");

  if (has_h_equation)
  {
    hydro_neq = 2;

    hydro_dof_names.resize(hydro_neq);
    hydro_dof_names[0] = "Hydraulic Potential";
    hydro_dof_names[1] = "Water Thickness";

    if (unsteady)
    {
      hydro_dof_names_dot.resize(1);
      hydro_dof_names_dot[0] = "Water Thickness Dot";
    }

    hydro_resid_names.resize(hydro_neq);
    hydro_resid_names[0] = "Residual Hydrology Potential Eqn";
    hydro_resid_names[1] = "Residual Hydrology Thickness Eqp";
  }
  else
  {
    hydro_neq = 1;

    hydro_dof_names.resize(hydro_neq);
    hydro_dof_names[0] = "Hydraulic Potential";

    hydro_resid_names.resize(hydro_neq);
    hydro_resid_names[0] = "Residual Hydrology Potential Eqn";
  }

  // Set the number of eq of the problem
  this->neq = stokes_neq + hydro_neq;
  this->setNumEquations(neq);
  this->rigidBodyModes->setNumPDEs(neq);

  // Set the hydrology equations as side set equations on the basal side
  for (int eq=stokes_neq; eq<hydro_neq; ++eq)
    this->sideSetEquations[eq].push_back(basalSideName);
}

FELIX::StokesFOHydrology::
~StokesFOHydrology()
{
  // Nothing to be done here
}

void FELIX::StokesFOHydrology::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                                             Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

  // Building cell basis and cubature
  const CellTopologyData * const cell_top = &meshSpecs[0]->ctd;
  cellBasis = Albany::getIntrepid2Basis(*cell_top);
  cellType = rcp(new shards::CellTopology (cell_top));

  Intrepid2::DefaultCubatureFactory<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > cubFactory;
  cellCubature = cubFactory.create(*cellType, meshSpecs[0]->cubatureDegree);

  elementBlockName = meshSpecs[0]->ebName;

  TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(basalSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                              "Error! Either 'Basal Side Name' (" << basalSideName << ") is wrong or something went wrong while building the side mesh specs.\n");
  const Albany::MeshSpecsStruct& basalMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(basalSideName)[0];

  const int worksetSize     = meshSpecs[0]->worksetSize;
  const int vecDim          = stokes_neq;
  const int numCellSides    = cellType->getFaceCount();
  const int numCellVertices = cellType->getNodeCount();
  const int numCellNodes    = cellBasis->getCardinality();
  const int numCellQPs      = cellCubature->getNumPoints();

  dl = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim,vecDim));

  // Building also basal side structures
  const CellTopologyData * const basal_side_top = &basalMeshSpecs.ctd;
  basalSideBasis = Albany::getIntrepid2Basis(*basal_side_top);
  basalSideType = rcp(new shards::CellTopology (basal_side_top));

  basalEBName   = basalMeshSpecs.ebName;
  basalCubature = cubFactory.create(*basalSideType, basalMeshSpecs.cubatureDegree);

  int numBasalSideVertices = basalSideType->getNodeCount();
  int numBasalSideNodes    = basalSideBasis->getCardinality();
  int numBasalSideQPs      = basalCubature->getNumPoints();

  dl_basal = rcp(new Albany::Layouts(worksetSize,numBasalSideVertices,numBasalSideNodes,
                                     numBasalSideQPs,numDim-1,numDim,numCellSides,vecDim));
  dl->side_layouts[basalSideName] = dl_basal;

  int numSurfaceSideVertices = -1;
  int numSurfaceSideNodes    = -1;
  int numSurfaceSideQPs      = -1;

  if (surfaceSideName!="INVALID")
  {
    TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(surfaceSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                "Error! Either 'Surface Side Name' is wrong or something went wrong while building the side mesh specs.\n");

    const Albany::MeshSpecsStruct& surfaceMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(surfaceSideName)[0];

    // Building also surface side structures
    const CellTopologyData * const surface_side_top = &surfaceMeshSpecs.ctd;
    surfaceSideBasis = Albany::getIntrepid2Basis(*surface_side_top);
    surfaceSideType = rcp(new shards::CellTopology (surface_side_top));

    surfaceEBName   = surfaceMeshSpecs.ebName;
    surfaceCubature = cubFactory.create(*surfaceSideType, surfaceMeshSpecs.cubatureDegree);

    numSurfaceSideVertices = surfaceSideType->getNodeCount();
    numSurfaceSideNodes    = surfaceSideBasis->getCardinality();
    numSurfaceSideQPs      = surfaceCubature->getNumPoints();

    dl_surface = rcp(new Albany::Layouts(worksetSize,numSurfaceSideVertices,numSurfaceSideNodes,
                                         numSurfaceSideQPs,numDim-1,numDim,numCellSides,vecDim));
    dl->side_layouts[surfaceSideName] = dl_surface;
  }

#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  int commRank = Teuchos::GlobalMPISession::getRank();
  int commSize = Teuchos::GlobalMPISession::getNProc();
  out->setProcRankAndSize(commRank, commSize);
  out->setOutputToRootOnly(0);

  *out << "Field Dimensions: \n"
       << "  Workset             = " << worksetSize << "\n"
       << "  Vertices            = " << numCellVertices << "\n"
       << "  CellNodes           = " << numCellNodes << "\n"
       << "  CellQuadPts         = " << numCellQPs << "\n"
       << "  Dim                 = " << numDim << "\n"
       << "  VecDim              = " << vecDim << "\n"
       << "  BasalSideVertices   = " << numBasalSideVertices << "\n"
       << "  BasalSideNodes      = " << numBasalSideNodes << "\n"
       << "  BasalSideQuadPts    = " << numBasalQPs << "\n"
       << "  SurfaceSideVertices = " << numSurfaceSideVertices << "\n"
       << "  SurfaceSideNodes    = " << numSurfaceSideNodes << "\n"
       << "  SurfaceSideQuadPts  = " << numSurfaceQPs << std::endl;
#endif

  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM,Teuchos::null);

  if(meshSpecs[0]->nsNames.size() > 0) // Build a dirichlet field manager if nodesets are present
    constructDirichletEvaluators(*meshSpecs[0]);

  if(meshSpecs[0]->ssNames.size() > 0) // Build a neumann field manager if sidesets are present
     constructNeumannEvaluators(meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
FELIX::StokesFOHydrology::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<StokesFOHydrology> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
FELIX::StokesFOHydrology::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> stokes_dir_names(neq);
  for (int i=0; i<stokes_neq; i++) {
    std::stringstream s; s << "U" << i;
    stokes_dir_names[i] = s.str();
  }

  std::map<std::string,std::vector<std::string>> hydro_dir_names;
  hydro_dir_names[basalSideName].push_back("Hydrostatic Potential");
  if (hydro_neq>1)
    hydro_dir_names[basalSideName].push_back("Water Thickness");

  std::map<std::string,std::vector<std::string>> ss_nsNames;
  ss_nsNames[basalSideName] = meshSpecs.sideSetMeshSpecs.at(basalSideName)[0]->nsNames;

  std::map<std::string,std::vector<int>> ss_bcOffsets;
  ss_bcOffsets[basalSideName].push_back(stokes_neq);
  if (hydro_neq>1)
    ss_bcOffsets[basalSideName].push_back(stokes_neq+1);

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, stokes_dir_names, this->params, this->paramLib, neq);
}

// Neumann BCs
void FELIX::StokesFOHydrology::constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
  Albany::BCUtils<Albany::NeumannTraits> nbcUtils;

  // Check to make sure that Neumann BCs are given in the input file
  if(!nbcUtils.haveBCSpecified(this->params))
     return;

  // Construct BC evaluators for all side sets and names
  // Note that the string index sets up the equation offset, so ordering is important
  // Also, note that we only have neumann conditions for the ice. Hydrology can also
  // have neumann BC, but they are homogeneous (do-nothing).

  // Stokes BCs
  std::vector<std::string> stokes_neumann_names(stokes_neq + 1);
  Teuchos::Array<Teuchos::Array<int> > stokes_offsets;
  stokes_offsets.resize(stokes_neq + 1);

  stokes_neumann_names[0] = "U0";
  stokes_offsets[0].resize(1);
  stokes_offsets[0][0] = 0;
  stokes_offsets[stokes_neq].resize(stokes_neq);
  stokes_offsets[stokes_neq][0] = 0;

  if (neq>1)
  {
    stokes_neumann_names[1] = "U1";
    stokes_offsets[1].resize(1);
    stokes_offsets[1][0] = 1;
    stokes_offsets[stokes_neq][1] = 1;
  }

  stokes_neumann_names[stokes_neq] = "all";

  std::vector<std::string> stokes_cond_names(1);
  stokes_cond_names[0] = "lateral";

  nfm.resize(1); // FELIX problem only has one element block

  nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, stokes_neumann_names, stokes_dof_names, true, 0,
                                          stokes_cond_names, stokes_offsets, dl,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
FELIX::StokesFOHydrology::getValidProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidStokesFOHydrologyProblemParams");

  validPL->set<int> ("Layered Data Length", 0, "Number of layers in input layered data files.");
  validPL->set<Teuchos::Array<std::string> > ("Required Fields", Teuchos::Array<std::string>(), "");
  validPL->set<Teuchos::Array<std::string> > ("Required Basal Fields", Teuchos::Array<std::string>(), "");
  validPL->set<Teuchos::Array<std::string> > ("Required Surface Fields", Teuchos::Array<std::string>(), "");
  validPL->set<std::string> ("Basal Side Name", "", "Name of the basal side set");
  validPL->set<std::string> ("Surface Side Name", "", "Name of the surface side set");
  validPL->sublist("Stereographic Map", false, "");
  validPL->sublist("FELIX Hydrology", false, "");
  validPL->sublist("FELIX Viscosity", false, "");
  validPL->sublist("FELIX Basal Friction Coefficient", false, "Parameters needed to compute the basal friction coefficient");
  validPL->sublist("FELIX Surface Gradient", false, "");
  validPL->sublist("FELIX Field Norm", false, "");
  validPL->sublist("Body Force", false, "");
  validPL->sublist("FELIX Physical Parameters", false, "");
  validPL->sublist("FELIX Noise", false, "");
  validPL->sublist("Parameter Fields", false, "Parameter Fields to be registered");

  return validPL;
}
