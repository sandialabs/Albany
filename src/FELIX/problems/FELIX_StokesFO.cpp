//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <string>

#include "Intrepid2_FieldContainer.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_FancyOStream.hpp"

#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "FELIX_StokesFO.hpp"

FELIX::StokesFO::
StokesFO( const Teuchos::RCP<Teuchos::ParameterList>& params_,
          const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_) :
  Albany::AbstractProblem(params_, paramLib_, numDim_),
  numDim(numDim_),
  discParams(discParams_)
{
  //Set # of PDEs per node based on the Equation Set.
  //Equation Set is FELIX by default (2 dofs / node -- usual FELIX Stokes FO).
  std::string eqnSet = params_->sublist("Equation Set").get<std::string>("Type", "FELIX");
  if (eqnSet == "FELIX")
    neq = 2; //FELIX FO Stokes system is a system of 2 PDEs
  else if (eqnSet == "Poisson" || eqnSet == "FELIX X-Z")
    neq = 1; //1 PDE/node for Poisson or FELIX X-Z physics


  // Set the num PDEs for the null space object to pass to ML
  this->rigidBodyModes->setNumPDEs(neq);

  // the following function returns the problem information required for setting the rigid body modes (RBMs) for elasticity problems
  //written by IK, Feb. 2012
  //Check if we want to give ML RBMs (from parameterlist)
  int numRBMs = params_->get<int>("Number RBMs for ML", 0);
  bool setRBMs = false;
  if (numRBMs > 0) {
    setRBMs = true;
    int numScalar = 0;
    if (numRBMs == 2 || numRBMs == 3)
      rigidBodyModes->setParameters(neq, numDim, numScalar, numRBMs, setRBMs);
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"The specified number of RBMs "
                                     << numRBMs << " is not valid!  Valid values are 0, 2 and 3.");
  }

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
  basalEBName = "INVALID";
  surfaceEBName = "INVALID";
  sliding = params->isSublist("FELIX Basal Friction Coefficient");
  TEUCHOS_TEST_FOR_EXCEPTION (sliding && basalSideName=="INVALID", std::logic_error,
                              "Error! With sliding, you need to provide a valid 'Basal Side Name',\n" );

  if (params->isParameter("Required Basal Fields"))
  {
    TEUCHOS_TEST_FOR_EXCEPTION (basalSideName=="INVALID", std::logic_error, "Error! In order to specify basal requirements, you must also specify a valid 'Basal Side Name'.\n");

    // Need to allocate a fields in basal mesh database
    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Basal Fields");
    this->ss_requirements[basalSideName].reserve(req.size()); // Note: this is not for performance, but to guarantee
    for (int i(0); i<req.size(); ++i)                         //       that ss_requirements.at(basalSideName) does not
      this->ss_requirements[basalSideName].push_back(req[i]); //       throw, even if it's empty...
  }
  if (params->isParameter("Required Surface Fields"))
  {
    TEUCHOS_TEST_FOR_EXCEPTION (surfaceSideName=="INVALID", std::logic_error, "Error! In order to specify surface requirements, you must also specify a valid 'Surface Side Name'.\n");

    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Surface Fields");
    this->ss_requirements[surfaceSideName].reserve(req.size()); // Note: same motivation as for the basal side
    for (int i(0); i<req.size(); ++i)
      this->ss_requirements[surfaceSideName].push_back(req[i]);
  }
}

FELIX::StokesFO::
~StokesFO()
{
  // Nothing to be done here
}

void FELIX::StokesFO::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
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

  const int worksetSize     = meshSpecs[0]->worksetSize;
  const int vecDim          = 2;
  const int numCellSides    = cellType->getFaceCount();
  const int numCellVertices = cellType->getNodeCount();
  const int numCellNodes    = cellBasis->getCardinality();
  const int numCellQPs      = cellCubature->getNumPoints();

  dl = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim,vecDim));

  int numSurfaceSideVertices = -1;
  int numSurfaceSideNodes    = -1;
  int numSurfaceSideQPs      = -1;
  int numBasalSideVertices   = -1;
  int numBasalSideNodes      = -1;
  int numBasalSideQPs        = -1;

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
    basalCubature = cubFactory.create(*basalSideType, basalMeshSpecs.cubatureDegree);

    numBasalSideVertices = basalSideType->getNodeCount();
    numBasalSideNodes    = basalSideBasis->getCardinality();
    numBasalSideQPs      = basalCubature->getNumPoints();

    dl_basal = rcp(new Albany::Layouts(worksetSize,numBasalSideVertices,numBasalSideNodes,
                                       numBasalSideQPs,numDim-1,numDim,numCellSides,vecDim));
    dl->side_layouts[basalSideName] = dl_basal;
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
  constructDirichletEvaluators(*meshSpecs[0]);

  if(meshSpecs[0]->ssNames.size() > 0) // Build a sideset evaluator if sidesets are present
     constructNeumannEvaluators(meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
FELIX::StokesFO::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<StokesFO> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
FELIX::StokesFO::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
   for (int i=0; i<neq; i++) {
     std::stringstream s; s << "U" << i;
     dirichletNames[i] = s.str();
   }
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
   offsets_ = dirUtils.getOffsets();
}

// Neumann BCs
void FELIX::StokesFO::constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
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
   offsets[neq].resize(neq);
   offsets[neq][0] = 0;

   if (neq>1){
     neumannNames[1] = "U1";
     offsets[1].resize(1);
     offsets[1][0] = 1;
     offsets[neq][1] = 1;
   }

   if (neq>2){
     neumannNames[2] = "U2";
     offsets[2].resize(1);
     offsets[2][0] = 2;
     offsets[neq][2] = 2;
   }

   neumannNames[neq] = "all";

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dCdx, dCdy, dCdz), or dCdn, not both
   std::vector<std::string> condNames(6); //(dCdx, dCdy, dCdz), dCdn, basal, P, lateral, basal_scalar_field
   Teuchos::ArrayRCP<std::string> dof_names(1);
     dof_names[0] = "Velocity";

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

   nfm.resize(1); // FELIX problem only has one element block

   nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, true, 0,
                                           condNames, offsets, dl,
                                           this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
FELIX::StokesFO::getValidProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidStokesFOProblemParams");

  validPL->set<bool> ("Extruded Column Coupled in 2D Response", false, "Boolean describing whether the extruded column is coupled in 2D response");
  validPL->set<int> ("Layered Data Length", 0, "Number of layers in input layered data files.");
  validPL->set<int> ("importCellTemperatureFromMesh", 0, "");
  validPL->set<Teuchos::Array<std::string> > ("Required Fields", Teuchos::Array<std::string>(), "");
  validPL->set<Teuchos::Array<std::string> > ("Required Basal Fields", Teuchos::Array<std::string>(), "");
  validPL->set<Teuchos::Array<std::string> > ("Required Surface Fields", Teuchos::Array<std::string>(), "");
  validPL->set<std::string> ("Basal Side Name", "", "Name of the basal side set");
  validPL->set<std::string> ("Surface Side Name", "", "Name of the surface side set");
  validPL->sublist("Stereographic Map", false, "");
  validPL->sublist("FELIX Viscosity", false, "");
  validPL->sublist("FELIX Effective Pressure Surrogate", false, "Parameters needed to compute the effective pressure surrogate");
  validPL->sublist("FELIX Basal Friction Coefficient", false, "Parameters needed to compute the basal friction coefficient");
  validPL->sublist("FELIX Surface Gradient", false, "");
  validPL->sublist("Equation Set", false, "");
  validPL->sublist("Body Force", false, "");
  validPL->sublist("FELIX Field Norm", false, "");
  validPL->sublist("FELIX Physical Parameters", false, "");
  validPL->sublist("FELIX Noise", false, "");
  validPL->sublist("Parameter Fields", false, "Parameter Fields to be registered");
  validPL->set<bool>("Use Time Parameter", false, "Solely to use Solver Method = Continuation");

  return validPL;
}
