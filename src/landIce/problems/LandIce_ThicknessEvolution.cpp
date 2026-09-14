/*
 * LandIce_Enthalpy.cpp
 *
 *  Created on: May 11, 2016
 *      Author: A. Barone, M. Perego
 */
#include "LandIce_ThicknessEvolution.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include <string>

LandIce::ThicknessEvolution::
ThicknessEvolution(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                 const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
		 const Teuchos::RCP<ParamLib>& paramLib_,
		 const int numDim_): Albany::AbstractProblem(params_, paramLib_, numDim_), 
  numDim(numDim_), discParams(discParams_), 
  use_sdbcs_(false)
{
  this->setNumEquations(1);
    std::string sol_method = params->get<std::string>("Solution Method");
  if (sol_method=="Transient") {
    unsteady = true;
  } else {
    unsteady = false;
  }

  vecDim = 2;
  surface_height_name = params->sublist("Variables Names").get<std::string>("Surface Height Name","surface_height");
  ice_thickness_name  = params->sublist("Variables Names").get<std::string>("Ice Thickness Name" ,"ice_thickness");

  dof_names.resize(1);
  dof_names[0] = ice_thickness_name + "_change";
  if(unsteady) {
    dof_names_dot.resize(1);
    dof_names_dot[0] = ice_thickness_name + "_dot";
  }

  resid_names.resize(1);
  resid_names[0] = dof_names[0] + " Residual";

  scatter_names.resize(1);
  scatter_names[0] = "Scatter " + resid_names[0];

  dof_offsets.resize(1,0);

  // We have two values for ice_thickness: the initial one, and the updated one.
  initial_ice_thickness_name = ice_thickness_name;

  if(unsteady) {
    surface_height_name += "_computed";
    ice_thickness_name += "_computed";
  }

  Teuchos::ParameterList& physics_list = params->sublist("LandIce Physical Parameters");
  if (!physics_list.isParameter("Seconds per Year")) {
    physics_list.set("Seconds per Year", 3.1536e7);
  }

  // Compute Rigid Body Modes for near null space to pass to preconditioners
  const bool computeConstantModes = true;
  rigidBodyModes->setParameters(neq, computeConstantModes);
}

LandIce::ThicknessEvolution::
~ThicknessEvolution()
{
}

void LandIce::ThicknessEvolution::
buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs, Albany::StateManager& stateMgr)
{
	  using Teuchos::RCP;
	  using Teuchos::rcp;
	  using Teuchos::ParameterList;
	  using PHX::DataLayout;
	  using PHX::MDALayout;
	  using std::vector;
	  using std::string;
	  using std::map;
	  using PHAL::AlbanyTraits;

	  const CellTopologyData* const elem_top = &meshSpecs[0]->ctd;

	  cellBasis = Albany::getIntrepid2Basis(*elem_top);
	  cellType = rcp(new shards::CellTopology (elem_top));

	  const int numNodes = cellBasis->getCardinality();
	  const int worksetSize = meshSpecs[0]->worksetSize;
    const int cubDegree = this->params->get("Cubature Degree", 4);
	  Intrepid2::DefaultCubatureFactory cubFactory;
	  cellCubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, cubDegree);

	  const int numQPtsCell = cellCubature->getNumPoints();
	  const int numVertices = cellType->getNodeCount();
	  const int velDim = 2;

	   *out << "Field Dimensions: Workset=" << worksetSize
	        << ", Vertices= " << numVertices
	        << ", Nodes= " << numNodes
	        << ", QuadPts= " << numQPtsCell
	        << ", Dim= " << numDim << std::endl;

	  // Using the utility for the common evaluators
	  dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPtsCell,numDim,velDim));

	  // Volume Fields
	  Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
	  unsigned int num_fields = req_fields_info.get<int>("Number Of Fields",0);

	  std::string fieldType, fieldUsage, meshPart;
	  for (unsigned int ifield=0; ifield<num_fields; ++ifield) {
	    Teuchos::ParameterList& thisFieldList = req_fields_info.sublist(util::strint("Field", ifield));

	    // Get current state specs
	    volumeFields.insert(thisFieldList.get<std::string>("Field Name"));
	  }

	  int numBasalSideVertices   = -1;
	  int numBasalSideNodes      = -1;
	  int numBasalSideQPs        = -1;

    std::string lateralSideName = "boundary_side_set_3";
	  if (lateralSideName!="INVALID")
	  {
		  TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(lateralSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
	                                  "Error! Either 'Lateral Side Name' is wrong or something went wrong while building the side mesh specs.\n");
		  const Albany::MeshSpecsStruct& basalMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(lateralSideName)[0];

		  // Building also basal side structures
		  const CellTopologyData * const side_lateral = &basalMeshSpecs.ctd;
		  sideBasis = Albany::getIntrepid2Basis(*side_lateral);
		  sideType = rcp(new shards::CellTopology (side_lateral));
      int sideCubDegree = params->get("Side Cubature Degree", 4);
      sideCubature = cubFactory.create<PHX::Device, RealType, RealType>(*sideType, sideCubDegree);

      auto numSideVertices = sideType->getNodeCount();
      auto numSideNodes    = sideBasis->getCardinality();
      auto numSideQPs      = sideCubature->getNumPoints();

		  *out << "Sideset Field Dimensions: Workset=" << basalMeshSpecs.worksetSize
			   << ", SideVertices= " << numBasalSideVertices
	           << ", SideNodes= " << numBasalSideNodes
	           << ", SideQuadPts= " << numBasalSideQPs << std::endl;

      dl_side = rcp(new Albany::Layouts(numSideVertices,numSideNodes, numSideQPs,numDim-1,numDim,1,lateralSideName));
      dl->side_layouts[lateralSideName] = dl_side;
	  }


#ifdef OUTPUT_TO_SCREEN
  *out << "Field Dimensions: \n"
       << "  Workset             = " << worksetSize << "\n"
       << "  Vertices            = " << numCellVertices << "\n"
       << "  CellNodes           = " << numCellNodes << "\n"
       << "  CellQuadPts         = " << numCellQPs << "\n"
       << "  Dim                 = " << numDim << "\n"
       << "  VecDim              = " << vecDim << "\n"
       << "  BasalSideVertices   = " << numBasalSideVertices << "\n"
       << "  BasalSideNodes      = " << numBasalSideNodes << "\n"
       << "  BasalSideQuadPts    = " << numBasalQPs << std::endl;
#endif


      /* Construct All Phalanx Evaluators */
	  elementBlockName = meshSpecs[0]->ebName;
	  fm.resize(1);
	  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
	  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM, Teuchos::null);
	  buildFields(*fm[0]);
	  constructDirichletEvaluators(*meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
LandIce::ThicknessEvolution::buildEvaluators( PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
								  const Albany::MeshSpecsStruct& meshSpecs,
								  Albany::StateManager& stateMgr,
								  Albany::FieldManagerChoice fmchoice,
								  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<ThicknessEvolution> op(*this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
LandIce::ThicknessEvolution::buildFields(PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  // Allocate memory for unmanaged fields
  fieldUtils = Teuchos::rcp(new Albany::FieldUtils(fm0, dl));
  fieldUtils->allocateComputeBasisFunctionsFields();

  // Call constructFields<EvalT>() for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructFieldsOp<ThicknessEvolution> op(*this, fm0);
  Sacado::mpl::for_each_no_kokkos<PHAL::AlbanyTraits::BEvalTypes> fe(op);
}

void LandIce::ThicknessEvolution::
constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
   dirichletNames[0] = "H";

   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames, this->params, this->paramLib);
   use_sdbcs_ = dirUtils.useSDBCs(); 
   offsets_ = dirUtils.getOffsets();
   nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

void LandIce::ThicknessEvolution::
constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
	// Note: we only enter this function if sidesets are defined in the mesh file
	// i.e. meshSpecs.ssNames.size() > 0

	Albany::BCUtils<Albany::NeumannTraits> nbcUtils;

	// Check to make sure that Neumann BCs are given in the input file

	if(!nbcUtils.haveBCSpecified(this->params))
		return;

	// Construct BC evaluators for all side sets and names
	// Note that the string index sets up the equation offset, so ordering is important

	std::vector<std::string> neumannNames(neq);
	Teuchos::Array<Teuchos::Array<int> > offsets;
	offsets.resize(neq);

	neumannNames[0] = "H";
	offsets[0].resize(1);
	offsets[0][0] = 0;

    // Construct BC evaluators for all possible names of conditions
    // Should only specify flux vector components (dCdx, dCdy, dCdz), or dCdn, not both
    std::vector<std::string> condNames(2); //(dCdx, dCdy, dCdz), dCdn, basal, P, lateral, basal_scalar_field


  // Note that sidesets are only supported for two and 3D currently
  condNames[0] = "(dFluxdx, dFluxdy)";

	condNames[1] = "dFluxdn";

	nfm.resize(1); // LandIce problem only has one element block

	nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, true, 0,
	                                        condNames, offsets, dl, this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
LandIce::ThicknessEvolution::getValidProblemParameters() const
{
	Teuchos::RCP<Teuchos::ParameterList> validPL = this->getGenericProblemParams("ValidThicknessEvolParams");
	validPL->sublist("LandIce Physical Parameters", false, "");
  validPL->sublist("Variables Names", false, "");
  validPL->set<double>("Time Step", 1.0, "Time step for divergence flux ");
  validPL->set<Teuchos::RCP<double> >("Time Step Ptr", Teuchos::null, "Time step ptr for divergence flux ");
	validPL->set<int>("Cubature Degree", 4, "Cubature degree used on the basal side");
  validPL->set<int>("Lateral Cubature Degree", 4, "Cubature degree used on the lateral side");

	return validPL;
}
