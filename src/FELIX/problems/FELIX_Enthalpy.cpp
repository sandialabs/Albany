/*
 * FELIX_Enthalpy.cpp
 *
 *  Created on: May 11, 2016
 *      Author: abarone
 */
#include "FELIX_Enthalpy.hpp"

#include "Intrepid2_FieldContainer.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include <string>

FELIX::Enthalpy::
Enthalpy(const Teuchos::RCP<Teuchos::ParameterList>& params_,
		 const Teuchos::RCP<ParamLib>& paramLib_,
		 const int numDim_): Albany::AbstractProblem(params_, paramLib_, numDim_), numDim(numDim_)
{
	neq = 1;
	basalSideName = params->isParameter("Basal Side Name") ? params->get<std::string>("Basal Side Name") : "INVALID";
	basalEBName = "INVALID";
	Teuchos::ParameterList SUPG_list = params->get<Teuchos::ParameterList>("SUPG Settings");
	haveSUPG = SUPG_list.get("Have SUPG Stabilization",false);
	needsDiss = params->get<bool> ("Needs Dissipation",true);
	needsBasFric = params->get<bool> ("Needs Basal Friction",true);
	isGeoFluxConst = params->get<bool> ("Constant Geotermal Flux",true);

	TEUCHOS_TEST_FOR_EXCEPTION (basalSideName=="INVALID", std::logic_error, "Error! In order to specify basal requirements, you must also specify a valid 'Basal Side Name'.\n");
    // Need to allocate a fields in basal mesh database
    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Basal Fields");
    std::cout << req[0] << std::endl;
    this->ss_requirements[basalSideName].reserve(req.size()); // Note: this is not for performance, but to guarantee

    for (int i(0); i<req.size(); ++i)                         //       that ss_requirements.at(basalSideName) does not
    	this->ss_requirements[basalSideName].push_back(req[i]); //       throw, even if it's empty...
}

FELIX::Enthalpy::
~Enthalpy()
{
}

void FELIX::Enthalpy::
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
	  const int numCellSides = cellType->getFaceCount();

	  Intrepid2::DefaultCubatureFactory<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > cubFactory;
	  cellCubature = cubFactory.create(*cellType, meshSpecs[0]->cubatureDegree);

	  const int numQPtsCell = cellCubature->getNumPoints();
	  const int numVertices = cellType->getNodeCount();
	  const int velDim = 2;
	  const int vecDim = 2;

	   *out << "Field Dimensions: Workset=" << worksetSize
	        << ", Vertices= " << numVertices
	        << ", Nodes= " << numNodes
	        << ", QuadPts= " << numQPtsCell
	        << ", Dim= " << numDim << std::endl;

	  // Using the utility for the common evaluators
	  dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPtsCell,numDim,velDim));

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

	      dl_basal = rcp(new Albany::Layouts(worksetSize,numBasalSideVertices,numBasalSideNodes,numBasalSideQPs,numDim-1,numDim,numCellSides,vecDim));
		  dl->side_layouts[basalSideName] = dl_basal;
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
	  constructDirichletEvaluators(*meshSpecs[0]);

	  //if(meshSpecs[0]->ssNames.size() > 0) // Build a sideset evaluator if sidesets are present
		  //constructNeumannEvaluators(meshSpecs[0]);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
FELIX::Enthalpy::buildEvaluators( PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
								  const Albany::MeshSpecsStruct& meshSpecs,
								  Albany::StateManager& stateMgr,
								  Albany::FieldManagerChoice fmchoice,
								  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<Enthalpy> op(*this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void FELIX::Enthalpy::
constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);

   std::stringstream s;
   s << "Enth";
   dirichletNames[0] = s.str();

   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames, this->params, this->paramLib);
   offsets_ = dirUtils.getOffsets();
}

void FELIX::Enthalpy::
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

	std::vector<std::string> neumannNames(neq); 	//neq + 1
	Teuchos::Array<Teuchos::Array<int> > offsets;
	offsets.resize(neq);	//neq + 1

	neumannNames[0] = "Enth";
	offsets[0].resize(1);
	offsets[0][0] = 0;
	offsets[neq].resize(neq);
	offsets[neq][0] = 0;

    neumannNames[neq] = "all";

    // Construct BC evaluators for all possible names of conditions
    // Should only specify flux vector components (dCdx, dCdy, dCdz), or dCdn, not both
    std::vector<std::string> condNames(6); //(dCdx, dCdy, dCdz), dCdn, basal, P, lateral, basal_scalar_field
    Teuchos::ArrayRCP<std::string> dof_names(1);
    dof_names[0] = "Enthalpy";

    // Note that sidesets are only supported for two and 3D currently
    if(numDim == 2)
  	   condNames[0] = "(dFluxdx, dFluxdy)";
    else if(numDim == 3)
	   condNames[0] = "(dFluxdx, dFluxdy, dFluxdz)";
	else
	   TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

	condNames[1] = "dFluxdn";
	condNames[2] = "basal";
	condNames[3] = "P";
	condNames[4] = "lateral";
	condNames[5] = "basal_scalar_field";

	nfm.resize(1); // FELIX problem only has one element block

	nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, true, 0,
	                                        condNames, offsets, dl, this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
FELIX::Enthalpy::getValidProblemParameters() const
{
	Teuchos::RCP<Teuchos::ParameterList> validPL = this->getGenericProblemParams("ValidEnthalpyParams");
	validPL->sublist("Options", false, "");
	validPL->sublist("SUPG Settings", false, "");
	validPL->sublist("FELIX Viscosity", false, "");
	validPL->sublist("Stereographic Map", false, "");
	validPL->set<std::string> ("Basal Side Name", "", "Name of the basal side set");
	validPL->set<Teuchos::Array<std::string> > ("Required Basal Fields", Teuchos::Array<std::string>(), "");
	validPL->set<bool> ("Needs Dissipation", true, "Boolean describing whether we take into account the heat generated by dissipation");
	validPL->set<bool> ("Needs Basal Friction", true, "Boolean describing whether we take into account the heat generated by basal friction");
	validPL->set<bool> ("Constant Geotermal Flux", true, "Boolean describing whether the geotermal flux is constant");

/*
	validPL->set<bool> ("Extruded Column Coupled in 2D Response", false, "Boolean describing whether the extruded column is coupled in 2D response");
	validPL->set<int> ("Layered Data Length", 0, "Number of layers in input layered data files.");
	validPL->set<int> ("importCellTemperatureFromMesh", 0, "");
	validPL->set<Teuchos::Array<std::string> > ("Required Fields", Teuchos::Array<std::string>(), "");
	validPL->set<Teuchos::Array<std::string> > ("Required Surface Fields", Teuchos::Array<std::string>(), "");
	validPL->set<std::string> ("Surface Side Name", "", "Name of the surface side set");
	validPL->sublist("FELIX Basal Friction Coefficient", false, "Parameters needed to compute the basal friction coefficient");
	validPL->sublist("FELIX Surface Gradient", false, "");
	validPL->sublist("Equation Set", false, "");
	validPL->sublist("Body Force", false, "");
	validPL->sublist("FELIX Physical Parameters", false, "");
	validPL->sublist("FELIX Noise", false, "");
	validPL->sublist("Parameter Fields", false, "Parameter Fields to be registered");
*/
	return validPL;
}
