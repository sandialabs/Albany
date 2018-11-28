//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_Hydrology.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include <string>

namespace LandIce {

Hydrology::Hydrology (const Teuchos::RCP<Teuchos::ParameterList>& problemParams_,
                      const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                      const Teuchos::RCP<ParamLib>& paramLib,
                      const int numDimensions) :
  Albany::AbstractProblem (problemParams_, paramLib,1),
  numDim (numDimensions),
  discParams(discParams_),
  use_sdbcs_(false)
{
  TEUCHOS_TEST_FOR_EXCEPTION (numDim!=1 && numDim!=2,std::logic_error,"Problem supports only 1D and 2D");

  eliminate_h = params->sublist("LandIce Hydrology").get<bool>("Eliminate Water Thickness", false);
  has_h_till  = params->sublist("LandIce Hydrology").get<double>("Maximum Till Water Storage",0.0) > 0.0;
  has_p_dot   = params->sublist("LandIce Hydrology").get<double>("Englacial Porosity",0.0) > 0.0;

  std::string sol_method = params->get<std::string>("Solution Method");
  if (sol_method=="Transient" || sol_method=="Transient Tempus") {
    unsteady = true;
  } else {
    unsteady = false;
  }

  TEUCHOS_TEST_FOR_EXCEPTION (eliminate_h && unsteady, std::logic_error,
                              "Error! Water Thickness can be eliminated only in the steady case.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (has_h_till && !unsteady, std::logic_error,
                              "Error! Till Water Storage equation only makes sense in the unsteady case.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (has_p_dot && !unsteady, std::logic_error,
                              "Error! Englacial porosity model only makes sense in the unsteady case.\n");

  // Need to allocate a fields in mesh database
  if (params->isParameter("Required Fields")) {
    // Need to allocate a fields in mesh database
    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Fields");
    for (int i(0); i<req.size(); ++i) {
      this->requirements.push_back(req[i]);
    }
  }

  // Some fields that we know are FOR SURE required (if not added already)
  if (std::find(this->requirements.begin(),this->requirements.end(),surface_height_name)==this->requirements.end()) {
    this->requirements.push_back(surface_height_name);
  }
  if (std::find(this->requirements.begin(),this->requirements.end(),ice_thickness_name)==this->requirements.end()) {
    this->requirements.push_back(ice_thickness_name);
  }

  // Set the num PDEs depending on the problem specs
  if (eliminate_h) {
    this->setNumEquations(1);
  } else if (has_h_till) {
    this->setNumEquations(3);
  } else {
    this->setNumEquations(2);
  }
  dofs_names.resize(neq);
  resid_names.resize(neq);

  // We always solve for the water pressure
  dofs_names[0] = water_pressure_name;
  resid_names[0] = "Residual Mass Eqn";

  if (!eliminate_h) {
    dofs_names[1] = water_thickness_name;
    resid_names[1] = "Residual Cavities Eqn";
  }

  if (neq>2) {
    dofs_names[2] = till_water_storage_name;
    resid_names[2] = "Residual Till Storage Eqn";
  }

  if (unsteady) {
    // Prognositc dofs appear under time derivative, while diagnostic dofs do not.
    // In math terms, progrnostic dofs need an initial condition, while diagnostic dofs do not.
    if (has_p_dot) {
      diagnostic_dofs_names.resize(0);
      prognostic_dofs_names = dofs_names;
      prognostic_dofs_names_dot.resize(neq);
      prognostic_dofs_names_dot[0] = water_pressure_dot_name;
      prognostic_dofs_names_dot[1] = water_thickness_dot_name;
      if (has_h_till) {
        prognostic_dofs_names_dot[2] = till_water_storage_dot_name;
      }
    } else {
      diagnostic_dofs_names.resize(1);
      prognostic_dofs_names.resize(neq-1);
      prognostic_dofs_names_dot.resize(neq-1);

      diagnostic_dofs_names[0] = water_pressure_name;
      prognostic_dofs_names[0] = water_thickness_name;
      prognostic_dofs_names_dot[0] = water_thickness_dot_name;

      if (has_h_till) {
        prognostic_dofs_names[1] = till_water_storage_name;
        prognostic_dofs_names_dot[1] = till_water_storage_dot_name;
      }
    }
  }
}

void Hydrology::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                                     Albany::StateManager& stateMgr)
{
  // Building cell basis and cubature
  const CellTopologyData * const cell_top = &meshSpecs[0]->ctd;
  intrepidBasis = Albany::getIntrepid2Basis(*cell_top);
  cellType = Teuchos::rcp(new shards::CellTopology (cell_top));

  Intrepid2::DefaultCubatureFactory cubFactory;
  cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs[0]->cubatureDegree);

  elementBlockName = meshSpecs[0]->ebName;

  const int worksetSize     = meshSpecs[0]->worksetSize;
  const int numCellVertices = cellType->getNodeCount();
  const int numCellNodes    = intrepidBasis->getCardinality();
  const int numCellQPs      = cubature->getNumPoints();

  dl = Teuchos::rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim));

  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM,Teuchos::null);

  if(meshSpecs[0]->nsNames.size() > 0) {
    // Build a nodeset evaluator if nodesets are present
    constructDirichletEvaluators(*meshSpecs[0]);
  }
  if(meshSpecs[0]->ssNames.size() > 0) {
    // Build a sideset evaluator if sidesets are present
     constructNeumannEvaluators(meshSpecs[0]);
  }
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
Hydrology::buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                   const Albany::MeshSpecsStruct& meshSpecs,
                                   Albany::StateManager& stateMgr,
                                   Albany::FieldManagerChoice fmchoice,
                                   const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<Hydrology> op(*this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);

  return *op.tags;
}

void Hydrology::constructDirichletEvaluators (const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  dirichletNames[0] = dofs_names[0];
  if (!eliminate_h) {
    dirichletNames[1] = dofs_names[1];
  }

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames, this->params, this->paramLib);
  offsets_ = dirUtils.getOffsets();
  nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

void Hydrology::constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
  // Note: we only enter this function if sidesets are defined in the mesh file
  // i.e. meshSpecs.ssNames.size() > 0

  Albany::BCUtils<Albany::NeumannTraits> nbcUtils;

  // Check to make sure that Neumann BCs are given in the input file
  if (!nbcUtils.haveBCSpecified(this->params)) {
     return;
  }

  // Construct BC evaluators for all side sets and names
  // Note that the string index sets up the equation offset, so ordering is important

  std::vector<std::string> neumannNames(neq + 1);
  Teuchos::Array<Teuchos::Array<int> > offsets;
  offsets.resize(neq);

  neumannNames[0] = "Water Pressure";
  if (!eliminate_h) {
    neumannNames[1] = "Water Thickness";
  }
  neumannNames[neq] = "all";

  offsets[0].resize(1);
  offsets[0][0] = 0;
  if (!eliminate_h) {
    offsets[1].resize(1);
    offsets[1][0] = 1;
  }

  // Construct BC evaluators for all possible names of conditions
  std::vector<std::string> condNames(1);
  condNames[0] = "neumann";

  nfm.resize(1); // LandIce problem only has one element block

  nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, neumannNames, dofs_names, false, 0,
                                          condNames, offsets, dl,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
Hydrology::getValidProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getGenericProblemParams("ValidHydrologyProblemParams");

  validPL->sublist("LandIce Hydrology", false, "");
  validPL->sublist("LandIce Field Norm", false, "");
  validPL->sublist("LandIce Viscosity", false, "");
  validPL->sublist("LandIce Physical Parameters", false, "");
  validPL->sublist("LandIce Basal Friction Coefficient", false, "Parameters needed to compute the basal friction coefficient");

  return validPL;
}

constexpr char Hydrology::water_pressure_name[]                   ;  //= "water_pressure";
constexpr char Hydrology::water_thickness_name[]                  ;  //= "water_thickness";
constexpr char Hydrology::till_water_storage_name[]               ;  //= "till_water_storage";

constexpr char Hydrology::water_pressure_dot_name[]               ;  //= "water_pressure_dot";
constexpr char Hydrology::water_thickness_dot_name[]              ;  //= "water_thickness_dot";
constexpr char Hydrology::till_water_storage_dot_name[]           ;  //= "till_water_storage_dot";

constexpr char Hydrology::hydraulic_potential_name[]              ;  //= "hydraulic_potential";
constexpr char Hydrology::hydraulic_potential_gradient_name[]     ;  //= "hydraulic_potential Gradient";
constexpr char Hydrology::hydraulic_potential_gradient_norm_name[];  //= "hydraulic_potential Gradient Norm";
constexpr char Hydrology::ice_softness_name[]                     ;  //= "ice_softness";
constexpr char Hydrology::ice_overburden_name[]                   ;  //= "ice_overburden";
constexpr char Hydrology::effective_pressure_name[]               ;  //= "effective_pressure";
constexpr char Hydrology::ice_temperature_name[]                  ;  //= "ice_temperature";
constexpr char Hydrology::ice_thickness_name[]                    ;  //= "ice_thickness";
constexpr char Hydrology::surface_height_name[]                   ;  //= "surface_height";
constexpr char Hydrology::beta_name[]                             ;  //= "beta";
constexpr char Hydrology::melting_rate_name[]                     ;  //= "melting_rate";
constexpr char Hydrology::surface_water_input_name[]              ;  //= "surface_water_input";
constexpr char Hydrology::surface_mass_balance_name[]             ;  //= "surface_mass_balance";
constexpr char Hydrology::geothermal_flux_name[]                  ;  //= "geothermal_flux";
constexpr char Hydrology::water_discharge_name[]                  ;  //= "water_discharge";
constexpr char Hydrology::sliding_velocity_name[]                 ;  //= "sliding_velocity";
constexpr char Hydrology::basal_velocity_name[]                   ;  //= "basal_velocity";
constexpr char Hydrology::basal_grav_water_potential_name[]  ;  //= "basal_gravitational_water_potential";

} // namespace LandIce
