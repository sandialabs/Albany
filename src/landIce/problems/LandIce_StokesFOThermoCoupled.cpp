//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <string>

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_FancyOStream.hpp"

#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "LandIce_StokesFOThermoCoupled.hpp"

namespace LandIce
{

StokesFOThermoCoupled::
StokesFOThermoCoupled( const Teuchos::RCP<Teuchos::ParameterList>& params_,
                       const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                       const Teuchos::RCP<ParamLib>& paramLib_,
                       const int numDim_) :
  StokesFOBase(params_, discParams_, paramLib_, numDim_)
{
  fluxDivIsPartOfSolution = params->isSublist("LandIce Flux Divergence") &&
      params->sublist("LandIce Flux Divergence").get<bool>("Flux Divergence Is Part Of Solution");

  l2ProjectedBoundaryEquation = params->isSublist("LandIce L2 Projected Boundary Laplacian");

  // 2 eqns for Stokes FO + 1 eqn. for enthalpy + 1 eqn. for w. Optionally 1 eqn. for fluxDiv
  neq = unsigned(fluxDivIsPartOfSolution) + unsigned(l2ProjectedBoundaryEquation) + 4;
  this->setNumEquations(neq);

  //Teuchos::ParameterList SUPG_list = params->get<Teuchos::ParameterList>("SUPG Settings");
  //haveSUPG = SUPG_list.get("Have SUPG Stabilization",false);
  needsDiss = params->get<bool> ("Needs Dissipation",true);
  needsBasFric = params->get<bool> ("Needs Basal Friction",true);

  TEUCHOS_TEST_FOR_EXCEPTION (basalSideName=="INVALID", std::logic_error,
                              "Error! StokesFOThermoCoupled requires a valid basalSideName.\n");
  if (needsBasFric) {
    // We must have a BasalFriction landice bc on ss basalSideName
    bool found = false;
    for (auto it : this->landice_bcs[LandIceBC::BasalFriction]) {
      if (it->get<std::string>("Side Set Name")==basalSideName) {
        found = true;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(!found, std::logic_error, "Error! If 'Needs Basal Friction' is true, there must be a 'Basal Friction' bc on Basal Side Set.\n");
  }

  compute_dissipation &= needsDiss;

  dof_names.resize(neq-1);
  dof_names[1] = "W";
  dof_names[2] = "Enthalpy";

  resid_names.resize(neq-1);
  resid_names[1] = dof_names[1] + " Residual";
  resid_names[2] = dof_names[2] + " Residual";

  scatter_names.resize(neq-1);
  scatter_names[1] = "Scatter " + resid_names[1];
  scatter_names[2] = "Scatter " + resid_names[2];

  dof_offsets.resize(neq-1);
  dof_offsets[1] = vecDimFO;
  dof_offsets[2] = dof_offsets[1]+1;


  if(l2ProjectedBoundaryEquation) {
    int eqId=3;
    dof_names[eqId] = "L2 Projected Boundary Laplacian";
    resid_names[eqId] = dof_names[eqId] + " Residual";
    scatter_names[eqId] = "Scatter " + resid_names[eqId];
    dof_offsets[eqId] = dof_offsets[eqId-1]+1;
  }

  if(fluxDivIsPartOfSolution) {
    int eqId = l2ProjectedBoundaryEquation ? 4 : 3;
    dof_names[eqId] = "layered_flux_divergence";
    resid_names[eqId] = dof_names[eqId] + " Residual";
    scatter_names[eqId] = "Scatter " + resid_names[eqId];
    dof_offsets[eqId] = dof_offsets[eqId-1]+1;
  }

  // We *always* use corrected temperature in this problem
  viscosity_use_corrected_temperature = true;

  hydrostatic_pressure_name = params->sublist("Variables Names").get<std::string>("Hydrostatic Pressure Name","hydrostatic_pressure");
  melting_enthalpy_name     = params->sublist("Variables Names").get<std::string>("Melting Enthalpy Name","melting_enthalpy");
  melting_temperature_name  = params->sublist("Variables Names").get<std::string>("Melting Temperature Name","melting_temperature");
  surface_enthalpy_name     = params->sublist("Variables Names").get<std::string>("Surface Enthalpy Name","surface_enthalpy");
  water_content_name        = params->sublist("Variables Names").get<std::string>("Water Content Name","phi");
  geothermal_flux_name      = params->sublist("Variables Names").get<std::string>("Geothermal Flux Name","heat_flux");

  adjustSurfaceHeight = false;
  adjustBedTopo = false;

  // Set parameters for passing coords/near-null space to preconditioner
  rigidBodyModes->setParameters(neq, computeConstantModes, vecDimFO, computeRotationModes);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
StokesFOThermoCoupled::buildEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    Albany::FieldManagerChoice fmchoice,
    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<StokesFOThermoCoupled> op(
      *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);

  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);

  return *op.tags;
}

void
StokesFOThermoCoupled::buildFields(PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  // Allocate memory for unmanaged fields
  fieldUtils = Teuchos::rcp(new Albany::FieldUtils(fm0, dl));
  buildStokesFOBaseFields();

  // Call constructFields<EvalT>() for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructFieldsOp<StokesFOThermoCoupled> op(*this, fm0);
  Sacado::mpl::for_each_no_kokkos<PHAL::AlbanyTraits::BEvalTypes> fe(op);
}

void StokesFOThermoCoupled::constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(vecDimFO+2);
  for (unsigned int i=0; i<vecDimFO; i++) {
    std::stringstream s; s << "U" << i;
    dirichletNames[i] = s.str();
  }
  dirichletNames[vecDimFO] = "W";
  dirichletNames[vecDimFO+1] = "Enth";
  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
  use_sdbcs_ = dirUtils.useSDBCs();
  offsets_ = dirUtils.getOffsets();
  nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

// Neumann BCs
void StokesFOThermoCoupled::constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
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

  int neq_Stokes = 2;
  std::vector<std::string> neumannNames(neq_Stokes + 1);
  Teuchos::Array<Teuchos::Array<int> > offsets;
  offsets.resize(neq_Stokes + 1);

  neumannNames[0] = "U0";
  offsets[0].resize(1);
  offsets[0][0] = 0;
  offsets[neq_Stokes].resize(neq_Stokes);
  offsets[neq_Stokes][0] = 0;
  neumannNames[1] = "U1";
  offsets[1].resize(1);
  offsets[1][0] = 1;
  offsets[neq_Stokes][1] = 1;
  /*
  neumannNames[2] = "W";
  offsets[2].resize(1);
  offsets[2][0] = 2;
  offsets[neq][0] = 2;

  neumannNames[3] = "Enth";
  offsets[3].resize(1);
  offsets[3][0] = 3;
  offsets[neq][0] = 3;
   */

  neumannNames[neq_Stokes] = "Stokes";

  // Construct BC evaluators for all possible names of conditions
  // Should only specify flux vector components (dCdx, dCdy, dCdz), or dCdn, not both
  std::vector<std::string> condNames(6); //(dCdx, dCdy, dCdz), dCdn, basal, P, lateral, basal_scalar_field
  Teuchos::ArrayRCP<std::string> dof_name(1);
  dof_name[0] = "Velocity";

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

  nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_name, true, 0,
                                          condNames, offsets, dl,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
StokesFOThermoCoupled::getValidProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = StokesFOBase::getStokesFOBaseProblemParameters();

  validPL->set<bool>("Adjust Bed Topography to Account for Thickness Changes", false, "");
  validPL->set<bool>("Adjust Surface Height to Account for Thickness Changes", false, "");
  validPL->sublist("LandIce Flux Divergence", false, "Parameters used for Flux Divergence Computation");
  validPL->set<int> ("importCellTemperatureFromMesh", 0, "");
  validPL->set<bool> ("Needs Dissipation", true, "Boolean describing whether we take into account the heat generated by dissipation");
  validPL->set<bool> ("Needs Basal Friction", true, "Boolean describing whether we take into account the heat generated by basal friction");
  validPL->sublist("LandIce Enthalpy", false, "Parameters used for Enthalpy equation.");

  return validPL;
}

void StokesFOThermoCoupled::setupEvaluatorRequests ()
{
  // Add all the StokesFO needs
  StokesFOBase::setupEvaluatorRequests();

  // Volume required interpolations
  build_interp_ev[dof_names[1]][IReq::QP_VAL     ] = true;
  build_interp_ev[dof_names[2]][IReq::QP_VAL     ] = true;
  build_interp_ev[dof_names[2]][IReq::GRAD_QP_VAL] = true;

  build_interp_ev[temperature_name          ][IReq::CELL_VAL   ] = true;
  build_interp_ev[corrected_temperature_name][IReq::CELL_VAL   ] = true;
  build_interp_ev[corrected_temperature_name][IReq::QP_VAL     ] = true;
  build_interp_ev[stiffening_factor_log_name][IReq::QP_VAL     ] = true;
  build_interp_ev[surface_height_name       ][IReq::QP_VAL     ] = true;
  build_interp_ev[surface_height_name       ][IReq::GRAD_QP_VAL] = true;
  build_interp_ev[water_content_name        ][IReq::QP_VAL     ] = true;
  build_interp_ev[water_content_name        ][IReq::GRAD_QP_VAL] = true;
  build_interp_ev[melting_temperature_name  ][IReq::QP_VAL     ] = true;
  build_interp_ev[melting_temperature_name  ][IReq::GRAD_QP_VAL] = true;
  build_interp_ev[melting_enthalpy_name     ][IReq::QP_VAL     ] = true;
  build_interp_ev[melting_enthalpy_name     ][IReq::GRAD_QP_VAL] = true;

  // Side set required interpolations
  ss_build_interp_ev[basalSideName][dof_names[2]            ][IReq::QP_VAL      ] = true;
  ss_build_interp_ev[basalSideName][dof_names[2]            ][IReq::CELL_TO_SIDE] = true;
  ss_build_interp_ev[basalSideName][dof_names[1]            ][IReq::CELL_TO_SIDE] = true;
  ss_build_interp_ev[basalSideName][dof_names[1]            ][IReq::QP_VAL      ] = true;
  ss_build_interp_ev[basalSideName][dof_names[0]            ][IReq::CELL_TO_SIDE] = true;
  ss_build_interp_ev[basalSideName][dof_names[0]            ][IReq::QP_VAL      ] = true;
  ss_build_interp_ev[basalSideName]["basal_dTdz"            ][IReq::QP_VAL      ] = true;
  ss_build_interp_ev[basalSideName][melting_temperature_name][IReq::GRAD_QP_VAL ] = true;
  ss_build_interp_ev[basalSideName][melting_temperature_name][IReq::CELL_TO_SIDE] = true;
  ss_build_interp_ev[basalSideName][melting_enthalpy_name   ][IReq::CELL_TO_SIDE] = true;
  ss_build_interp_ev[basalSideName][melting_enthalpy_name   ][IReq::QP_VAL      ] = true;
  ss_build_interp_ev[basalSideName][water_content_name      ][IReq::CELL_TO_SIDE] = true;
  ss_build_interp_ev[basalSideName][water_content_name      ][IReq::QP_VAL      ] = true;

  if(needsBasFric) {
    ss_build_interp_ev[basalSideName]["Basal Heat"     ][IReq::QP_VAL] = true;
    ss_build_interp_ev[basalSideName]["Basal Heat SUPG"][IReq::QP_VAL] = true;
  }

  ss_build_interp_ev[basalSideName][geothermal_flux_name][IReq::CELL_TO_SIDE] = true;
  ss_build_interp_ev[basalSideName][geothermal_flux_name][IReq::QP_VAL      ] = true;

  ss_utils_needed[basalSideName][UtilityRequest::BFS      ] = true;
  ss_utils_needed[basalSideName][UtilityRequest::QP_COORDS] = true;
  ss_utils_needed[basalSideName][UtilityRequest::NORMALS] = true;

  if(l2ProjectedBoundaryEquation) {
    auto& proj_lapl_params = params->sublist("LandIce L2 Projected Boundary Laplacian");
    auto& ssName = proj_lapl_params.get<std::string>("Side Set Name",basalSideName);
    std::string field_name = proj_lapl_params.get<std::string>("Field Name","basal_friction");

    int eqId = 3;
    ss_build_interp_ev[ssName][dof_names[eqId]][IReq::CELL_TO_SIDE] = true;
    ss_build_interp_ev[ssName][dof_names[eqId]][IReq::QP_VAL] = true;
    ss_build_interp_ev[ssName][field_name][IReq::GRAD_QP_VAL] = true;
  }

  if(fluxDivIsPartOfSolution) {
    ss_build_interp_ev[basalSideName][flux_divergence_name][IReq::GRAD_QP_VAL] = true;
  }
}

void StokesFOThermoCoupled::setFieldsProperties () {
  StokesFOBase::setFieldsProperties();

  if (Albany::mesh_depends_on_parameters() && is_dist_param[ice_thickness_name]) {
    adjustBedTopo = params->get("Adjust Bed Topography to Account for Thickness Changes", false);
    adjustSurfaceHeight = params->get("Adjust Surface Height to Account for Thickness Changes", false);
    TEUCHOS_TEST_FOR_EXCEPTION(adjustBedTopo == adjustSurfaceHeight, std::logic_error, "Error! When the ice thickness is a parameter,\n "
        "either 'Adjust Bed Topography to Account for Thickness Changes' or\n"
        " 'Adjust Surface Height to Account for Thickness Changes' needs to be true.\n");
  }


  // UpdateZCoordinate expects the (observed) bed topography and (observed) surface height to have scalar type MeshScalarT.
  setSingleFieldProperties("observed_bed_topography", FRT::Scalar);
  setSingleFieldProperties("observed_surface_height", FRT::Scalar);

  // All dofs have scalar type Scalar (i.e., they depend on the solution)
  setSingleFieldProperties(dof_names[1], FRT::Scalar, FST::Scalar);  // Vertical velocity
  setSingleFieldProperties(dof_names[2], FRT::Scalar, FST::Scalar);  // Enthalpy
  if (fluxDivIsPartOfSolution || l2ProjectedBoundaryEquation) {
    setSingleFieldProperties(dof_names[3], FRT::Scalar, FST::Scalar);  // L2 Projected Boundary Laplacian || FluxDiv
  }
  if (fluxDivIsPartOfSolution && l2ProjectedBoundaryEquation) {
    setSingleFieldProperties(dof_names[4], FRT::Scalar, FST::Scalar);  // FluxDiv
  }

  setSingleFieldProperties(surface_enthalpy_name     , FRT::Scalar);
  setSingleFieldProperties(flow_factor_name          , FRT::Scalar); // Already processed in StokesFOBase, but need to adjust scalar type
  setSingleFieldProperties("basal_melt_rate"         , FRT::Scalar);
  setSingleFieldProperties("basal_dTdz"              , FRT::Scalar);
  setSingleFieldProperties("Basal Heat"              , FRT::Scalar);
  setSingleFieldProperties("Basal Heat SUPG"         , FRT::Scalar);
  setSingleFieldProperties(geothermal_flux_name      , FRT::Scalar, FST::ParamScalar);
  setSingleFieldProperties(water_content_name        , FRT::Scalar, FST::Scalar);
  setSingleFieldProperties(temperature_name          , FRT::Scalar, FST::Scalar);
  setSingleFieldProperties(stiffening_factor_log_name, FRT::Scalar);
  setSingleFieldProperties(corrected_temperature_name, FRT::Scalar, FST::Scalar); // Already processed in StokesFOBase, but need to adjust scalar type
  setSingleFieldProperties(melting_temperature_name  , FRT::Scalar, FST::MeshScalar); //depends on z coord
  setSingleFieldProperties(melting_enthalpy_name     , FRT::Scalar, FST::MeshScalar); //depends on z coord
  setSingleFieldProperties(hydrostatic_pressure_name , FRT::Scalar);
  setSingleFieldProperties("basal_vert_velocity"     , FRT::Scalar);
}

} // namespace LandIce
