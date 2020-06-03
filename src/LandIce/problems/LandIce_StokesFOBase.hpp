//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_FO_BASE_HPP
#define LANDICE_STOKES_FO_BASE_HPP

#include "LandIce_GatherVerticallyContractedSolution.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "PHAL_AddNoise.hpp"
#include "PHAL_FieldFrobeniusNorm.hpp"
#include "PHAL_LoadSideSetStateField.hpp"
#include "PHAL_LoadStateField.hpp"
#include "PHAL_SaveSideSetStateField.hpp"
#include "PHAL_SaveStateField.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"
#include "LandIce_ResponseUtilities.hpp"

#include "LandIce_BasalFrictionCoefficient.hpp"
#include "LandIce_BasalFrictionCoefficientGradient.hpp"
#include "LandIce_DOFDivInterpolationSide.hpp"
#include "LandIce_EffectivePressure.hpp"
#include "LandIce_FlowRate.hpp"
#include "LandIce_FluxDiv.hpp"
#include "LandIce_IceOverburden.hpp"
#include "LandIce_ParamEnum.hpp"
#include "LandIce_ProblemUtils.hpp"
#include "PHAL_SharedParameter.hpp"
#include "LandIce_StokesFOBasalResid.hpp"
#include "LandIce_StokesFOLateralResid.hpp"
#include "LandIce_StokesFOResid.hpp"
#ifdef CISM_HAS_LANDICE
#include "LandIce_CismSurfaceGradFO.hpp"
#endif
#include "LandIce_StokesFOBodyForce.hpp"
#include "LandIce_StokesFOStress.hpp"
#include "LandIce_Time.hpp"
#include "LandIce_ViscosityFO.hpp"
#include "LandIce_Dissipation.hpp"
#include "LandIce_UpdateZCoordinate.hpp"

#include <string.hpp> // For util::upper_case (do not confuse this with <string>! string.hpp is an Albany file)

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce
{

/*!
 *  Base class for all StokesFO* problems.
 *
 *  This class implements some methods that are used across all StokesFO* problems,
 *  so to reduce code duplication. In particular, this class offers methods for:
 *    - register all the states, and create evaluators for load/save/gather/scatter of states/parameters,
 *    - create evaluators for landice bcs (basal friction and lateral)
 *    - create evaluators for surface velocity and SMB diagnostistc
 *    - create evaluators for responses
 */
class StokesFOBase : public Albany::AbstractProblem {
public:

  //! Return number of spatial dimensions
  int spatialDimension() const { return numDim; }

  //! Get boolean telling code if SDBCs are utilized
  bool useSDBCs() const {return use_sdbcs_; }

  //! Build the PDE instantiations, boundary conditions, and initial solution
  void buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                     Albany::StateManager& stateMgr);

protected:
  StokesFOBase (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                const Teuchos::RCP<ParamLib>& paramLib_,
                const int numDim_);

  template <typename EvalT>
  void constructStokesFOBaseEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                        const Albany::MeshSpecsStruct& meshSpecs,
                                        Albany::StateManager& stateMgr,
                                        Albany::FieldManagerChoice fieldManagerChoice);

  template <typename EvalT>
  void constructStatesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                  const Albany::MeshSpecsStruct& meshSpecs,
                                  Albany::StateManager& stateMgr);

  template <typename EvalT>
  void constructVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                    const Albany::MeshSpecsStruct& meshSpecs,
                                    Albany::StateManager& stateMgr,
                                    Albany::FieldManagerChoice fieldManagerChoice);

  template <typename EvalT>
  void constructInterpolationEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  template <typename EvalT>
  void constructSideUtilityFields (PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  template <typename EvalT>
  void constructBasalBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  template <typename EvalT>
  void constructLateralBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  template <typename EvalT>
  void constructSurfaceVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  template <typename EvalT>
  void constructSMBEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                               const Albany::MeshSpecsStruct& meshSpecs);

  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructStokesFOBaseResponsesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                            const Albany::MeshSpecsStruct& meshSpecs,
                                            Albany::StateManager& stateMgr,
                                            Albany::FieldManagerChoice fieldManagerChoice,
                                            const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  virtual void constructDirichletEvaluators (const Albany::MeshSpecsStruct& /* meshSpecs */) {}
  virtual void constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecsStruct>& /* meshSpecs */) {}

  Teuchos::RCP<Teuchos::ParameterList>
  getStokesFOBaseProblemParameters () const;

  void setSingleFieldProperties (const std::string& fname,
                                 const int rank,
                                 const FieldScalarType st,
                                 const FieldLocation location);

  void parseInputFields ();

  // This method sets the properties of fields that need to be handled automatically (e.g., need interpolation evaluators)
  virtual void setFieldsProperties ();

  // Note: derived problems should override this function to add more requests. Their implementation should *most likely*
  //       include a call to the base class' implementation.
  virtual void setupEvaluatorRequests ();

  // ------------------- Members ----------------- //

  using IntrepidBasis    = Intrepid2::Basis<PHX::Device, RealType, RealType>;
  using IntrepidCubature = Intrepid2::Cubature<PHX::Device>;

  // Topology, basis and cubature of cells
  Teuchos::RCP<shards::CellTopology>  cellType;
  Teuchos::RCP<IntrepidBasis>         cellBasis;
  Teuchos::RCP<IntrepidCubature>      cellCubature;

  // Topology, basis and cubature of side sets
  std::map<std::string,Teuchos::RCP<shards::CellTopology>>  sideType;
  std::map<std::string,Teuchos::RCP<IntrepidBasis>>         sideBasis;
  std::map<std::string,Teuchos::RCP<IntrepidCubature>>      sideCubature;

  //! Discretization parameters
  Teuchos::RCP<Teuchos::ParameterList> discParams;

  // Layouts
  Teuchos::RCP<Albany::Layouts> dl;

  // Parameter lists for LandIce-specific BCs
  std::map<LandIceBC,std::vector<Teuchos::RCP<Teuchos::ParameterList>>>  landice_bcs;

  // Surface side, where velocity diagnostics are computed (e.g., velocity mismatch)
  std::string surfaceSideName;

  // Basal side, where thickness-related diagnostics are computed (e.g., SMB)
  std::string basalSideName;

  // In these three, the entry [0] always refers to the velocity
  Teuchos::ArrayRCP<std::string> dof_names;
  Teuchos::ArrayRCP<int>         dof_offsets;
  Teuchos::ArrayRCP<std::string> resid_names;
  Teuchos::ArrayRCP<std::string> scatter_names;

  int numDim;
  int vecDimFO;

  /// Boolean marking whether SDBCs are used
  bool use_sdbcs_;

  // Whether the problem is coupled with other physics
  bool temperature_coupled;
  bool hydrology_coupled;

  // Whether to use corrected temperature in the viscosity
  bool viscosity_use_corrected_temperature;
  bool compute_dissipation;

  // Variables used to track properties of fields and parameters
  std::map<std::string, bool>               is_input_field;
  std::map<std::string, bool>               is_computed_field;
  std::map<std::string, FieldLocation>      field_location;
  std::map<std::string, int>                field_rank;
  std::map<std::string, FieldScalarType>    field_scalar_type;

  std::map<std::string,bool> is_dist;
  std::map<std::string,bool> save_sensitivities;
  std::map<std::string,std::string> dist_params_name_to_mesh_part;

  std::map<std::string, std::map<std::string,bool>>   is_ss_input_field;
  std::map<std::string, std::map<std::string,bool>>   is_ss_computed_field;

  std::map<std::string,bool>  is_dist_param;
  std::map<std::string,bool>  is_extruded_param;
  std::map<std::string, int>  extruded_params_levels;

  // Track the utility evaluators that a field needs
  std::map<std::string, std::map<InterpolationRequest,bool>> build_interp_ev;
  std::map<std::string, std::map<std::string, std::map<InterpolationRequest,bool>>> ss_build_interp_ev;

  // Track the utility evaluators needed by each side set
  std::map<std::string,std::map<UtilityRequest,bool>>  ss_utils_needed;

  // Name of common variables (constructor provides defaults)
  std::string body_force_name;
  std::string surface_height_name;
  std::string ice_thickness_name;
  std::string flux_divergence_name;
  std::string bed_topography_name;
  std::string temperature_name;
  std::string corrected_temperature_name;
  std::string flow_factor_name;
  std::string stiffening_factor_name;
  std::string effective_pressure_name;
  std::string vertically_averaged_velocity_name;

  //! Problem PL 
  const Teuchos::RCP<Teuchos::ParameterList> params;

  template<typename T>
  std::string print_map_keys (const std::map<std::string,T>& map);
};

template <typename EvalT>
void StokesFOBase::
constructStokesFOBaseEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                 const Albany::MeshSpecsStruct& meshSpecs,
                                 Albany::StateManager& stateMgr,
                                 Albany::FieldManagerChoice fieldManagerChoice)
{
  // --- States/parameters --- //
  constructStatesEvaluators<EvalT> (fm0, meshSpecs, stateMgr);

  // --- Interpolation utilities for fields ---//
  constructInterpolationEvaluators<EvalT> (fm0);

  // --- Sides utility fields ---//
  constructSideUtilityFields<EvalT> (fm0);

  // --- Velocity evaluators --- //
  constructVelocityEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice);

  // --- Lateral BC evaluators (if needed) --- //
  constructLateralBCEvaluators<EvalT> (fm0);

  // --- Basal BC evaluators (if needed) --- //
  constructBasalBCEvaluators<EvalT> (fm0);

  // --- SMB-related evaluators (if needed) --- //
  constructSMBEvaluators<EvalT> (fm0, meshSpecs);
}

template <typename EvalT>
void StokesFOBase::constructStatesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                              const Albany::MeshSpecsStruct& meshSpecs,
                                              Albany::StateManager& stateMgr)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Temporary variables used numerous times below
  Albany::StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // ---------------------------- Registering state variables ------------------------- //

  std::string stateName, fieldName;

  // Volume mesh requirements
  Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
  int num_fields = req_fields_info.get<int>("Number Of Fields",0);

  std::string fieldType, fieldUsage, meshPart;
  bool nodal_state, scalar_state;
  for (int ifield=0; ifield<num_fields; ++ifield) {
    Teuchos::ParameterList& thisFieldList = req_fields_info.sublist(Albany::strint("Field", ifield));

    // Get current state specs
    fieldName = thisFieldList.get<std::string>("Field Name");
    stateName = thisFieldList.get<std::string>("State Name", fieldName);
    fieldUsage = thisFieldList.get<std::string>("Field Usage","Input"); // WARNING: assuming Input if not specified

    if (fieldUsage == "Unused") {
      continue;
    }

    fieldType  = thisFieldList.get<std::string>("Field Type");

    is_dist_param.insert(std::pair<std::string,bool>(stateName, false)); //gets inserted only if not there.
    is_dist.insert(std::pair<std::string,bool>(stateName, false)); //gets inserted only if not there.

    meshPart = is_dist[stateName] ? dist_params_name_to_mesh_part[stateName] : "";

    if(fieldType == "Elem Scalar") {
      entity = Albany::StateStruct::ElemData;
      p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, meshSpecs.ebName, true, &entity, meshPart);
      nodal_state = false;
      scalar_state = true;
    } else if(fieldType == "Node Scalar") {
      entity = is_dist[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
      if(is_dist[stateName] && save_sensitivities[stateName]) {
        p = stateMgr.registerStateVariable(stateName + "_sensitivity", dl->node_scalar, meshSpecs.ebName, true, &entity, meshPart);
      }
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, meshSpecs.ebName, true, &entity, meshPart);
      nodal_state = true;
      scalar_state = true;
    } else if(fieldType == "Elem Vector") {
      entity = Albany::StateStruct::ElemData;
      p = stateMgr.registerStateVariable(stateName, dl->cell_vector, meshSpecs.ebName, true, &entity, meshPart);
      nodal_state = false;
      scalar_state = false;
    } else if(fieldType == "Node Vector") {
      entity = is_dist[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(stateName, dl->node_vector, meshSpecs.ebName, true, &entity, meshPart);
      nodal_state = true;
      scalar_state = false;
    }

    // Do we need to load/gather the state/parameter?
    if (is_dist[stateName]) {
      // A distributed field (likely a parameter): gather or scatter it (depending on whether is marked as computed)
      if (is_computed_field[stateName]) {
        ev = evalUtils.constructScatterScalarNodalParameter(stateName,fieldName);
        fm0.template registerEvaluator<EvalT>(ev);
        // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
        if (ev->evaluatedFields().size()>0) {
          fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
        }
      } else {
        // Not computed: gather it
        if (is_extruded_param[stateName]) {
          ev = evalUtils.constructGatherScalarExtruded2DNodalParameter(stateName,fieldName);
          fm0.template registerEvaluator<EvalT>(ev);
        } else {
          ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
          fm0.template registerEvaluator<EvalT>(ev);
        }
      }
    } else {
      // Do we need to save the state?
      if (fieldUsage == "Output" || fieldUsage == "Input-Output") {
        // An output: save it.
        p->set<bool>("Nodal State", nodal_state);
        ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);

        // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
        if (ev->evaluatedFields().size()>0) {
          fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
        }
      }

      if (fieldUsage == "Input" || fieldUsage == "Input-Output") {
        // Not a parameter but still required as input: load it.
        p->set<std::string>("Field Name", fieldName);
        if (field_scalar_type[stateName]==FieldScalarType::ParamScalar) {
          ev = Teuchos::rcp(new PHAL::LoadStateFieldPST<EvalT,PHAL::AlbanyTraits>(*p));
        } else if (field_scalar_type[stateName]==FieldScalarType::MeshScalar) {
          ev = Teuchos::rcp(new PHAL::LoadStateFieldMST<EvalT,PHAL::AlbanyTraits>(*p));
        } else {
          ev = Teuchos::rcp(new PHAL::LoadStateFieldRT<EvalT,PHAL::AlbanyTraits>(*p));
        }
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }
  }

  // Side set requirements
  Teuchos::Array<std::string> ss_names;
  if (discParams->sublist("Side Set Discretizations").isParameter("Side Sets")) {
    ss_names = discParams->sublist("Side Set Discretizations").get<Teuchos::Array<std::string>>("Side Sets");
  } 
  for (int i=0; i<ss_names.size(); ++i) {
    const std::string& ss_name = ss_names[i];
    Teuchos::ParameterList& info = discParams->sublist("Side Set Discretizations").sublist(ss_name).sublist("Required Fields Info");
    num_fields = info.get<int>("Number Of Fields",0);
    Teuchos::RCP<PHX::DataLayout> dl_temp;
    Teuchos::RCP<PHX::DataLayout> sns;
    int numLayers;

    const std::string& sideEBName = meshSpecs.sideSetMeshSpecs.at(ss_name)[0]->ebName;
    Teuchos::RCP<Albany::Layouts> ss_dl = dl->side_layouts.at(ss_name);
    for (int ifield=0; ifield<num_fields; ++ifield) {
      Teuchos::ParameterList& thisFieldList =  info.sublist(Albany::strint("Field", ifield));

      // Get current state specs
      fieldName = thisFieldList.get<std::string>("Field Name");
      stateName = thisFieldList.get<std::string>("State Name", fieldName);
      fieldName = fieldName + "_" + ss_name;
      fieldUsage = thisFieldList.get<std::string>("Field Usage","Input"); // WARNING: assuming Input if not specified

      if (fieldUsage == "Unused") {
        continue;
      }

      //meshPart = is_dist_param[stateName] ? dist_params_name_to_mesh_part[stateName] : "";
      meshPart = ""; // Distributed parameters are defined either on the whole volume mesh or on a whole side mesh. Either way, here we want "" as part (the whole mesh).

      fieldType  = thisFieldList.get<std::string>("Field Type");

      // Registering the state
      if(fieldType == "Elem Scalar") {
        entity = Albany::StateStruct::ElemData;
        p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->cell_scalar2, sideEBName, true, &entity, meshPart);
        nodal_state = false;
        scalar_state = true;
      } else if(fieldType == "Node Scalar") {
        entity = is_dist[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->node_scalar, sideEBName, true, &entity, meshPart);
        nodal_state = true;
        scalar_state = true;
      } else if(fieldType == "Elem Vector") {
        entity = Albany::StateStruct::ElemData;
        p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->cell_vector, sideEBName, true, &entity, meshPart);
        nodal_state = false;
        scalar_state = false;
      } else if(fieldType == "Node Vector") {
        entity = is_dist[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->node_vector, sideEBName, true, &entity, meshPart);
        nodal_state = true;
        scalar_state = false;
      } else if(fieldType == "Elem Layered Scalar") {
        entity = Albany::StateStruct::ElemData;
        sns = ss_dl->cell_scalar2;
        numLayers = thisFieldList.get<int>("Number Of Layers");
        dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,LayerDim>(sns->extent(0),sns->extent(1),numLayers));
        stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
        nodal_state = false;
        scalar_state = false;
      } else if(fieldType == "Node Layered Scalar") {
        entity = is_dist[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        sns = ss_dl->node_scalar;
        numLayers = thisFieldList.get<int>("Number Of Layers");
        dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,LayerDim>(sns->extent(0),sns->extent(1),sns->extent(2),numLayers));
        stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
        scalar_state = false;
        nodal_state = true;
      } else if(fieldType == "Elem Layered Vector") {
        entity = Albany::StateStruct::ElemData;
        sns = ss_dl->cell_vector;
        numLayers = thisFieldList.get<int>("Number Of Layers");
        dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Dim,LayerDim>(sns->extent(0),sns->extent(1),sns->extent(2),numLayers));
        stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
        scalar_state = false;
        nodal_state = false;
      } else if(fieldType == "Node Layered Vector") {
        entity = is_dist[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        sns = ss_dl->node_vector;
        numLayers = thisFieldList.get<int>("Number Of Layers");
        dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,Dim,LayerDim>(sns->extent(0),sns->extent(1),sns->extent(2),
                                                                               sns->extent(3),numLayers));
        stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
        scalar_state = false;
        nodal_state = true;
      }

      // Creating load/save evaluator(s)
      // Note:
      //  - dist fields should not be loaded/gathered on the ss; instead, gather them in 3D, and project on the ss;
      //  - dist fields should not be saved on the ss if they are computed, since they are not correct until scattered, which does not happen before projection.
      if ( !(is_dist[stateName] && is_computed_field[stateName]) && (fieldUsage == "Output" || fieldUsage == "Input-Output")) {
        // An output: save it.
        p->set<bool>("Nodal State", nodal_state);
        p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
        ev = Teuchos::rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,ss_dl));
        fm0.template registerEvaluator<EvalT>(ev);

        // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
        if (ev->evaluatedFields().size()>0) {
          fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
        }
      }

      if (!is_dist[stateName] && (fieldUsage == "Input" || fieldUsage == "Input-Output")) {
        // Not a parameter but required as input: load it.
        p->set<std::string>("Field Name", fieldName);
        if (field_scalar_type[stateName]==FieldScalarType::ParamScalar) {
          ev = Teuchos::rcp(new PHAL::LoadSideSetStateFieldPST<EvalT,PHAL::AlbanyTraits>(*p));
        } else if (field_scalar_type[stateName]==FieldScalarType::MeshScalar) {
          ev = Teuchos::rcp(new PHAL::LoadSideSetStateFieldMST<EvalT,PHAL::AlbanyTraits>(*p));
        } else {
          ev = Teuchos::rcp(new PHAL::LoadSideSetStateFieldRT<EvalT,PHAL::AlbanyTraits>(*p));
        } 
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }
  }
}

template <typename EvalT>
void StokesFOBase::
constructInterpolationEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  std::map<FieldScalarType,Teuchos::RCP<const Albany::EvaluatorUtilsBase<PHAL::AlbanyTraits>>> utils_map;
  utils_map[FieldScalarType::Scalar]      = Teuchos::rcpFromRef(evalUtils.getSTUtils());
  utils_map[FieldScalarType::ParamScalar] = Teuchos::rcpFromRef(evalUtils.getPSTUtils());
  utils_map[FieldScalarType::MeshScalar]  = Teuchos::rcpFromRef(evalUtils.getMSTUtils());
  utils_map[FieldScalarType::Real]        = Teuchos::rcpFromRef(evalUtils.getRTUtils());

  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;

  // Loop on all input fields
  for (auto& it : build_interp_ev) {
    // Get the field name
    const std::string& fname = it.first;

    // If there's no information about this field, we assume it is not needed, so we skip it.
    // If it WAS indeed needed, Phalanx DAG will miss a node, and an exception will be thrown.
    if (field_scalar_type.find(fname)==field_scalar_type.end()) {
      continue;
    }

    // Get the right evaluator utils for this field.
    TEUCHOS_TEST_FOR_EXCEPTION (field_scalar_type.find(fname)==field_scalar_type.end(), std::runtime_error,
                                "Error! Scalar type for field '" + fname + "' not found.\n" +
                                "       Current map keys:" + print_map_keys(field_scalar_type) + "\n");
    const FieldScalarType st = field_scalar_type.at(fname);

    TEUCHOS_TEST_FOR_EXCEPTION (utils_map.find(st)==utils_map.end(), std::runtime_error,
                                "Error! Evaluators utils for scalar type '" + e2str(st) + "' not found.\n");
    const auto& utils = *utils_map.at(st);

    // Check whether we can use memoization for this field. Criteria:
    //  - need to have memoization enabled
    //  - cannot be a solution-dependent field
    //  - cannot be a distributed parameter
    //  - if param scalar, params must not depend on solution
    //  - if mesh scalar, mesh must not depend on solution/parameters
    // Note: if a field depends on a distributed parameter (and not on the solution), it would pass these tests,
    //       but memoization would be wrong. Therefore, we actually enable memoization if there are NO dist params.
    //       A better choice would be to track dependencies, but we cannot do this easily. We should probably
    //       have our own Albany::FieldManager (and Albany::Evaluator), which checks that all evaluators that
    //       try to use memoization actually can do it (i.e., they do not depend on fields that cannot be memoized).
    //
    // TODO: We can now track dependencies but we need to specify which MDField changes based on whether
    //       the dist param changes
    //
//    bool useMemoization = enableMemoizer && (is_dist_param.size()==0) && st!=FieldScalarType::Scalar;
//    if (st==FieldScalarType::ParamScalar) {
//      useMemoization &= !Albany::params_depend_on_solution();
//    } else if (st==FieldScalarType::MeshScalar) {
//      useMemoization &= !Albany::mesh_depends_on_solution() && !Albany::mesh_depends_on_parameters();
//    }

    // Get the needs of this field
    auto& needs = it.second;
    TEUCHOS_TEST_FOR_EXCEPTION (field_rank.find(fname)==field_rank.end(), std::runtime_error,
                                "Error! Rank of field '" + fname + "' not found.\n" +
                                "       Current map keys:" + print_map_keys(field_rank) + "\n");
    int rank = field_rank.at(fname);

    // For dofs, we can get a faster interpolation, knowing the offset
    auto dof_it = std::find(dof_names.begin(),dof_names.end(),fname);
    int offset = dof_it==dof_names.end() ? -1 : dof_offsets[std::distance(dof_names.begin(),dof_it)];

    TEUCHOS_TEST_FOR_EXCEPTION (field_location.find(fname)==field_location.end(), std::runtime_error,
                                "Error! Location of field '" + fname + "' not found.\n" +
                                "       Current map keys:" + print_map_keys(field_location) + "\n");
    const FieldLocation entity = field_location.at(fname);
    if (needs[InterpolationRequest::QP_VAL]) {
      TEUCHOS_TEST_FOR_EXCEPTION(entity==FieldLocation::Cell, std::logic_error, "Error! Cannot interpolate a field not defined on nodes.\n");
      if (rank==0) {
        ev = utils.constructDOFInterpolationEvaluator(fname, offset);
      } else if (rank==1) {
        ev = utils.constructDOFVecInterpolationEvaluator(fname, offset);
      } else if (rank==2) {
        ev = utils.constructDOFTensorInterpolationEvaluator(fname, offset);
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Cannot interpolate to the quad points a field of rank " + std::to_string(rank) << ".\n");
      }
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (needs[InterpolationRequest::GRAD_QP_VAL]) {
      TEUCHOS_TEST_FOR_EXCEPTION(entity==FieldLocation::Cell, std::logic_error, "Error! Cannot interpolate a field not defined on nodes.\n");
      if (rank==0) {
        ev = utils.constructDOFGradInterpolationEvaluator(fname, offset);
      } else if (rank==1) {
        ev = utils.constructDOFVecGradInterpolationEvaluator(fname, offset);
      } else if (rank==2) {
        ev = utils.constructDOFTensorGradInterpolationEvaluator(fname, offset);
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Cannot interpolate to the quad points the gradient of a field of rank " + std::to_string(rank) << ".\n");
      }
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (needs[InterpolationRequest::CELL_VAL] && entity==FieldLocation::Node) {
      if (rank==0) {
        ev = utils.constructNodesToCellInterpolationEvaluator (fname, /*isVectorField = */ false);
      } else if (rank==1) {
        ev = utils.constructNodesToCellInterpolationEvaluator (fname, /*isVectorField = */ true);
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Cannot interpolate to the cell a field of rank " + std::to_string(rank) << ".\n");
      }
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (needs[InterpolationRequest::CELL_VAL] && entity==FieldLocation::QuadPoint) {
      if (rank==0) {
        ev = utils.constructQuadPointsToCellInterpolationEvaluator (fname, dl->qp_scalar, dl->cell_scalar2);
      } else if (rank==1) {
        ev = utils.constructQuadPointsToCellInterpolationEvaluator (fname, dl->qp_vector, dl->cell_vector);
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Cannot interpolate to the cell a field of rank " + std::to_string(rank) << ".\n");
      }
      fm0.template registerEvaluator<EvalT> (ev);
    }
  }

  // Loop on all side sets
  for (auto& it_outer : ss_build_interp_ev) {
    const std::string& ss_name = it_outer.first;

    // Loop on all fields
    for (auto& it : it_outer.second) {

      // Get field name (with and without side name)
      const std::string fname = it.first;
      const std::string fname_side = fname + "_" + ss_name;

      // Get the needs of this field
      auto& needs = it.second;

      // Get location, rank, and scalar type of the field.
      // If we are missing some information about this field, we assume it is not needed, so we skip it.
      // If it WAS indeed needed, Phalanx DAG will miss a node, and an exception will be thrown.
      if (field_scalar_type.find(fname)==field_scalar_type.end()) {
        continue;
      }

      TEUCHOS_TEST_FOR_EXCEPTION (field_location.find(fname)==field_location.end(), std::runtime_error,
                                  "Error! Location of field '" + fname + "' not found (ss name: " + ss_name + ").\n" +
                                  "       Current map keys:" + print_map_keys(field_location) + "\n");
      const FieldLocation entity = field_location.at(fname);
      TEUCHOS_TEST_FOR_EXCEPTION (field_rank.find(fname)==field_rank.end(), std::runtime_error,
                                  "Error! Rank of field '" + fname + "' not found (ss name: " + ss_name + ").\n" +
                                  "       Current map keys:" + print_map_keys(field_rank) + "\n");
      const int rank = field_rank.at(fname);

      TEUCHOS_TEST_FOR_EXCEPTION (rank<0 || rank>1, std::logic_error, "Error! Interpolation on side only available for scalar and vector fields.\n");

      const std::string layout = e2str(entity) + " " + rank2str(rank);
      TEUCHOS_TEST_FOR_EXCEPTION (field_scalar_type.find(fname)==field_scalar_type.end(), std::runtime_error,
                                  "Error! Scalar type for field '" + fname + "' not found (ss name: " + ss_name + ").\n" +
                                  "       Current map keys:" + print_map_keys(field_scalar_type) + "\n");
      const FieldScalarType st = field_scalar_type.at(fname);

      // Check whether we can use memoization for this field. Criteria:
      //  - need to have memoization enabled
      //  - cannot be a solution-dependent field
      //  - cannot be a distributed parameter
      //  - if param scalar, params must not depend on solution
      //  - if mesh scalar, mesh must not depend on solution/parameters
      // Note: if a field depends on a distributed parameter (and not on the solution), it would pass these tests,
      //       but memoization would be wrong. Therefore, we actually enable memoization if there are NO dist params.
      //       A better choice would be to track dependencies, but we cannot do this easily. We should probably
      //       have our own Albany::FieldManager (and Albany::Evaluator), which checks that all evaluators that
      //       try to use memoization actually can do it (i.e., they do not depend on fields that cannot be memoized).
      //
      // TODO: We can now track dependencies but we need to specify which MDField changes based on whether
      //       the dist param changes
      //
//      bool useMemoization = enableMemoizer && (is_dist_param.size()==0) && st!=FieldScalarType::Scalar;
//      if (st==FieldScalarType::ParamScalar) {
//        useMemoization &= !Albany::params_depend_on_solution();
//      } else if (st==FieldScalarType::MeshScalar) {
//        useMemoization &= !Albany::mesh_depends_on_solution() && !Albany::mesh_depends_on_parameters();
//      }

      // Get the right evaluator utils for this field.
      TEUCHOS_TEST_FOR_EXCEPTION (utils_map.find(st)==utils_map.end(), std::runtime_error,
                                  "Error! Evaluators utils for scalar type '" + e2str(st) + "' not found (ss name: " + ss_name + ").\n");
      const auto& utils = *utils_map.at(st);

      if (needs[InterpolationRequest::QP_VAL]) {
        TEUCHOS_TEST_FOR_EXCEPTION (entity!=FieldLocation::Node, std::logic_error, "Error! DOF interpolation is only for fields defined at nodes.\n");
        if (rank==0) {
          ev = utils.constructDOFInterpolationSideEvaluator (fname_side, ss_name);
        } else {
          ev = utils.constructDOFVecInterpolationSideEvaluator (fname_side, ss_name);
        }
        fm0.template registerEvaluator<EvalT> (ev);
      }

      if (needs[InterpolationRequest::GRAD_QP_VAL]) {
        TEUCHOS_TEST_FOR_EXCEPTION (entity!=FieldLocation::Node, std::logic_error, "Error! DOF Grad interpolation is only for fields defined at nodes.\n");
        if (rank==0) {
          ev = utils.constructDOFGradInterpolationSideEvaluator (fname_side, ss_name);
        } else {
          ev = utils.constructDOFVecGradInterpolationSideEvaluator (fname_side, ss_name);
        }
        fm0.template registerEvaluator<EvalT> (ev);
      }

      if (needs[InterpolationRequest::CELL_VAL]) {
        // Intepolate field at Side from Quad points values
        ev = utils.constructSideQuadPointsToSideInterpolationEvaluator (fname_side, ss_name, rank==1);
        fm0.template registerEvaluator<EvalT> (ev);
      }

      // Project to the side only if it is requested, it is NOT an input on the side,
      // and it is NOT computed on the side. Furthermore, the corresponding 3D field mus
      // be an input or computed field in 3D
      // Note: computed does not mean that it depends on the solution or on distributed parameters.
      //       It only means that it is computed from other quantities.
      // Note: this does not check that the 3D field exists. You must ensure that.
      if ( needs[InterpolationRequest::CELL_TO_SIDE] &&
          !is_ss_input_field[ss_name][fname]         &&
          !is_ss_computed_field[ss_name][fname]) {
          // (is_input_field[fname] || is_computed_field[fname] || is_dist_param[fname])) {
        // Project from cell to side
        ev = utils.constructDOFCellToSideEvaluator(fname, ss_name, layout, cellType, fname_side);
        fm0.template registerEvaluator<EvalT> (ev);
      }

      if (needs[InterpolationRequest::SIDE_TO_CELL]) {
        // Project from cell to side
        ev = utils.constructDOFSideToCellEvaluator(fname_side, ss_name, layout, cellType, fname);
        fm0.template registerEvaluator<EvalT> (ev);
      }
    }
  }
}

template <typename EvalT>
void StokesFOBase::
constructSideUtilityFields (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;

  for (auto& it : ss_utils_needed) {
    const std::string& ss_name = it.first;

    //---- Compute side basis functions
    if (it.second[UtilityRequest::BFS] || it.second[UtilityRequest::NORMALS]) {
      // BF, GradBF, w_measure, Tangents, Metric, Metric Det, Inverse Metric
      ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis[ss_name], sideCubature[ss_name],
                                                                 ss_name, it.second[UtilityRequest::NORMALS]);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (it.second[UtilityRequest::QP_COORDS]) {
      // QP coordinates
      ev = evalUtils.constructMapToPhysicalFrameSideEvaluator(cellType, sideCubature[ss_name], ss_name);
      fm0.template registerEvaluator<EvalT> (ev);

      // Baricenter coordinate
      ev = evalUtils.getMSTUtils().constructSideQuadPointsToSideInterpolationEvaluator(Albany::coord_vec_name + "_" + ss_name, ss_name, 1);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    // If any of the above was true, we need coordinates of vertices on the side
    if (it.second[UtilityRequest::BFS] || it.second[UtilityRequest::QP_COORDS] || it.second[UtilityRequest::NORMALS]) {
      ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,ss_name,"Vertex Vector",cellType,Albany::coord_vec_name +" " + ss_name);
      fm0.template registerEvaluator<EvalT> (ev);
    }
  }
}

template <typename EvalT>
void StokesFOBase::
constructVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                             const Albany::MeshSpecsStruct& meshSpecs,
                             Albany::StateManager& stateMgr,
                             Albany::FieldManagerChoice fieldManagerChoice)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  std::string param_name;

  // ------------------- Interpolations and utilities ------------------ //

  // Map to physical frame
  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Compute basis funcitons
  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Get coordinate of cell baricenter
  ev = evalUtils.getMSTUtils().constructQuadPointsToCellInterpolationEvaluator(Albany::coord_vec_name, dl->qp_gradient, dl->cell_gradient);
  fm0.template registerEvaluator<EvalT> (ev);

  // -------------------------------- LandIce evaluators ------------------------- //

  // --- FO Stokes Stress --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Stress"));

  //Input
  p->set<std::string>("Velocity QP Variable Name", dof_names[0]);
  p->set<std::string>("Velocity Gradient QP Variable Name", dof_names[0] + " Gradient");
  p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
  p->set<std::string>("Surface Height QP Name", surface_height_name);
  p->set<std::string>("Coordinate Vector Name", Albany::coord_vec_name);
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

  //Output
  p->set<std::string>("Stress Variable Name", "Stress Tensor");

  ev = Teuchos::rcp(new LandIce::StokesFOStress<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- FO Stokes Resid --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Resid"));

  //Input
  p->set<std::string>("Weighted BF Variable Name", Albany::weighted_bf_name);
  p->set<std::string>("Weighted Gradient BF Variable Name", Albany::weighted_grad_bf_name);
  p->set<std::string>("Velocity QP Variable Name", dof_names[0]);
  p->set<std::string>("Velocity Gradient QP Variable Name", dof_names[0] + " Gradient");
  p->set<std::string>("Body Force Variable Name", body_force_name);
  p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
  p->set<std::string>("Coordinate Vector Name", Albany::coord_vec_name);
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Equation Set"));

  //Output
  p->set<std::string>("Residual Variable Name", resid_names[0]);

  ev = Teuchos::rcp(new LandIce::StokesFOResid<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Shared Parameter for Continuation:  ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Homotopy Parameter"));

  param_name = "Glen's Law Homotopy Parameter";
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>> ptr_homotopy;
  ptr_homotopy = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>(*p,dl));
  ptr_homotopy->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Viscosity").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_homotopy);

  //--- LandIce Flow Rate ---//
  if(params->sublist("LandIce Viscosity").isParameter("Flow Rate Type")) {
    if((params->sublist("LandIce Viscosity").get<std::string>("Flow Rate Type") == "From File") ||
       (params->sublist("LandIce Viscosity").get<std::string>("Flow Rate Type") == "From CISM")) {
      // The field *should* already be specified as an 'Elem Scalar' required field in the mesh.
      // Interpolate ice softness (aka, flow_factor) from nodes to cell
      // ev = evalUtils.getPSTUtils().constructNodesToCellInterpolationEvaluator ("flow_factor",false);
      // fm0.template registerEvaluator<EvalT> (ev);
    } else {
      p = Teuchos::rcp(new Teuchos::ParameterList("LandIce FlowRate"));

      //Input
      if (viscosity_use_corrected_temperature) {
        p->set<std::string>("Temperature Variable Name", corrected_temperature_name);
      } else {
        // Avoid pointless calculation, and use original temperature in viscosity calculation
        p->set<std::string>("Temperature Variable Name", temperature_name);
      }
      p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Viscosity"));

      //Output
      p->set<std::string>("Flow Rate Variable Name", flow_factor_name);

      ev = Teuchos::rcp(new LandIce::FlowRate<EvalT,PHAL::AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  //--- LandIce viscosity ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Viscosity"));

  //Input
  p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name);
  p->set<std::string>("Velocity QP Variable Name", dof_names[0]);
  p->set<std::string>("Velocity Gradient QP Variable Name", dof_names[0] + " Gradient");
  if (viscosity_use_corrected_temperature) {
    p->set<std::string>("Temperature Variable Name", corrected_temperature_name);
  } else {
    // Avoid pointless calculation, and use original temperature in viscosity calculation
    p->set<std::string>("Temperature Variable Name", temperature_name);
  }
  p->set<std::string>("Ice Softness Variable Name", flow_factor_name);
  p->set<std::string>("Stiffening Factor QP Name", stiffening_factor_name);
  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Viscosity"));
  p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

  //Output
  p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
  p->set<std::string>("EpsilonSq QP Variable Name", "LandIce EpsilonSq");

  // The st of T for viscosity is complicated: you need to get the st of the T used (temp or corrected temp),
  // and consider that you do NodesToCell interp, which introduces MeshScalar type in the result.
  FieldScalarType viscosity_temp_st = FieldScalarType::MeshScalar | field_scalar_type[(viscosity_use_corrected_temperature ? corrected_temperature_name : temperature_name)];
  ev = createEvaluatorWithTwoScalarTypes<LandIce::ViscosityFO,EvalT>(p,dl,FieldScalarType::Scalar,viscosity_temp_st);
  fm0.template registerEvaluator<EvalT>(ev);

  // --- Print LandIce Dissipation ---
  if(compute_dissipation) {
    // LandIce Dissipation
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Dissipation"));

    //Input
    p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
    p->set<std::string>("EpsilonSq QP Variable Name", "LandIce EpsilonSq");
    p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

    //Output
    p->set<std::string>("Dissipation QP Variable Name", "LandIce Dissipation");

    ev = Teuchos::rcp(new LandIce::Dissipation<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    ev = evalUtils.getPSTUtils().constructQuadPointsToCellInterpolationEvaluator("LandIce Dissipation");
    fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructQuadPointsToCellInterpolationEvaluator("LandIce Dissipation"));

    // Saving the dissipation heat in the output mesh
    std::string stateName = "dissipation_heat";
    auto entity = Albany::StateStruct::ElemData;
    p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, meshSpecs.ebName, true, &entity);
    p->set<std::string>("Field Name", "LandIce Dissipation");
    p->set<std::string>("Weights Name",Albany::weights_name);
    p->set("Weights Layout", dl->qp_scalar);
    p->set("Field Layout", dl->cell_scalar2);
    p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
      // Only PHAL::AlbanyTraits::Residual evaluates something
      if (ev->evaluatedFields().size()>0)
      {
        // Require save friction heat
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
      }
    }
  }

  // Saving the stress tensor in the output mesh
  if(params->get<bool>("Print Stress Tensor", false))
  {
    // Interpolate stress tensor, from qps to a single cell scalar
    ev = evalUtils.constructQuadPointsToCellInterpolationEvaluator("Stress Tensor", dl->qp_tensor, dl->cell_tensor);
    fm0.template registerEvaluator<EvalT> (ev);

    // Save stress tensor (if needed)
    std::string stateName = "Stress Tensor";
    auto entity = Albany::StateStruct::ElemData;
    p = stateMgr.registerStateVariable(stateName, dl->cell_tensor, meshSpecs.ebName, true, &entity);
    p->set< Teuchos::RCP<PHX::DataLayout> >("State Field Layout",dl->cell_tensor);
    p->set<std::string>("Field Name", "Stress Tensor");
    p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    if (fieldManagerChoice == Albany::BUILD_RESID_FM)
    {
      // Only PHAL::AlbanyTraits::Residual evaluates something
      if (ev->evaluatedFields().size()>0)
      {
        // Require save friction heat
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
      }
    }
  }

#ifdef CISM_HAS_LANDICE
  //--- LandIce surface gradient from CISM ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Surface Gradient"));

  //Input
  p->set<std::string>("CISM Surface Height Gradient X Variable Name", "xgrad_surface_height");
  p->set<std::string>("CISM Surface Height Gradient Y Variable Name", "ygrad_surface_height");
  p->set<std::string>("BF Variable Name", Albany::bf_name);

  //Output
  p->set<std::string>("Surface Height Gradient QP Variable Name", "CISM Surface Height Gradient");
  ev = Teuchos::rcp(new LandIce::CismSurfaceGradFO<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);
#endif

  //--- Body Force ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Body Force"));

  //Input
  p->set<std::string>("LandIce Viscosity QP Variable Name", "LandIce Viscosity");
#ifdef CISM_HAS_LANDICE
  p->set<std::string>("Surface Height Gradient QP Variable Name", "CISM Surface Height Gradient");
#endif
  p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name);
  p->set<std::string>("Surface Height Gradient Name", surface_height_name + " Gradient");
  p->set<std::string>("Surface Height Name", surface_height_name);
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Body Force"));
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

  //Output
  p->set<std::string>("Body Force Variable Name", body_force_name);

  ev = Teuchos::rcp(new LandIce::StokesFOBodyForce<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if (fieldManagerChoice == Albany::BUILD_RESID_FM) {

    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> res_tag(scatter_names[0], dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }

  // ---------- Add time as a Sacado-ized parameter (only if specified) ------- //
  bool isTimeAParameter = false;
  if (params->isParameter("Use Time Parameter")) isTimeAParameter = params->get<bool>("Use Time Parameter");
  if (isTimeAParameter) {
    p = Teuchos::rcp(new Teuchos::ParameterList("Time"));
    p->set<Teuchos::RCP<PHX::DataLayout>>("Workset Scalar Data Layout", dl->workset_scalar);
    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);
    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
    ev = Teuchos::rcp(new LandIce::Time<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Time", dl->workset_scalar, dl->dummy, meshSpecs.ebName, "scalar", 0.0, true);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }
}

template <typename EvalT>
void StokesFOBase::constructBasalBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  std::string param_name;

  for (auto pl : landice_bcs[LandIceBC::BasalFriction]) {
    const std::string& ssName = pl->get<std::string>("Side Set Name");

    auto dl_side = dl->side_layouts.at(ssName);

    // We may have more than 1 basal side set. The layout of all the side fields is the
    // same, so we need to differentiate them by name (just like we do for the basis functions already).

    std::string velocity_side_name = dof_names[0] + "_" + ssName;
    std::string sliding_velocity_side_name = "sliding_velocity_" + ssName;
    std::string beta_side_name = "beta_" + ssName;
    std::string ice_thickness_side_name = ice_thickness_name + "_" + ssName;
    std::string ice_overburden_side_name = "ice_overburden_" + ssName;
    std::string effective_pressure_side_name = "effective_pressure_" + ssName;
    std::string bed_roughness_side_name = "bed_roughness_" + ssName;
    std::string mu_coulomb_side_name = "mu_coulomb_" + ssName;
    std::string mu_power_law_side_name = "mu_power_law_" + ssName;
    std::string bed_topography_side_name = bed_topography_name + "_" + ssName;
    std::string flow_factor_side_name = flow_factor_name +"_" + ssName;

    // -------------------------------- LandIce evaluators ------------------------- //

    // --- Basal Residual --- //
    p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Basal Residual"));

    //Input
    p->set<std::string>("BF Side Name", Albany::bf_name + " "+ssName);
    p->set<std::string>("Weighted Measure Name", Albany::weighted_measure_name + " "+ssName);
    p->set<std::string>("Basal Friction Coefficient Side QP Variable Name", beta_side_name);
    p->set<std::string>("Velocity Side QP Variable Name", velocity_side_name);
    p->set<std::string>("Side Set Name", ssName);
    p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<Teuchos::ParameterList*>("Parameter List", &pl->sublist("Basal Friction Coefficient"));

    //Output
    p->set<std::string>("Residual Variable Name", resid_names[0]);

    ev = Teuchos::rcp(new LandIce::StokesFOBasalResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Sliding velocity calculation at nodes ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Velocity Norm"));

    // Input
    p->set<std::string>("Field Name",velocity_side_name);
    p->set<std::string>("Field Layout","Cell Side Node Vector");
    p->set<std::string>("Side Set Name", ssName);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Field Norm"));

    // Output
    p->set<std::string>("Field Norm Name",sliding_velocity_side_name);

    ev = Teuchos::rcp(new PHAL::FieldFrobeniusNorm<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Sliding velocity calculation ---//
    p->set<std::string>("Field Layout","Cell Side QuadPoint Vector");
    ev = Teuchos::rcp(new PHAL::FieldFrobeniusNorm<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Ice Overburden (QPs) ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Effective Pressure Surrogate"));

    // Input
    p->set<bool>("Nodal",false);
    p->set<std::string>("Side Set Name", ssName);
    p->set<std::string>("Ice Thickness Variable Name", ice_thickness_side_name);
    p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

    // Output
    p->set<std::string>("Ice Overburden Variable Name", ice_overburden_side_name);

    ev = Teuchos::rcp(new LandIce::IceOverburden<EvalT,PHAL::AlbanyTraits,true>(*p,dl_side));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Ice Overburden (Nodes) ---//
    p->set<bool>("Nodal",true);
    ev = Teuchos::rcp(new LandIce::IceOverburden<EvalT,PHAL::AlbanyTraits,true>(*p,dl_side));
    fm0.template registerEvaluator<EvalT>(ev);

    // If we are given an effective pressure field, we don't need a surrogate model for it
    if (!(is_input_field["effective_pressure"] || is_ss_input_field[ssName]["effective_pressure"])) {
      //--- Effective pressure surrogate (QPs) ---//
      p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Effective Pressure Surrogate"));

      // Input
      p->set<bool>("Nodal",false);
      p->set<std::string>("Side Set Name", ssName);
      p->set<std::string>("Ice Overburden Variable Name", ice_overburden_side_name);

      // Output
      p->set<std::string>("Effective Pressure Variable Name", effective_pressure_side_name);

      ev = Teuchos::rcp(new LandIce::EffectivePressure<EvalT,PHAL::AlbanyTraits,true,true>(*p,dl_side));
      fm0.template registerEvaluator<EvalT>(ev);

      //--- Effective pressure surrogate (Nodes) ---//
      p->set<bool>("Nodal",true);
      ev = Teuchos::rcp(new LandIce::EffectivePressure<EvalT,PHAL::AlbanyTraits,true,true>(*p,dl_side));
      fm0.template registerEvaluator<EvalT>(ev);

      //--- Shared Parameter for basal friction coefficient: alpha ---//
      p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: alpha"));

      param_name = "Hydraulic-Over-Hydrostatic Potential Ratio";
      p->set<std::string>("Parameter Name", param_name);
      p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

      Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Alpha>> ptr_alpha;
      ptr_alpha = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Alpha>(*p,dl));
      ptr_alpha->setNominalValue(params->sublist("Parameters"),pl->sublist("Basal Friction Coefficient").get<double>(param_name,-1.0));
      fm0.template registerEvaluator<EvalT>(ptr_alpha);
    }

    //--- Shared Parameter for basal friction coefficient: lambda ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: lambda"));

    param_name = "Bed Roughness";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Lambda>> ptr_lambda;
    ptr_lambda = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Lambda>(*p,dl));
    ptr_lambda->setNominalValue(params->sublist("Parameters"),pl->sublist("Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_lambda);

    //--- Shared Parameter for basal friction coefficient: muCoulomb ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: muCoulomb"));

    param_name = "Coulomb Friction Coefficient";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::MuCoulomb>> ptr_muC;
    ptr_muC = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::MuCoulomb>(*p,dl));
    ptr_muC->setNominalValue(params->sublist("Parameters"),pl->sublist("Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_muC);

    //--- Shared Parameter for basal friction coefficient: muPowerLaw ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: muPowerLaw"));

    param_name = "Power Law Coefficient";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::MuPowerLaw>> ptr_muP;
    ptr_muP = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::MuPowerLaw>(*p,dl));
    ptr_muP->setNominalValue(params->sublist("Parameters"),pl->sublist("Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_muP);

    //--- Shared Parameter for basal friction coefficient: power ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: power"));

    param_name = "Power Exponent";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Power>> ptr_power;
    ptr_power = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Power>(*p,dl));
    ptr_power->setNominalValue(params->sublist("Parameters"),pl->sublist("Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_power);

    //--- LandIce basal friction coefficient ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Basal Friction Coefficient"));

    //Input
    p->set<std::string>("Sliding Velocity Variable Name", sliding_velocity_side_name);
    p->set<std::string>("BF Variable Name", Albany::bf_name + " " + ssName);
    p->set<std::string>("Effective Pressure QP Variable Name", effective_pressure_side_name);
    p->set<std::string>("Ice Softness Variable Name", flow_factor_side_name);
    p->set<std::string>("Bed Roughness Variable Name", bed_roughness_side_name);
    p->set<std::string>("Coulomb Friction Coefficient Variable Name", mu_coulomb_side_name);
    p->set<std::string>("Power Law Coefficient Variable Name", mu_power_law_side_name);
    p->set<std::string>("Side Set Name", ssName);
    p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name + " " + ssName);
    p->set<Teuchos::ParameterList*>("Parameter List", &pl->sublist("Basal Friction Coefficient"));
    p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));
    p->set<Teuchos::ParameterList*>("Viscosity Parameter List", &params->sublist("LandIce Viscosity"));
    p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
    p->set<std::string>("Bed Topography Variable Name", bed_topography_side_name);
    p->set<std::string>("Effective Pressure Variable Name", effective_pressure_side_name);
    p->set<std::string>("Ice Thickness Variable Name", ice_thickness_side_name);
    p->set<bool>("Is Thickness A Parameter",is_dist_param[ice_thickness_name]);
    p->set<Teuchos::RCP<std::map<std::string,bool>>>("Dist Param Query Map",Teuchos::rcpFromRef(is_dist_param));

    //Output
    p->set<std::string>("Basal Friction Coefficient Variable Name", beta_side_name);

    std::string bft = util::upper_case(pl->sublist("Basal Friction Coefficient").get<std::string>("Type"));

    if (temperature_coupled) {
      if (hydrology_coupled) {
        ev = createEvaluatorWithThreeScalarTypes<LandIce::BasalFrictionCoefficient, EvalT>(p,dl_side, FieldScalarType::Scalar, FieldScalarType::Scalar,FieldScalarType::Scalar);
      } else {
        ev = createEvaluatorWithThreeScalarTypes<LandIce::BasalFrictionCoefficient, EvalT>(p,dl_side, FieldScalarType::ParamScalar, FieldScalarType::Scalar,FieldScalarType::Scalar);
      }
    } else {
      if (hydrology_coupled) {
        ev = createEvaluatorWithThreeScalarTypes<LandIce::BasalFrictionCoefficient, EvalT>(p,dl_side, FieldScalarType::Scalar, FieldScalarType::Scalar,FieldScalarType::ParamScalar);
      } else {
        ev = createEvaluatorWithThreeScalarTypes<LandIce::BasalFrictionCoefficient, EvalT>(p,dl_side, FieldScalarType::Real, FieldScalarType::Scalar,FieldScalarType::ParamScalar);
      }
    }
    fm0.template registerEvaluator<EvalT>(ev);

    //--- LandIce basal friction coefficient at nodes ---//
    p->set<bool>("Nodal",true);
    if (temperature_coupled) {
      if (hydrology_coupled) {
        ev = createEvaluatorWithThreeScalarTypes<LandIce::BasalFrictionCoefficient, EvalT>(p,dl_side, FieldScalarType::Scalar, FieldScalarType::Scalar,FieldScalarType::Scalar);
      } else {
        ev = createEvaluatorWithThreeScalarTypes<LandIce::BasalFrictionCoefficient, EvalT>(p,dl_side, FieldScalarType::ParamScalar, FieldScalarType::Scalar,FieldScalarType::Scalar);
      }
    } else {
      if (hydrology_coupled) {
        ev = createEvaluatorWithThreeScalarTypes<LandIce::BasalFrictionCoefficient, EvalT>(p,dl_side, FieldScalarType::Scalar, FieldScalarType::Scalar,FieldScalarType::ParamScalar);
      } else {
        ev = createEvaluatorWithThreeScalarTypes<LandIce::BasalFrictionCoefficient, EvalT>(p,dl_side, FieldScalarType::Real, FieldScalarType::Scalar,FieldScalarType::ParamScalar);
      }
    }
    fm0.template registerEvaluator<EvalT>(ev);
  }
}

template <typename EvalT>
void StokesFOBase::constructLateralBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  for (auto pl : landice_bcs[LandIceBC::Lateral]) {
    const std::string& ssName = pl->get<std::string>("Side Set Name");

    // We may have more than 1 lateral side set. The layout of all the side fields is the
    // same, so we need to differentiate them by name (just like we do for the basis functions already).

    std::string ice_thickness_side_name = ice_thickness_name + "_" + ssName;
    std::string surface_height_side_name = surface_height_name + "_" + ssName;

    // -------------------------------- LandIce evaluators ------------------------- //

    // Lateral residual
    p = Teuchos::rcp( new Teuchos::ParameterList("Lateral Residual") );

    // Input
    p->set<std::string>("Ice Thickness Variable Name", ice_thickness_side_name);
    p->set<std::string>("Ice Surface Elevation Variable Name", surface_height_side_name);
    p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name + " " + ssName);
    p->set<std::string>("BF Side Name", Albany::bf_name + " " + ssName);
    p->set<std::string>("Weighted Measure Name", Albany::weighted_measure_name + " " + ssName);
    p->set<std::string>("Side Normal Name", Albany::normal_name + " " + ssName);
    p->set<std::string>("Side Set Name", ssName);
    p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
    p->set<Teuchos::ParameterList*>("Lateral BC Parameters",pl.get());
    p->set<Teuchos::ParameterList*>("Physical Parameters",&params->sublist("LandIce Physical Parameters"));
    p->set<Teuchos::ParameterList*>("Stereographic Map",&params->sublist("Stereographic Map"));

    // Output
    p->set<std::string>("Residual Variable Name", resid_names[0]);

    ev = createEvaluatorWithOneScalarType<LandIce::StokesFOLateralResid,EvalT>(p,dl,field_scalar_type[ice_thickness_name]);
    fm0.template registerEvaluator<EvalT>(ev);
  }
}

template <typename EvalT>
void StokesFOBase::constructSurfaceVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  if (!isInvalid(surfaceSideName)) {
    auto dl_side = dl->side_layouts.at(surfaceSideName);

    //--- LandIce noise (for synthetic inverse problem) ---//
    if (params->sublist("LandIce Noise").isSublist("Observed Surface Velocity"))
    {
      // ---- Add noise to the measures ---- //
      p = Teuchos::rcp(new Teuchos::ParameterList("Noisy Observed Velocity"));

      //Input
      p->set<std::string>("Field Name", "observed_surface_velocity");
      p->set<Teuchos::RCP<PHX::DataLayout>>("Field Layout", dl_side->qp_vector);
      p->set<Teuchos::ParameterList*>("PDF Parameters", &params->sublist("LandIce Noise").sublist("Observed Surface Velocity"));

      // Output
      p->set<std::string>("Noisy Field Name", "observed_surface_velocity_noisy");

      ev = Teuchos::rcp(new PHAL::AddNoiseParam<EvalT,PHAL::AlbanyTraits> (*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Surface Velocity Mismatch may require the gradient of the basal friction coefficient as a regularization.
    for (auto pl : landice_bcs[LandIceBC::BasalFriction]) {
      std::string ssName  = pl->get<std::string>("Side Set Name");

      std::string velocity_side_name = dof_names[0] + "_" + ssName;
      std::string velocity_gradient_side_name = dof_names[0] + "_" + ssName  + " Gradient";
      std::string sliding_velocity_side_name = "sliding_velocity_" + ssName;
      std::string beta_side_name = "beta_" + ssName;
      std::string beta_gradient_side_name = "beta_" + ssName + " Gradient";
      std::string effective_pressure_side_name = "effective_pressure_" + ssName;
      std::string effective_pressure_gradient_side_name = "effective_pressure_" + ssName + " Gradient";

      //--- LandIce basal friction coefficient gradient ---//
      p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Basal Friction Coefficient Gradient"));
  
      // Input
      p->set<std::string>("Gradient BF Side Variable Name", Albany::grad_bf_name + " "+ssName);
      p->set<std::string>("Side Set Name", ssName);
      p->set<std::string>("Effective Pressure QP Name", effective_pressure_side_name);
      p->set<std::string>("Effective Pressure Gradient QP Name", effective_pressure_gradient_side_name);
      p->set<std::string>("Basal Velocity QP Name", velocity_side_name);
      p->set<std::string>("Basal Velocity Gradient QP Name", velocity_gradient_side_name);
      p->set<std::string>("Sliding Velocity QP Name", sliding_velocity_side_name);
      p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name +" "+ssName);
      p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
      p->set<Teuchos::ParameterList*>("Parameter List", &pl->sublist("Basal Friction Coefficient"));
      p->set<Teuchos::RCP<std::map<std::string,bool>>>("Dist Param Query Map",Teuchos::rcpFromRef(is_dist_param));
  
      // Output
      p->set<std::string>("Basal Friction Coefficient Gradient Name",beta_gradient_side_name);
  
      ev = Teuchos::rcp(new LandIce::BasalFrictionCoefficientGradient<EvalT,PHAL::AlbanyTraits>(*p,dl->side_layouts.at(ssName)));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }
}

template <typename EvalT>
void StokesFOBase::constructSMBEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                           const Albany::MeshSpecsStruct& meshSpecs)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // Evaluators needed for thickness-related diagnostics (e.g., SMB)
  if (!isInvalid(basalSideName)) {
    auto dl_side = dl->side_layouts.at(basalSideName);

    std::string basalSideNamePlanar = basalSideName + "_planar";

    {
      //---- Compute side basis functions
      auto ss_util_needed = ss_utils_needed[basalSideName];
      if (ss_util_needed[UtilityRequest::BFS] || ss_util_needed[UtilityRequest::NORMALS]) {
        // BF, GradBF, w_measure, Tangents, Metric, Metric Det, Inverse Metric
        ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis[basalSideName], sideCubature[basalSideName],
            basalSideName, false, true);
        fm0.template registerEvaluator<EvalT> (ev);
      }
    }

    // We may have more than 1 basal side set. 'basalSideName' should be the union of all of them.
    // However, some of the fields used here, may be used also to compute quantities defined on
    // only some of the sub-sidesets of 'basalSideName'. The layout of all the side fields is the
    // same, so we need to differentiate them by name (just like we do for the basis functions already).

    std::string velocity_side_name = dof_names[0] + "_" + basalSideName;
    std::string ice_thickness_side_name = ice_thickness_name + "_" + basalSideName;
    std::string ice_thickness_side_name_planar = ice_thickness_name + "_" + basalSideNamePlanar;
    std::string surface_height_side_name = surface_height_name + "_" + basalSideName;
    std::string apparent_mass_balance_side_name = "apparent_mass_balance_" + basalSideName;
    std::string apparent_mass_balance_RMS_side_name = "apparent_mass_balance_RMS_" + basalSideName;
    std::string stiffening_factor_side_name = stiffening_factor_name + "_" + basalSideName;
    std::string effective_pressure_side_name = effective_pressure_name + "_" + basalSideName;
    std::string vertically_averaged_velocity_side_name = vertically_averaged_velocity_name + "_" + basalSideName;
    std::string bed_roughness_side_name = "bed_roughness_" + basalSideName;

    // ------------------- Interpolations and utilities ------------------ //

    // if (is_dist_param[ice_thickness_name])
    // {
    //   //---- Restrict ice thickness from cell-based to cell-side-based
    //   ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator(ice_thickness_name,basalSideName,"Node Scalar",cellType,ice_thickness_side_name);
    //   fm0.template registerEvaluator<EvalT> (ev);
    // }

    // -------------------------------- LandIce evaluators ------------------------- //

    {

      std::map<FieldScalarType,Teuchos::RCP<const Albany::EvaluatorUtilsBase<PHAL::AlbanyTraits>>> utils_map;
      utils_map[FieldScalarType::Scalar]      = Teuchos::rcpFromRef(evalUtils.getSTUtils());
      utils_map[FieldScalarType::ParamScalar] = Teuchos::rcpFromRef(evalUtils.getPSTUtils());
      utils_map[FieldScalarType::MeshScalar]  = Teuchos::rcpFromRef(evalUtils.getMSTUtils());
      utils_map[FieldScalarType::Real]        = Teuchos::rcpFromRef(evalUtils.getRTUtils());

      // If there's no information about this field, we assume it is not needed, so we skip it.
      // If it WAS indeed needed, Phalanx DAG will miss a node, and an exception will be thrown.
      if (field_scalar_type.find(ice_thickness_name)!=field_scalar_type.end()) {

        // Get the right evaluator utils for this field.
        const FieldScalarType st = field_scalar_type.at(ice_thickness_name);
        const auto& utils = *utils_map.at(st);

        ev = utils.constructDOFGradInterpolationSideEvaluator (ice_thickness_side_name, basalSideName, true);
        fm0.template registerEvaluator<EvalT> (ev);
      }
    }

    // Vertically averaged velocity
    p = Teuchos::rcp(new Teuchos::ParameterList("Gather Averaged Velocity"));

    p->set<std::string>("Contracted Solution Name", vertically_averaged_velocity_name + "_" + basalSideName);
    p->set<std::string>("Mesh Part", "basalside");
    p->set<int>("Solution Offset", dof_offsets[0]);
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<bool>("Is Vector", true);
    p->set<std::string>("Contraction Operator", "Vertical Average");


    p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));

    ev = Teuchos::rcp(new GatherVerticallyContractedSolution<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    if(!params->isSublist("LandIce Flux Divergence") ||
       !params->sublist("LandIce Flux Divergence").get<bool>("Flux Divergence Is Part Of Solution")){
      // Flux divergence
      p = Teuchos::rcp(new Teuchos::ParameterList("Flux Divergence"));

      //Input
      p->set<std::string>("Averaged Velocity Side QP Variable Name", vertically_averaged_velocity_side_name);
      p->set<std::string>("Averaged Velocity Side QP Divergence Name", vertically_averaged_velocity_side_name + " Divergence");
      p->set<std::string>("Thickness Side QP Variable Name", ice_thickness_side_name);
      p->set<std::string>("Thickness Gradient Name", ice_thickness_side_name + " Planar Gradient");
      p->set<std::string>("Side Tangents Name", Albany::tangents_name + " " + basalSideNamePlanar);

      p->set<std::string>("Field Name",  "flux_divergence_basalside");
      p->set<std::string>("Side Set Name", basalSideName);

      ev = createEvaluatorWithOneScalarType<LandIce::FluxDiv,EvalT>(p,dl_side,field_scalar_type[ice_thickness_name]);
      fm0.template registerEvaluator<EvalT>(ev);

      // --- 2D divergence of Averaged Velocity ---- //
      p = Teuchos::rcp(new Teuchos::ParameterList("DOF Div Interpolation Side Averaged Velocity"));

      // Input
      p->set<std::string>("Variable Name", vertically_averaged_velocity_side_name);
      p->set<std::string>("Gradient BF Name", Albany::grad_bf_name + " "+basalSideNamePlanar);
      p->set<std::string>("Tangents Name", "Tangents "+basalSideNamePlanar);
      p->set<std::string>("Side Set Name",basalSideName);

      // Output (assumes same Name as input)
      p->set<std::string>("Divergence Variable Name", vertically_averaged_velocity_side_name  + " Divergence");

      ev = Teuchos::rcp(new LandIce::DOFDivInterpolationSide<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }
}

template<typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
StokesFOBase::constructStokesFOBaseResponsesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                                        const Albany::MeshSpecsStruct& meshSpecs,
                                                        Albany::StateManager& stateMgr,
                                                        Albany::FieldManagerChoice fieldManagerChoice,
                                                        const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {

    // --- SurfaceVelocity-related evaluators (if needed) --- //
    constructSurfaceVelocityEvaluators<EvalT> (fm0);

    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));

    // Figure out if observed surface velocity RMS is scalar (if present at all)
    if (!isInvalid(surfaceSideName)) {
      if (is_ss_input_field[surfaceSideName]["observed_surface_velocity_RMS"]) {
        if (field_rank.at("observed_surface_velocity_RMS")==0) {
          paramList->set<bool>("Scalar RMS",true);
        }
      }
    }

    // ----------------------- Responses --------------------- //
    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
    paramList->set<Teuchos::ParameterList>("LandIce Physical Parameters List", params->sublist("LandIce Physical Parameters"));
    paramList->set<Teuchos::RCP<std::map<std::string, int>>> ("Extruded Params Levels", Teuchos::rcpFromRef(extruded_params_levels));
    paramList->set<std::string>("Coordinate Vector Side Variable Name", Albany::coord_vec_name + " " + basalSideName);
    paramList->set<std::string>("Basal Friction Coefficient Name","beta");
    paramList->set<std::string>("Stiffening Factor Gradient Name",stiffening_factor_name + "_" + basalSideName + " Gradient");
    paramList->set<std::string>("Stiffening Factor Name", stiffening_factor_name + "_" + basalSideName);
    paramList->set<std::string>("Thickness Side Variable Name",ice_thickness_name + "_" + basalSideName);
    paramList->set<std::string>("Bed Topography Side Variable Name",bed_topography_name + "_" + basalSideName);
    paramList->set<std::string>("Surface Velocity Side QP Variable Name",dof_names[0] + "_" + surfaceSideName);
    paramList->set<std::string>("Averaged Vertical Velocity Side Variable Name",vertically_averaged_velocity_name + "_" + basalSideName);
    paramList->set<std::string>("Observed Surface Velocity Side QP Variable Name","observed_surface_velocity_" + surfaceSideName);
    paramList->set<std::string>("Observed Surface Velocity RMS Side QP Variable Name","observed_surface_velocity_RMS_" + surfaceSideName);
    paramList->set<std::string>("Flux Divergence Side QP Variable Name","flux_divergence_basalside");
    paramList->set<std::string>("Thickness RMS Side QP Variable Name","observed_ice_thickness_RMS_" + basalSideName);
    paramList->set<std::string>("Observed Thickness Side QP Variable Name","observed_ice_thickness_" + basalSideName);
    paramList->set<std::string>("SMB Side QP Variable Name","apparent_mass_balance_" + basalSideName);
    paramList->set<std::string>("SMB RMS Side QP Variable Name","apparent_mass_balance_RMS_" + basalSideName);
    paramList->set<std::string>("Thickness Gradient Name", ice_thickness_name + "_" + basalSideName + " Planar Gradient");
    paramList->set<std::string>("Thickness Side QP Variable Name",ice_thickness_name + "_" + basalSideName);
    paramList->set<std::string>("Basal Side Name", basalSideName);
    paramList->set<std::string>("Weighted Measure Basal Name",Albany::weighted_measure_name + " " + basalSideName);
    paramList->set<std::string>("Weighted Measure Surface Name",Albany::weighted_measure_name + " " + surfaceSideName);
    paramList->set<std::string>("Metric 2D Name",Albany::metric_name + " " + basalSideName);
    paramList->set<std::string>("Metric Basal Name",Albany::metric_name + " " + basalSideName);
    paramList->set<std::string>("Metric Surface Name",Albany::metric_name + " " + surfaceSideName);
    paramList->set<std::string>("Basal Side Tangents Name",Albany::tangents_name + " " + basalSideName);
    paramList->set<std::string>("Weighted Measure 2D Name",Albany::weighted_measure_name + " " + basalSideName + "_planar");
    paramList->set<std::string>("Inverse Metric Basal Name",Albany::metric_inv_name + " " + basalSideName);
    paramList->set<std::string>("Surface Side Name", surfaceSideName);
    paramList->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));
    paramList->set<std::vector<Teuchos::RCP<Teuchos::ParameterList>>*>("Basal Regularization Params",&landice_bcs[LandIceBC::BasalFriction]);
    paramList->set<std::string>("Ice Thickness Scalar Type",e2str(field_scalar_type[ice_thickness_name]));

    LandIce::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

template<typename T>
std::string StokesFOBase::
print_map_keys (const std::map<std::string,T>& map) {
  std::string s;
  for (auto it : map) {
    s += " ";
    s += it.first;
  }
  return s;
}

} // namespace LandIce

#endif // LANDICE_STOKES_FO_BASE_HPP
