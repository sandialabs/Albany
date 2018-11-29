//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_FO_BASE_HPP
#define LANDICE_STOKES_FO_BASE_HPP

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
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "LandIce_BasalFrictionCoefficient.hpp"
#include "LandIce_BasalFrictionCoefficientGradient.hpp"
#include "LandIce_DOFDivInterpolationSide.hpp"
#include "LandIce_EffectivePressure.hpp"
#include "LandIce_FlowRate.hpp"
#include "LandIce_PressureCorrectedTemperature.hpp"
#include "LandIce_GatherVerticallyAveragedVelocity.hpp"
#include "LandIce_FluxDiv.hpp"
#include "LandIce_IceOverburden.hpp"
#include "LandIce_ParamEnum.hpp"
#include "LandIce_ProblemUtils.hpp"
#include "LandIce_ProblemUtils.hpp"
#include "LandIce_SharedParameter.hpp"
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
                                        Albany::FieldManagerChoice fieldManagerChoice,
                                        std::map<std::string, int>& extruded_params_levels,
                                        std::map<std::string,bool>& is_dist_param,
                                        std::map<std::string,bool>&& has_input = {},
                                        std::map<std::string,std::map<std::string,bool>>&& has_ss_input = {});

  template <typename EvalT>
  void constructStatesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                  const Albany::MeshSpecsStruct& meshSpecs,
                                  Albany::StateManager& stateMgr,
                                  std::map<std::string, int>& extruded_params_levels,
                                  std::map<std::string,bool>& is_dist_param,
                                  std::map<std::string,bool>& has_input,
                                  std::map<std::string,std::map<std::string,bool>>& has_ss_input);

  template <typename EvalT>
  void constructVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                    const Albany::MeshSpecsStruct& meshSpecs,
                                    Albany::StateManager& stateMgr,
                                    Albany::FieldManagerChoice fieldManagerChoice,
                                    std::map<std::string,bool>& is_dist_param);

  template <typename EvalT>
  void constructBasalBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                   std::map<std::string,bool>& is_dist_param,
                                   std::map<std::string,bool>& has_input,
                                   std::map<std::string,std::map<std::string,bool>>& has_ss_input);

  template <typename EvalT>
  void constructLateralBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  template <typename EvalT>
  void constructSurfaceVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                           const Albany::MeshSpecsStruct& meshSpecs,
                                           Albany::StateManager& stateMgr);

  template <typename EvalT>
  void constructSMBEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                               const Albany::MeshSpecsStruct& meshSpecs,
                               std::map<std::string,bool>& is_dist_param);

  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructStokesFOBaseResponsesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                            const Albany::MeshSpecsStruct& meshSpecs,
                                            Albany::StateManager& stateMgr,
                                            Albany::FieldManagerChoice fieldManagerChoice,
                                            std::map<std::string,bool>& is_dist_param,
                                            const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  virtual void constructDirichletEvaluators (const Albany::MeshSpecsStruct& /* meshSpecs */) {}
  virtual void constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecsStruct>& /* meshSpecs */) {}

  Teuchos::RCP<Teuchos::ParameterList>
  getStokesFOBaseProblemParameters () const;

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
  Teuchos::ArrayRCP<std::string> resid_names;
  Teuchos::ArrayRCP<std::string> scatter_names;

  int numDim;
  int vecDimFO;
  int offsetVelocity;

  /// Boolean marking whether SDBCs are used
  bool use_sdbcs_;
};

template <typename EvalT>
void StokesFOBase::
constructStokesFOBaseEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                 const Albany::MeshSpecsStruct& meshSpecs,
                                 Albany::StateManager& stateMgr,
                                 Albany::FieldManagerChoice fieldManagerChoice,
                                 std::map<std::string, int>& extruded_params_levels,
                                 std::map<std::string,bool>& is_dist_param,
                                 std::map<std::string,bool>&& has_input,
                                 std::map<std::string,std::map<std::string,bool>>&& has_ss_input)
{
  // --- States/parameters --- //
  constructStatesEvaluators<EvalT> (fm0, meshSpecs, stateMgr, extruded_params_levels, is_dist_param, has_input, has_ss_input);

  // --- Velocity evaluators --- //
  constructVelocityEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice, is_dist_param);

  // --- Lateral BC evaluators (if needed) --- //
  constructLateralBCEvaluators<EvalT> (fm0);

  // --- Basal BC evaluators (if needed) --- //
  constructBasalBCEvaluators<EvalT> (fm0, is_dist_param, has_input, has_ss_input);
}

template <typename EvalT>
void StokesFOBase::constructStatesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                              const Albany::MeshSpecsStruct& meshSpecs,
                                              Albany::StateManager& stateMgr,
                                              std::map<std::string, int>& extruded_params_levels,
                                              std::map<std::string,bool>& is_dist_param,
                                              std::map<std::string,bool>& has_input,
                                              std::map<std::string,std::map<std::string,bool>>& has_ss_input)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Temporary variables used numerous times below
  Albany::StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // ---------------------------- Registering state variables ------------------------- //

  std::string stateName, fieldName, param_name;

  // Getting the names of the distributed parameters (they won't have to be loaded as states)
  std::map<std::string,bool> is_dist;
  std::map<std::string,bool> save_sensitivities;
  std::map<std::string,std::string> dist_params_name_to_mesh_part;
  std::map<std::string,bool> is_extruded_param;
  if (this->params->isSublist("Distributed Parameters"))
  {
    Teuchos::ParameterList& dist_params_list =  this->params->sublist("Distributed Parameters");
    Teuchos::ParameterList* param_list;
    int numParams = dist_params_list.get<int>("Number of Parameter Vectors",0);
    for (int p_index=0; p_index< numParams; ++p_index)
    {
      std::string parameter_sublist_name = Albany::strint("Distributed Parameter", p_index);
      if (dist_params_list.isSublist(parameter_sublist_name))
      {
        // The better way to specify dist params: with sublists
        param_list = &dist_params_list.sublist(parameter_sublist_name);
        param_name = param_list->get<std::string>("Name");
        dist_params_name_to_mesh_part[param_name] = param_list->get<std::string>("Mesh Part","");
        is_extruded_param[param_name] = param_list->get<bool>("Extruded",false);
        int extruded_param_level = param_list->get<int>("Extruded Param Level",0);
        extruded_params_levels.insert(std::make_pair(param_name, extruded_param_level));
        save_sensitivities[param_name]=param_list->get<bool>("Save Sensitivity",false);
      }
      else
      {
        // Legacy way to specify dist params: with parameter entries. Note: no mesh part can be specified.
        param_name = dist_params_list.get<std::string>(Albany::strint("Parameter", p_index));
        dist_params_name_to_mesh_part[param_name] = "";
      }
      is_dist_param[param_name] = true;
      is_dist[param_name] = true;
      is_dist[param_name+"_upperbound"] = true;
      dist_params_name_to_mesh_part[param_name+"_upperbound"] = dist_params_name_to_mesh_part[param_name];
      is_dist[param_name+"_lowerbound"] = true;
      dist_params_name_to_mesh_part[param_name+"_lowerbound"] = dist_params_name_to_mesh_part[param_name];
    }
  }

  //Dirichlet fields need to be distributed but they are not necessarily parameters.
  if (this->params->isSublist("Dirichlet BCs")) {
    Teuchos::ParameterList dirichlet_list = this->params->sublist("Dirichlet BCs");
    for(auto it = dirichlet_list.begin(); it !=dirichlet_list.end(); ++it) {
      std::string pname = dirichlet_list.name(it);
      if(dirichlet_list.isParameter(pname) && dirichlet_list.isType<std::string>(pname)){ //need to check, because pname could be the name sublist
        is_dist[dirichlet_list.get<std::string>(pname)]=true;
        dist_params_name_to_mesh_part[dirichlet_list.get<std::string>(pname)]="";
      }
    }
  }

  // Volume mesh requirements
  Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
  int num_fields = req_fields_info.get<int>("Number Of Fields",0);

  std::string fieldType, fieldUsage, meshPart;
  bool nodal_state;
  for (int ifield=0; ifield<num_fields; ++ifield)
  {
    Teuchos::ParameterList& thisFieldList = req_fields_info.sublist(Albany::strint("Field", ifield));

    // Get current state specs
    stateName  = fieldName = thisFieldList.get<std::string>("Field Name");
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
    }
    else if(fieldType == "Node Scalar") {
      entity = is_dist[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
      if(is_dist[stateName] && save_sensitivities[param_name])
        p = stateMgr.registerStateVariable(stateName + "_sensitivity", dl->node_scalar, meshSpecs.ebName, true, &entity, meshPart);
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, meshSpecs.ebName, true, &entity, meshPart);
      nodal_state = true;
    }
    else if(fieldType == "Elem Vector") {
      entity = Albany::StateStruct::ElemData;
      p = stateMgr.registerStateVariable(stateName, dl->cell_vector, meshSpecs.ebName, true, &entity, meshPart);
      nodal_state = false;
    }
    else if(fieldType == "Node Vector") {
      entity = is_dist[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(stateName, dl->node_vector, meshSpecs.ebName, true, &entity, meshPart);
      nodal_state = true;
    }

    // Do we need to save the state?
    if (fieldUsage == "Output" || fieldUsage == "Input-Output")
    {
      // An output: save it.
      p->set<bool>("Nodal State", nodal_state);
      ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
      if (ev->evaluatedFields().size()>0)
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }

    // Do we need to load/gather the state/parameter?
    if (is_dist_param[stateName])
    {
      // A parameter: gather it
      if (is_extruded_param[stateName])
      {
        ev = evalUtils.constructGatherScalarExtruded2DNodalParameter(stateName,fieldName);
        fm0.template registerEvaluator<EvalT>(ev);
      }
      else
      {
        ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
        fm0.template registerEvaluator<EvalT>(ev);
      }
      has_input[fieldName] = true;
    }
    else if (fieldUsage == "Input" || fieldUsage == "Input-Output")
    {
      // Not a parameter but still required as input: load it.
      p->set<std::string>("Field Name", fieldName);
      ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      has_input[fieldName] = true;
    }
  }

  // Side set requirements
  Teuchos::Array<std::string> ss_names;
  if (discParams->sublist("Side Set Discretizations").isParameter("Side Sets")) {
    ss_names = discParams->sublist("Side Set Discretizations").get<Teuchos::Array<std::string>>("Side Sets");
  } 
  for (int i=0; i<ss_names.size(); ++i)
  {
    const std::string& ss_name = ss_names[i];
    Teuchos::ParameterList& info = discParams->sublist("Side Set Discretizations").sublist(ss_name).sublist("Required Fields Info");
    num_fields = info.get<int>("Number Of Fields",0);
    Teuchos::RCP<PHX::DataLayout> dl_temp;
    Teuchos::RCP<PHX::DataLayout> sns;
    int numLayers;

    const std::string& sideEBName = meshSpecs.sideSetMeshSpecs.at(ss_name)[0]->ebName;
    Teuchos::RCP<Albany::Layouts> ss_dl = dl->side_layouts.at(ss_name);
    for (int ifield=0; ifield<num_fields; ++ifield)
    {
      Teuchos::ParameterList& thisFieldList =  info.sublist(Albany::strint("Field", ifield));

      // Get current state specs
      stateName  = thisFieldList.get<std::string>("Field Name");
      fieldName = stateName + "_" + ss_name;
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
      }
      else if(fieldType == "Node Scalar") {
        entity = is_dist[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->node_scalar, sideEBName, true, &entity, meshPart);
        nodal_state = true;

      }
      else if(fieldType == "Elem Vector") {
        entity = Albany::StateStruct::ElemData;
        p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->cell_vector, sideEBName, true, &entity, meshPart);
        nodal_state = false;
      }
      else if(fieldType == "Node Vector") {
        entity = is_dist[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->node_vector, sideEBName, true, &entity, meshPart);
        nodal_state = true;
      }
      else if(fieldType == "Elem Layered Scalar") {
        entity = Albany::StateStruct::ElemData;
        sns = ss_dl->cell_scalar2;
        numLayers = thisFieldList.get<int>("Number Of Layers");
        dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,LayerDim>(sns->dimension(0),sns->dimension(1),numLayers));
        stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
      }
      else if(fieldType == "Node Layered Scalar") {
        entity = is_dist[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        sns = ss_dl->node_scalar;
        numLayers = thisFieldList.get<int>("Number Of Layers");
        dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,LayerDim>(sns->dimension(0),sns->dimension(1),sns->dimension(2),numLayers));
        stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
      }
      else if(fieldType == "Elem Layered Vector") {
        entity = Albany::StateStruct::ElemData;
        sns = ss_dl->cell_vector;
        numLayers = thisFieldList.get<int>("Number Of Layers");
        dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Dim,LayerDim>(sns->dimension(0),sns->dimension(1),sns->dimension(2),numLayers));
        stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
      }
      else if(fieldType == "Node Layered Vector") {
        entity = is_dist[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        sns = ss_dl->node_vector;
        numLayers = thisFieldList.get<int>("Number Of Layers");
        dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,Dim,LayerDim>(sns->dimension(0),sns->dimension(1),sns->dimension(2),
                                                                               sns->dimension(3),numLayers));
        stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
      }

      // Creating load/save/gather evaluator(s)
      if (fieldUsage == "Output" || fieldUsage == "Input-Output")
      {
        // An output: save it.
        p->set<bool>("Nodal State", nodal_state);
        p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
        ev = Teuchos::rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,ss_dl));
        fm0.template registerEvaluator<EvalT>(ev);

        // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
        if (ev->evaluatedFields().size()>0)
          fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
      }

      if (is_dist_param[stateName])
      {
        // A parameter: gather it
        if (is_extruded_param[stateName])
        {
          ev = evalUtils.constructGatherScalarExtruded2DNodalParameter(stateName,fieldName);
          fm0.template registerEvaluator<EvalT>(ev);
        }
        else
        {
          ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
          fm0.template registerEvaluator<EvalT>(ev);
        }
        has_ss_input[ss_name][stateName] = true;
      }
      else if (fieldUsage == "Input" || fieldUsage == "Input-Output")
      {
        // Not a parameter but requires as input: load it.
        p->set<std::string>("Field Name", fieldName);
        ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
        has_ss_input[ss_name][stateName] = true;
      }
    }
  }
}

template <typename EvalT>
void StokesFOBase::
constructVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                             const Albany::MeshSpecsStruct& meshSpecs,
                             Albany::StateManager& stateMgr,
                             Albany::FieldManagerChoice fieldManagerChoice,
                             std::map<std::string,bool>& is_dist_param)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  const bool enableMemoizer = this->params->get<bool>("Use MDField Memoization", false);
  std::string param_name;

  // ------------------- Interpolations and utilities ------------------ //

  // Interpolate solution field
  ev = evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0],offsetVelocity);
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate solution gradient
  ev = evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0],offsetVelocity);
  fm0.template registerEvaluator<EvalT> (ev);

  // Map to physical frame
  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Compute basis funcitons
  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Intepolate surface height
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("surface_height");
  fm0.template registerEvaluator<EvalT> (ev);

  // Intepolate stiffening_factor
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("stiffening_factor");
  fm0.template registerEvaluator<EvalT> (ev);

  // Intepolate surface height gradient
  ev = evalUtils.getPSTUtils().constructDOFGradInterpolationEvaluator("surface_height");
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate temperature from nodes to cell
  ev = evalUtils.getPSTUtils().constructNodesToCellInterpolationEvaluator ("temperature",false);
  fm0.template registerEvaluator<EvalT> (ev);

  // Get coordinate of cell baricenter
  ev = evalUtils.getMSTUtils().constructQuadPointsToCellInterpolationEvaluator(Albany::coord_vec_name, dl->qp_gradient, dl->cell_gradient);
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate surface height to a field defined on cells (needed by PressureCorrectedTemperature)
  ev = evalUtils.getPSTUtils().constructQuadPointsToCellInterpolationEvaluator("surface_height");
  fm0.template registerEvaluator<EvalT> (ev);

  // -------------------------------- LandIce evaluators ------------------------- //

  // --- FO Stokes Stress --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Stress"));

  //Input
  p->set<std::string>("Velocity QP Variable Name", "Velocity");
  p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
  p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
  p->set<std::string>("Surface Height QP Name", "surface_height");
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
  p->set<std::string>("Velocity QP Variable Name", "Velocity");
  p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
  p->set<std::string>("Body Force Variable Name", "Body Force");
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

  Teuchos::RCP<LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>> ptr_homotopy;
  ptr_homotopy = Teuchos::rcp(new LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>(*p,dl));
  ptr_homotopy->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Viscosity").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_homotopy);

  // --- LandIce pressure-melting temperature
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Pressure Corrected Temperature"));

  //Input
  p->set<std::string>("Surface Height Variable Name", "surface_height");
  p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name);
  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));
  p->set<std::string>("Temperature Variable Name", "temperature");
  p->set<bool>("Enable Memoizer", enableMemoizer);

  //Output
  p->set<std::string>("Corrected Temperature Variable Name", "corrected temperature");

  ev = Teuchos::rcp(new LandIce::PressureCorrectedTemperature<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- LandIce Flow Rate ---//

  if(params->sublist("LandIce Viscosity").isParameter("Flow Rate Type")) {
    if((params->sublist("LandIce Viscosity").get<std::string>("Flow Rate Type") == "From File") ||
       (params->sublist("LandIce Viscosity").get<std::string>("Flow Rate Type") == "From CISM")) {
      // Interpolate ice softness (aka, flow_factor) from nodes to cell
      ev = evalUtils.getPSTUtils().constructNodesToCellInterpolationEvaluator ("flow_factor",false);
      fm0.template registerEvaluator<EvalT> (ev);
    } else {
      p = Teuchos::rcp(new Teuchos::ParameterList("LandIce FlowRate"));

      //Input
      if (params->sublist("LandIce Physical Parameters").isParameter("Clausius-Clapeyron Coefficient") &&
          params->sublist("LandIce Physical Parameters").get<double>("Clausius-Clapeyron Coefficient")!=0.0) {
        p->set<std::string>("Temperature Variable Name", "corrected temperature");
      } else {
        // Avoid pointless calculation, and use original temperature in viscosity calculation
        p->set<std::string>("Temperature Variable Name", "temperature");
      }
      p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Viscosity"));

      //Output
      p->set<std::string>("Flow Rate Variable Name", "flow_factor");

      ev = Teuchos::rcp(new LandIce::FlowRate<EvalT,PHAL::AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }


  //--- LandIce viscosity ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Viscosity"));

  //Input
  p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name);
  p->set<std::string>("Velocity QP Variable Name", "Velocity");
  p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
  if (params->sublist("LandIce Physical Parameters").isParameter("Clausius-Clapeyron Coefficient") &&
      params->sublist("LandIce Physical Parameters").get<double>("Clausius-Clapeyron Coefficient")!=0.0) {
    p->set<std::string>("Temperature Variable Name", "corrected temperature");
  } else {
    // Avoid pointless calculation, and use original temperature in viscosity calculation
    p->set<std::string>("Temperature Variable Name", "temperature");
  }
  p->set<std::string>("Ice Softness Variable Name", "flow_factor");
  p->set<std::string>("Stiffening Factor QP Name", "stiffening_factor");
  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Viscosity"));
  p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

  //Output
  p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
  p->set<std::string>("EpsilonSq QP Variable Name", "LandIce EpsilonSq");

  ev = Teuchos::rcp(new LandIce::ViscosityFO<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ParamScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- Print LandIce Dissipation ---
  if(params->sublist("LandIce Viscosity").get("Extract Strain Rate Sq", false))
  {
    // LandIce Dissipation
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Dissipation"));

    //Input
    p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
    p->set<std::string>("EpsilonSq QP Variable Name", "LandIce EpsilonSq");

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
    p->set<std::string>("Weights Name","Weights");
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
  p->set<std::string>("Surface Height Gradient Name", "surface_height Gradient");
  p->set<std::string>("Surface Height Name", "surface_height");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Body Force"));
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

  //Output
  p->set<std::string>("Body Force Variable Name", "Body Force");

  if (enableMemoizer) p->set<bool>("Enable Memoizer", enableMemoizer);

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
void StokesFOBase::constructBasalBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                               std::map<std::string,bool>& is_dist_param,
                                               std::map<std::string,bool>& has_input,
                                               std::map<std::string,std::map<std::string,bool>>& has_ss_input)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  const bool enableMemoizer = this->params->get<bool>("Use MDField Memoization", false);
  bool basalMemoizer;

  std::string param_name;

  for (auto pl : landice_bcs[LandIceBC::BasalFriction]) {
    const std::string& ssName = pl->get<std::string>("Side Set Name");

    auto dl_side = dl->side_layouts.at(ssName);

    // We may have more than 1 basal side set. The layout of all the side fields is the
    // same, so we need to differentiate them by name (just like we do for the basis functions already).

    std::string velocity_side = "velocity_" + ssName;
    std::string sliding_velocity_side = "sliding_velocity_" + ssName;
    std::string basal_friction_side = "basal_friction_" + ssName;
    std::string beta_side = "beta_" + ssName;
    std::string ice_thickness_side = "ice_thickness_" + ssName;
    std::string ice_overburden_side = "ice_overburden_" + ssName;
    std::string surface_height_side = "surface_height_" + ssName;
    std::string stiffening_factor_side = "stiffenting_factor_" + ssName;
    std::string effective_pressure_side = "effective_pressure_" + ssName;
    std::string bed_roughness_side = "bed_roughness_" + ssName;
    std::string bed_topography_side = "bed_topography_" + ssName;
    std::string flow_factor_side = "flow_factor_" + ssName;

    // ------------------- Interpolations and utilities ------------------ //

    //---- Restrict vertex coordinates from cell-based to cell-side-based
    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,ssName,"Vertex Vector",cellType,Albany::coord_vec_name +" " + ssName, enableMemoizer);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute side basis functions
    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis[ssName], sideCubature[ssName], ssName, enableMemoizer);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Restrict velocity from cell-based to cell-side-based
    ev = evalUtils.constructDOFCellToSideEvaluator(dof_names[0],ssName,"Node Vector",cellType,velocity_side);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity on QP on side
    ev = evalUtils.constructDOFVecInterpolationSideEvaluator(velocity_side, ssName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate velocity gradient on QP on side
    ev = evalUtils.constructDOFVecGradInterpolationSideEvaluator(velocity_side, ssName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Compute Quad Points coordinates on the side set
    ev = evalUtils.constructMapToPhysicalFrameSideEvaluator(cellType, sideCubature[ssName], ssName, enableMemoizer);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate stiffening gradient on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(stiffening_factor_side, ssName, enableMemoizer);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate stiffening gradient on QP on side
    ev = evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator(stiffening_factor_side, ssName, enableMemoizer);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate the 3D state on the side (the BasalFrictionCoefficient evaluator needs a side field)
    ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("stiffening_factor",ssName,"Node Scalar",cellType,stiffening_factor_side);
    fm0.template registerEvaluator<EvalT> (ev);

    // Intepolate bed_topography
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(bed_topography_side, ssName, enableMemoizer);
    fm0.template registerEvaluator<EvalT> (ev);

    // Intepolate effective_pressure
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(effective_pressure_side, ssName, enableMemoizer);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate thickness on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(ice_thickness_side, ssName, enableMemoizer);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate thickness gradient on QP on side
    ev = evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator(ice_thickness_side, ssName, enableMemoizer);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface height on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(surface_height_side, ssName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate basal_friction (if needed) on QP on side
    basalMemoizer = enableMemoizer ? (!is_dist_param["basal_friction"] ? true : false) : false;
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(basal_friction_side, ssName, basalMemoizer);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate bed_roughness (if needed) on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(bed_roughness_side, ssName);
    fm0.template registerEvaluator<EvalT>(ev);

    if (!has_ss_input[ssName]["surface_height"]) {
      //---- Restrict surface height from cell-based to cell-side-based
      ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("surface_height",ssName,"Node Scalar",cellType,surface_height_side);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (is_dist_param["ice_thickness"] || !has_ss_input[ssName][ice_thickness_side] ) {
      //---- Restrict ice thickness from cell-based to cell-side-based
      ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("ice_thickness",ssName,"Node Scalar",cellType,ice_thickness_side);
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if (is_dist_param["basal_friction"] || !has_ss_input[ssName]["basal_friction"])
    {
      //---- Restrict basal friction from cell-based to cell-side-based
      ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("basal_friction",ssName,"Node Scalar",cellType,basal_friction_side);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (is_dist_param["bed_roughness"] || !has_ss_input[ssName]["bed_roughness"])
    {
      // Interpolate the 3D state on the side (the BasalFrictionCoefficient evaluator needs a side field)
      ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("bed_roughness",ssName,"Node Scalar",cellType,bed_roughness_side);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (is_dist_param["bed_topography"] || !has_ss_input[ssName]["bed_topography"])
    {
      // Interpolate the 3D state on the side (the BasalFrictionCoefficient evaluator needs a side field)
      ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("bed_topography",ssName,"Node Scalar",cellType,bed_topography_side);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (is_dist_param["effective_pressure"] || !has_ss_input[ssName]["effective_pressure"])
    {
      // Interpolate the 3D state on the side (the BasalFrictionCoefficient evaluator needs a side field)
      ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("effective_pressure",ssName,"Node Scalar",cellType,effective_pressure_side);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (is_dist_param["flow_factor"] || !has_ss_input[ssName]["flow_factor"])
    {
      // Interpolate the 3D state on the side (the BasalFrictionCoefficient evaluator needs a side field)
      ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("flow_factor",ssName,"Cell Scalar",cellType,flow_factor_side);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    // -------------------------------- LandIce evaluators ------------------------- //

    // --- Basal Residual --- //
    p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Basal Residual"));

    //Input
    p->set<std::string>("BF Side Name", Albany::bf_name + " "+ssName);
    p->set<std::string>("Weighted Measure Name", Albany::weighted_measure_name + " "+ssName);
    p->set<std::string>("Basal Friction Coefficient Side QP Variable Name", beta_side);
    p->set<std::string>("Velocity Side QP Variable Name", velocity_side);
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
    p->set<std::string>("Field Name",velocity_side);
    p->set<std::string>("Field Layout","Cell Side Node Vector");
    p->set<std::string>("Side Set Name", ssName);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Field Norm"));

    // Output
    p->set<std::string>("Field Norm Name",sliding_velocity_side);

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
    p->set<std::string>("Ice Thickness Variable Name", ice_thickness_side);
    p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

    // Output
    p->set<std::string>("Ice Overburden Variable Name", ice_overburden_side);

    ev = Teuchos::rcp(new LandIce::IceOverburden<EvalT,PHAL::AlbanyTraits,true>(*p,dl_side));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Ice Overburden (Nodes) ---//
    p->set<bool>("Nodal",true);
    ev = Teuchos::rcp(new LandIce::IceOverburden<EvalT,PHAL::AlbanyTraits,true>(*p,dl_side));
    fm0.template registerEvaluator<EvalT>(ev);

    // If we are given an effective pressure field, we don't need a surrogate model for it
    if (!has_input["effective_pressure"]) {
      //--- Effective pressure surrogate (QPs) ---//
      p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Effective Pressure Surrogate"));

      // Input
      p->set<bool>("Nodal",false);
      p->set<std::string>("Side Set Name", ssName);
      p->set<std::string>("Ice Overburden Variable Name", ice_overburden_side);

      // Output
      p->set<std::string>("Effective Pressure Variable Name", effective_pressure_side);

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

      Teuchos::RCP<LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Alpha>> ptr_alpha;
      ptr_alpha = Teuchos::rcp(new LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Alpha>(*p,dl));
      ptr_alpha->setNominalValue(params->sublist("Parameters"),pl->sublist("Basal Friction Coefficient").get<double>(param_name,-1.0));
      fm0.template registerEvaluator<EvalT>(ptr_alpha);
    }

    //--- Shared Parameter for basal friction coefficient: lambda ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: lambda"));

    param_name = "Bed Roughness";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Lambda>> ptr_lambda;
    ptr_lambda = Teuchos::rcp(new LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Lambda>(*p,dl));
    ptr_lambda->setNominalValue(params->sublist("Parameters"),pl->sublist("Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_lambda);

    //--- Shared Parameter for basal friction coefficient: muCoulomb ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: muCoulomb"));

    param_name = "Coulomb Friction Coefficient";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::MuCoulomb>> ptr_muC;
    ptr_muC = Teuchos::rcp(new LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::MuCoulomb>(*p,dl));
    ptr_muC->setNominalValue(params->sublist("Parameters"),pl->sublist("Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_muC);

    //--- Shared Parameter for basal friction coefficient: muPowerLaw ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: muPowerLaw"));

    param_name = "Power Law Coefficient";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::MuPowerLaw>> ptr_muP;
    ptr_muP = Teuchos::rcp(new LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::MuPowerLaw>(*p,dl));
    ptr_muP->setNominalValue(params->sublist("Parameters"),pl->sublist("Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_muP);

    //--- Shared Parameter for basal friction coefficient: power ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: power"));

    param_name = "Power Exponent";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Power>> ptr_power;
    ptr_power = Teuchos::rcp(new LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Power>(*p,dl));
    ptr_power->setNominalValue(params->sublist("Parameters"),pl->sublist("Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_power);

    //--- LandIce basal friction coefficient ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Basal Friction Coefficient"));

    //Input
    p->set<std::string>("Sliding Velocity Variable Name", sliding_velocity_side);
    p->set<std::string>("BF Variable Name", Albany::bf_name + " " + ssName);
    p->set<std::string>("Effective Pressure QP Variable Name", effective_pressure_side);
    p->set<std::string>("Ice Softness Variable Name", flow_factor_side);
    p->set<std::string>("Bed Roughness Variable Name", bed_roughness_side);
    p->set<std::string>("Side Set Name", ssName);
    p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name + " " + ssName);
    p->set<Teuchos::ParameterList*>("Parameter List", &pl->sublist("Basal Friction Coefficient"));
    p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));
    p->set<Teuchos::ParameterList*>("Viscosity Parameter List", &params->sublist("LandIce Viscosity"));
    p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
    p->set<std::string>("Bed Topography Variable Name", bed_topography_side);
    p->set<std::string>("Effective Pressure Variable Name", effective_pressure_side);
    p->set<std::string>("Ice Thickness Variable Name", ice_thickness_side);

    //Output
    p->set<std::string>("Basal Friction Coefficient Variable Name", beta_side);

    basalMemoizer = enableMemoizer ? (!is_dist_param["basal_friction"] ? true : false) : false;
    if (basalMemoizer) {
      p->set<bool>("Enable Memoizer", basalMemoizer);
    }

    ev = Teuchos::rcp(new LandIce::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits,false,true,false>(*p,dl_side));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- LandIce basal friction coefficient at nodes ---//
    p->set<bool>("Nodal",true);
    ev = Teuchos::rcp(new LandIce::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits,false,true,false>(*p,dl_side));
    fm0.template registerEvaluator<EvalT>(ev);
  }
}

template <typename EvalT>
void StokesFOBase::constructLateralBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  const bool enableMemoizer = this->params->get<bool>("Use MDField Memoization", false);

  for (auto pl : landice_bcs[LandIceBC::Lateral]) {
    const std::string& ssName = pl->get<std::string>("Side Set Name");

    // We may have more than 1 lateral side set. The layout of all the side fields is the
    // same, so we need to differentiate them by name (just like we do for the basis functions already).

    std::string velocity_side = "velocity_" + ssName;
    std::string ice_thickness_side = "ice_thickness_" + ssName;
    std::string surface_height_side = "surface_height_" + ssName;

    //---- Restrict vertex coordinates from cell-based to cell-side-based
    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,ssName,"Vertex Vector",cellType,Albany::coord_vec_name + " " + ssName, enableMemoizer);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute Quad Points coordinates on the side set
    ev = evalUtils.constructMapToPhysicalFrameSideEvaluator(cellType,sideCubature.at(ssName), ssName, enableMemoizer);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute side basis functions
    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis.at(ssName), sideCubature.at(ssName), ssName, enableMemoizer, true);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate ice thickness on QP on side
    ev = evalUtils.getPSTUtils().constructDOFCellToSideQPEvaluator("ice_thickness", ssName, "Node Scalar", cellType,ice_thickness_side, enableMemoizer);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface height on QP on side
    ev = evalUtils.getPSTUtils().constructDOFCellToSideQPEvaluator("surface_height", ssName, "Node Scalar", cellType,surface_height_side, enableMemoizer);
    fm0.template registerEvaluator<EvalT>(ev);

    // -------------------------------- LandIce evaluators ------------------------- //

    // Lateral residual
    p = Teuchos::rcp( new Teuchos::ParameterList("Lateral Residual") );

    // Input
    p->set<std::string>("Ice Thickness Variable Name", ice_thickness_side);
    p->set<std::string>("Ice Surface Elevation Variable Name", surface_height_side);
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

    ev = Teuchos::rcp( new LandIce::StokesFOLateralResid<EvalT,PHAL::AlbanyTraits,false>(*p,dl) );
    fm0.template registerEvaluator<EvalT>(ev);
  }

}


template <typename EvalT>
void StokesFOBase::constructSurfaceVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                                       const Albany::MeshSpecsStruct& meshSpecs,
                                                       Albany::StateManager& stateMgr)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  if (surfaceSideName!="__INVALID__")
  {
    const auto& surfMeshSpecs = *meshSpecs.sideSetMeshSpecs.at(surfaceSideName)[0];
    auto dl_side = dl->side_layouts.at(surfaceSideName);

    // Find out if the obs surf velocity RMS is a scalar
    bool isObsVelRMSScalar = false;
    auto states = stateMgr.getRegisteredSideSetStates().at(surfaceSideName).at(surfMeshSpecs.ebName);
    auto it = states.find("observed_surface_velocity_RMS");
    if (it!=states.end()) {
      if (it->second->rank()==3) {
        isObsVelRMSScalar = true;
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(it->second->rank()!=4, std::logic_error, "Error! The layout of 'observed_surface_velocity_RMS' must be either <Cell,Side,Node> or <Cell,Side,Node,Dim>.\n");
      }
    } 

    // ------------------- Interpolations and utilities ------------------ //

    //---- Restrict vertex coordinates from cell-based to cell-side-based
    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,surfaceSideName,"Vertex Vector",cellType,Albany::coord_vec_name + " " + surfaceSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute side basis functions
    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis[surfaceSideName], sideCubature[surfaceSideName], surfaceSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate surface velocity on QP on side
    ev = evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("observed_surface_velocity_" + surfaceSideName, surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface velocity rms on QP on side
    if(isObsVelRMSScalar) {
      ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("observed_surface_velocity_RMS_" + surfaceSideName, surfaceSideName);
      fm0.template registerEvaluator<EvalT>(ev);
    } else {
      ev = evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("observed_surface_velocity_RMS_" + surfaceSideName, surfaceSideName);
      fm0.template registerEvaluator<EvalT>(ev);
    }

    //---- Restrict velocity (the solution) from cell-based to cell-side-based on upper side
    ev = evalUtils.constructDOFCellToSideEvaluator(dof_names[0],surfaceSideName,"Node Vector",cellType,"surface_velocity");
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity (the solution) on QP on side
    ev = evalUtils.constructDOFVecInterpolationSideEvaluator("surface_velocity", surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);

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

      std::string velocity_side = "velocity_" + ssName;
      std::string velocity_gradient_side = "velocity_" + ssName  + " Gradient";
      std::string sliding_velocity_side = "sliding_velocity_" + ssName;
      std::string basal_friction_side = "basal_friction_" + ssName;
      std::string beta_side = "beta_" + ssName;
      std::string beta_gradient_side = "beta_" + ssName + " Gradient";
      std::string effective_pressure_side = "effective_pressure_" + ssName;
      std::string effective_pressure_gradient_side = "effective_pressure_" + ssName + " Gradient";

      //---- Interpolate effective pressure gradient on QP on side
      ev = evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator(effective_pressure_side, ssName);
      fm0.template registerEvaluator<EvalT>(ev);

      //--- LandIce basal friction coefficient gradient ---//
      p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Basal Friction Coefficient Gradient"));
  
      // Input
      p->set<std::string>("Gradient BF Side Variable Name", Albany::grad_bf_name + " "+ssName);
      p->set<std::string>("Side Set Name", ssName);
      p->set<std::string>("Effective Pressure QP Name", effective_pressure_side);
      p->set<std::string>("Effective Pressure Gradient QP Name", effective_pressure_gradient_side);
      p->set<std::string>("Basal Velocity QP Name", velocity_side);
      p->set<std::string>("Basal Velocity Gradient QP Name", velocity_gradient_side);
      p->set<std::string>("Sliding Velocity QP Name", sliding_velocity_side);
      p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name +" "+ssName);
      p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
      p->set<Teuchos::ParameterList*>("Parameter List", &pl->sublist("Basal Friction Coefficient"));
  
      // Output
      p->set<std::string>("Basal Friction Coefficient Gradient Name",beta_gradient_side);
  
      ev = Teuchos::rcp(new LandIce::BasalFrictionCoefficientGradient<EvalT,PHAL::AlbanyTraits>(*p,dl->side_layouts.at(ssName)));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // Surface velocity mismatch can have a regularization term on a 2d mesh on the basal side
  if (basalSideName != "__INVALID__") {
    std::string stiffening_factor_side = "stiffening_factor_" + basalSideName;

    //---- Interpolate the 3D state on the side (the BasalFrictionCoefficient evaluator needs a side field)
    ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("stiffening_factor",basalSideName,"Node Scalar",cellType,stiffening_factor_side);
    fm0.template registerEvaluator<EvalT> (ev);
  
    //---- Interpolate stiffening gradient on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(stiffening_factor_side, basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate stiffening gradient on QP on side
    ev = evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator(stiffening_factor_side, basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);
  }
}

template <typename EvalT>
void StokesFOBase::constructSMBEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                           const Albany::MeshSpecsStruct& meshSpecs,
                                           std::map<std::string,bool>& is_dist_param)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  const bool enableMemoizer = this->params->get<bool>("Use MDField Memoization", false);

  // Evaluators needed for thickness-related diagnostics (e.g., SMB)
  if (basalSideName!="__INVALID__")
  {
    auto dl_side = dl->side_layouts.at(basalSideName);

    // We may have more than 1 basal side set. 'basalSideName' should be the union of all of them.
    // However, some of the fields used here, may be used also to compute quantities defined on
    // only some of the sub-sidesets of 'basalSideName'. The layout of all the side fields is the
    // same, so we need to differentiate them by name (just like we do for the basis functions already).

    std::string velocity_side = "velocity_" + basalSideName;
    std::string basal_friction_side = "basal_friction_" + basalSideName;
    std::string ice_thickness_side = "ice_thickness_" + basalSideName;
    std::string surface_height_side = "surface_height_" + basalSideName;
    std::string surface_mass_balance_side = "surface_mass_balance_" + basalSideName;
    std::string surface_mass_balance_RMS_side = "surface_mass_balance_RMS_" + basalSideName;
    std::string stiffening_factor_side = "stiffenting_factor_" + basalSideName;
    std::string effective_pressure_side = "effective_pressure_" + basalSideName;
    std::string bed_roughness_side = "bed_roughness_" + basalSideName;

    // ------------------- Interpolations and utilities ------------------ //

    //---- Restrict vertex coordinates from cell-based to cell-side-based
    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,basalSideName,"Vertex Vector",cellType,Albany::coord_vec_name +" " + basalSideName,enableMemoizer);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute side basis functions
    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis.at(basalSideName), sideCubature.at(basalSideName), basalSideName, enableMemoizer);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Restrict velocity from cell-based to cell-side-based
    ev = evalUtils.constructDOFCellToSideEvaluator(dof_names[0],basalSideName,"Node Vector",cellType,velocity_side);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity on QP on side
    ev = evalUtils.constructDOFVecInterpolationSideEvaluator(velocity_side, basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate velocity gradient on QP on side
    ev = evalUtils.constructDOFVecGradInterpolationSideEvaluator(velocity_side, basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Compute Quad Points coordinates on the side set
    ev = evalUtils.constructMapToPhysicalFrameSideEvaluator(cellType,sideCubature.at(basalSideName),basalSideName, enableMemoizer);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate stiffening gradient on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(stiffening_factor_side, basalSideName, enableMemoizer);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate stiffening gradient on QP on side
    ev = evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator(stiffening_factor_side, basalSideName, enableMemoizer);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate thickness on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(ice_thickness_side, basalSideName, enableMemoizer && !is_dist_param["ice_thickness"]);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate thickness gradient on QP on side
    ev = evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator(ice_thickness_side, basalSideName, enableMemoizer && !is_dist_param["ice_thickness"]);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate observed thickness on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("observed_ice_thickness_" + basalSideName, basalSideName, enableMemoizer);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate observed thickness RMS on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("observed_ice_thickness_RMS_" + basalSideName, basalSideName, enableMemoizer);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate velocity on QP on side
    ev = evalUtils.constructDOFVecInterpolationSideEvaluator("Averaged Velocity", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface mass balance on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(surface_mass_balance_side, basalSideName, enableMemoizer);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface mass balance on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(surface_mass_balance_RMS_side, basalSideName, enableMemoizer);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate flux_divergence from side quad points to side
    ev = evalUtils.constructSideQuadPointsToSideInterpolationEvaluator("flux_divergence",basalSideName,false);
    fm0.template registerEvaluator<EvalT> (ev);

    if (is_dist_param["ice_thickness"])
    {
      //---- Restrict ice thickness from cell-based to cell-side-based
      ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("ice_thickness",basalSideName,"Node Scalar",cellType,ice_thickness_side);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    // -------------------------------- LandIce evaluators ------------------------- //

    // Vertically averaged velocity
    p = Teuchos::rcp(new Teuchos::ParameterList("Gather Averaged Velocity"));

    p->set<std::string>("Averaged Velocity Name", "Averaged Velocity");
    p->set<std::string>("Mesh Part", "basalside");
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));

    ev = Teuchos::rcp(new GatherVerticallyAveragedVelocity<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    // Flux divergence
    p = Teuchos::rcp(new Teuchos::ParameterList("Flux Divergence"));

    //Input
    p->set<std::string>("Averaged Velocity Side QP Variable Name", "Averaged Velocity");
    p->set<std::string>("Averaged Velocity Side QP Divergence Name", "Averaged Velocity Divergence");
    p->set<std::string>("Thickness Side QP Variable Name", ice_thickness_side);
    p->set<std::string>("Thickness Gradient Name", ice_thickness_side + " Gradient");
    p->set<std::string>("Side Tangents Name", Albany::tangents_name + " " + basalSideName);

    p->set<std::string>("Field Name",  "flux_divergence");
    p->set<std::string>("Side Set Name", basalSideName);

    ev = Teuchos::rcp(new LandIce::FluxDiv<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
    fm0.template registerEvaluator<EvalT>(ev);

    // --- 2D divergence of Averaged Velocity ---- //
    p = Teuchos::rcp(new Teuchos::ParameterList("DOF Div Interpolation Side Averaged Velocity"));

    // Input
    p->set<std::string>("Variable Name", "Averaged Velocity");
    p->set<std::string>("Gradient BF Name", Albany::grad_bf_name + " "+basalSideName);
    p->set<std::string>("Tangents Name", "Tangents "+basalSideName);
    p->set<std::string>("Side Set Name",basalSideName);

    // Output (assumes same Name as input)
    p->set<std::string>("Divergence Variable Name", "Averaged Velocity Divergence");

    ev = Teuchos::rcp(new LandIce::DOFDivInterpolationSide<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
    fm0.template registerEvaluator<EvalT>(ev);
  }
}

template<typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
StokesFOBase::constructStokesFOBaseResponsesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                                        const Albany::MeshSpecsStruct& meshSpecs,
                                                        Albany::StateManager& stateMgr,
                                                        Albany::FieldManagerChoice fieldManagerChoice,
                                                        std::map<std::string,bool>& is_dist_param,
                                                        const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {

    // --- SurfaceVelocity-related evaluators (if needed) --- //
    constructSurfaceVelocityEvaluators<EvalT> (fm0, meshSpecs, stateMgr);

    // --- SMB-related evaluators (if needed) --- //
    constructSMBEvaluators<EvalT> (fm0, meshSpecs, is_dist_param);

    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));

    // Figure out if observed surface velocity RMS is scalar (if present at all)
    if (surfaceSideName!="__INVALID__") {
      auto it1 = stateMgr.getRegisteredSideSetStates().find(surfaceSideName);
      if (it1!=stateMgr.getRegisteredSideSetStates().end()) {
        std::string surfEBName = meshSpecs.sideSetMeshSpecs.at(surfaceSideName)[0]->ebName;
        auto it2 = it1->second.find(surfEBName);
        if (it2!=it1->second.end()) {
          auto where = it2->second.find("observed_surface_velocity_RMS");
          paramList->set<bool>("Scalar RMS",where!=it2->second.end() && where->second->rank()==3);
        }
      }
    }

    // ----------------------- Responses --------------------- //
    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
    paramList->set<Teuchos::ParameterList>("LandIce Physical Parameters List", params->sublist("LandIce Physical Parameters"));
    paramList->set<std::string>("Coordinate Vector Side Variable Name", Albany::coord_vec_name + " " + basalSideName);
    paramList->set<std::string>("Basal Friction Coefficient Name","beta");
    paramList->set<std::string>("Stiffening Factor Gradient Name","stiffening_factor_" + basalSideName + " Gradient");
    paramList->set<std::string>("Stiffening Factor Name", "stiffening_factor_" + basalSideName);
    paramList->set<std::string>("Thickness Gradient Name", "ice_thickness_" + basalSideName + " Gradient");
    paramList->set<std::string>("Thickness Side QP Variable Name","ice_thickness_" + basalSideName);
    paramList->set<std::string>("Thickness Side Variable Name","ice_thickness_" + basalSideName);
    paramList->set<std::string>("Bed Topography Side Variable Name","bed_topography_" + basalSideName);
    paramList->set<std::string>("Surface Velocity Side QP Variable Name","surface_velocity");
    paramList->set<std::string>("Averaged Vertical Velocity Side Variable Name","Averaged Velocity");
    paramList->set<std::string>("SMB Side QP Variable Name","surface_mass_balance_" + basalSideName);
    paramList->set<std::string>("SMB RMS Side QP Variable Name","surface_mass_balance_RMS_" + basalSideName);
    paramList->set<std::string>("Flux Divergence Side QP Variable Name","flux_divergence");
    paramList->set<std::string>("Thickness RMS Side QP Variable Name","observed_ice_thickness_RMS_" + basalSideName);
    paramList->set<std::string>("Observed Thickness Side QP Variable Name","observed_ice_thickness_" + basalSideName);
    paramList->set<std::string>("Observed Surface Velocity Side QP Variable Name","observed_surface_velocity_" + surfaceSideName);
    paramList->set<std::string>("Observed Surface Velocity RMS Side QP Variable Name","observed_surface_velocity_RMS_" + surfaceSideName);
    paramList->set<std::string>("Weighted Measure Basal Name",Albany::weighted_measure_name + " " + basalSideName);
    paramList->set<std::string>("Weighted Measure 2D Name",Albany::weighted_measure_name + " " + basalSideName);
    paramList->set<std::string>("Weighted Measure Surface Name",Albany::weighted_measure_name + " " + surfaceSideName);
    paramList->set<std::string>("Metric 2D Name",Albany::metric_name + " " + basalSideName);
    paramList->set<std::string>("Metric Basal Name",Albany::metric_name + " " + basalSideName);
    paramList->set<std::string>("Metric Surface Name",Albany::metric_name + " " + surfaceSideName);
    paramList->set<std::string>("Inverse Metric Basal Name",Albany::metric_inv_name + " " + basalSideName);
    paramList->set<std::string>("Basal Side Tangents Name",Albany::tangents_name + " " + basalSideName);
    paramList->set<std::string>("Basal Side Name", basalSideName);
    paramList->set<std::string>("Surface Side Name", surfaceSideName);
    paramList->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));
    paramList->set<std::vector<Teuchos::RCP<Teuchos::ParameterList>>*>("Basal Regularization Params",&landice_bcs[LandIceBC::BasalFriction]);

    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

} // namespace LandIce

#endif // LANDICE_STOKES_FO_BASE_HPP
