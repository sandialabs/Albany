//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_FO_THICKNESS_PROBLEM_HPP
#define LANDICE_STOKES_FO_THICKNESS_PROBLEM_HPP

#include <type_traits>

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "PHAL_LoadStateField.hpp"
#include "PHAL_SaveStateField.hpp"
#include "PHAL_DOFCellToSide.hpp"
#include "PHAL_LoadSideSetStateField.hpp"
#include "PHAL_SaveSideSetStateField.hpp"
#include "PHAL_DOFVecInterpolationSide.hpp"

#include "LandIce_ProblemUtils.hpp"
#include "LandIce_SharedParameter.hpp"
#include "LandIce_ParamEnum.hpp"
#include "LandIce_IceOverburden.hpp"
#include "LandIce_EffectivePressure.hpp"
#include "LandIce_StokesFOResid.hpp"
#include "LandIce_StokesFOLateralResid.hpp"
#include "LandIce_StokesFOBasalResid.hpp"
#include "LandIce_StokesFOBodyForce.hpp"
#include "LandIce_ViscosityFO.hpp"
#include "PHAL_FieldFrobeniusNorm.hpp"
#include "LandIce_ProlongateVector.hpp"
#include "LandIce_BasalFrictionCoefficient.hpp"
#include "LandIce_BasalFrictionCoefficientGradient.hpp"

#include "LandIce_Gather2DField.hpp"
#include "LandIce_GatherVerticallyAveragedVelocity.hpp"
#include "LandIce_ScatterResidual2D.hpp"
#include "LandIce_UpdateZCoordinate.hpp"
#include "LandIce_ThicknessResid.hpp"
#include "LandIce_StokesFOImplicitThicknessUpdateResid.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce
{

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
 */
class StokesFOThickness : public Albany::AbstractProblem {
public:

  //! Default constructor
  StokesFOThickness(const Teuchos::RCP<Teuchos::ParameterList>& params,
                    const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                    const Teuchos::RCP<ParamLib>& paramLib,
                    const int numDim_);

  //! Destructor
  ~StokesFOThickness() = default;

  //! Prohibit copying
  StokesFOThickness (const StokesFOThickness&) = delete;
  StokesFOThickness& operator= (const StokesFOThickness&) = delete;

  //! Return number of spatial dimensions
  int spatialDimension() const { return numDim; }

  //! Get boolean telling code if SDBCs are utilized
  bool useSDBCs() const {return use_sdbcs_; }

  //! Build the PDE instantiations, boundary conditions, and initial solution
  void buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                    Albany::StateManager& stateMgr);

  // Build evaluators
  Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
  buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                   const Albany::MeshSpecsStruct& meshSpecs,
                   Albany::StateManager& stateMgr,
                   Albany::FieldManagerChoice fmchoice,
                   const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Each problem must generate it's list of valid parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  //! Main problem setup routine. Not directly called, but indirectly by following functions
  template <typename EvalT>
  Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  template <typename EvalT>
  void constructStatesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                  const Albany::MeshSpecsStruct& meshSpecs,
                                  Albany::StateManager& stateMgr,
                                  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels,
                                  std::map<std::string,bool>& is_dist_param);

  template <typename EvalT>
  void constructStokesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                  const Albany::MeshSpecsStruct& meshSpecs,
                                  Albany::FieldManagerChoice fmchoice);


  template <typename EvalT>
  void constructThicknessEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0, 
                                     const Albany::MeshSpecsStruct& meshSpecs,
                                     Albany::FieldManagerChoice fmchoice);

  template <typename EvalT>
  void constructLateralBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  template <typename EvalT>
  void constructBasalBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  template <typename EvalT>
  void constructSurfaceVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructResponsesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                const Albany::MeshSpecsStruct& meshSpecs,
                                Albany::StateManager& stateMgr,
                                Albany::FieldManagerChoice fieldManagerChoice,
                                const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
  void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

protected:

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

  int numDim, vecDimFO;
  bool have_SMB;

  Teuchos::ArrayRCP<std::string> dof_names;
  Teuchos::ArrayRCP<std::string> resid_names;
  Teuchos::ArrayRCP<std::string> scatter_names;

  // Basal side, where thickness-related diagnostics are computed (e.g., SMB)
  std::string basalSideName;

  // Surface side, where velocity diagnostics are computed (e.g., velocity mismatch)
  std::string surfaceSideName;

  // Parameter lists for LandIce-specific BCs
  std::map<LandIceBC,std::vector<Teuchos::RCP<Teuchos::ParameterList>>>  landice_bcs;

  /// Boolean marking whether SDBCs are used
  bool use_sdbcs_;
};

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
StokesFOThickness::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                        const Albany::MeshSpecsStruct& meshSpecs,
                                        Albany::StateManager& stateMgr,
                                        Albany::FieldManagerChoice fieldManagerChoice,
                                        const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels = Teuchos::rcp(new std::map<std::string, int> ());
  std::map<std::string,bool> is_dist_param;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  constructStatesEvaluators<EvalT> (fm0, meshSpecs, stateMgr, extruded_params_levels, is_dist_param);

  // Evaluators that gather-or-compute coordinates
#ifndef ALBANY_MESH_DEPENDS_ON_SOLUTION
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  //---- Gather coordinates
  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);
#else

  //--- Gather Coordinates ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Gather Coordinate Vector"));

  // Output:: Coordindate Vector at vertices
  p->set<std::string>("Coordinate Vector Name", "Coord Vec Old");

  ev =  Teuchos::rcp(new PHAL::GatherCoordinateVector<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Update Z Coordinate ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Update Z Coordinate"));

  // Input
  p->set<std::string>("Old Coords Name", "Coord Vec Old");
  p->set<std::string>("New Coords Name", Albany::coord_vec_name);
  p->set<std::string>("Thickness Increment Name", "ExtrudedThickness");
  p->set<std::string>("Past Thickness Name", "ice_thickness");
  p->set<std::string>("Top Surface Name", "surface_height");

  ev = Teuchos::rcp(new LandIce::UpdateZCoordinateMovingTop<EvalT,PHAL::AlbanyTraits>(*p, dl));
  fm0.template registerEvaluator<EvalT>(ev);
#endif

  //--- Shared Parameter for Continuation:  ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Homotopy Parameter"));

  std::string param_name = "Glen's Law Homotopy Parameter";
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>> ptr_homotopy;
  ptr_homotopy = Teuchos::rcp(new LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>(*p,dl));
  ptr_homotopy->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Viscosity").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_homotopy);

  // --- StokesFO volume evaluators --- //
  constructStokesEvaluators<EvalT> (fm0, meshSpecs, fieldManagerChoice);

  // --- Thickness equation evaluators --- //
  constructThicknessEvaluators<EvalT> (fm0, meshSpecs, fieldManagerChoice);

  // --- Lateral BC evaluators (if needed) --- //
  constructLateralBCEvaluators<EvalT> (fm0);

  // --- Basal BC evaluators (if needed) --- //
  constructBasalBCEvaluators<EvalT> (fm0);

  // --- SurfaceVelocity-related evaluators (if needed) --- //
  constructSurfaceVelocityEvaluators<EvalT> (fm0);

  // Finally, construct responses, and return the tags
  return constructResponsesEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice, responseList);
}

template <typename EvalT>
void StokesFOThickness::constructStatesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                                   const Albany::MeshSpecsStruct& meshSpecs,
                                                   Albany::StateManager& stateMgr,
                                                   Teuchos::RCP<std::map<std::string, int> > extruded_params_levels,
                                                   std::map<std::string,bool>& is_dist_param)
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
        int extruded_param_level = 0;
        extruded_params_levels->insert(std::make_pair(param_name, extruded_param_level));
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

    if (fieldUsage == "Unused")
      continue;

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
    }
    else if (fieldUsage == "Input" || fieldUsage == "Input-Output")
    {
      // Not a parameter but still required as input: load it.
      p->set<std::string>("Field Name", fieldName);
      ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
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

      if (fieldUsage == "Unused")
        continue;

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
      }
      else if (fieldUsage == "Input" || fieldUsage == "Input-Output")
      {
        // Not a parameter but requires as input: load it.
        p->set<std::string>("Field Name", fieldName);
        ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }
  }
}

template<typename EvalT>
void StokesFOThickness::constructStokesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                                   const Albany::MeshSpecsStruct& meshSpecs,
                                                   Albany::FieldManagerChoice fieldManagerChoice)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  std::string param_name;

  // ------------------- Interpolations and utilities ------------------ //
  int offsetVelocity = 0;
  int offsetThickness = vecDimFO;

  //---- Gather solution field (velocity only)
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names.persistingView(0,1), offsetVelocity);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate velocity field
  ev = evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0], offsetVelocity);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate solution field gradient
  ev = evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0], offsetVelocity);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Compute physical frame coordinates
  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Compute basis functions (and related fields)
  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate Surface Height gradient
  ev = evalUtils.getPSTUtils().constructDOFGradInterpolationEvaluator("surface_height");
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate temperature from nodes to cell
  ev = evalUtils.getPSTUtils().constructNodesToCellInterpolationEvaluator ("temperature",false);
  fm0.template registerEvaluator<EvalT> (ev);

  // -------------------------------- LandIce/PHAL evaluators ------------------------- //

  //--- LandIce Gather Extruded 2D Field (Thickness) ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Gather ExtrudedThickness"));

  //Input
  p->set<std::string>("2D Field Name", "ExtrudedThickness");
  p->set<int>("Offset of First DOF", offsetThickness);
  p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));

  ev = Teuchos::rcp(new LandIce::GatherExtruded2DField<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Shared Parameter for homotopy parameter: h ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Glen's Law Homotopy Parameter"));

  param_name = "Glen's Law Homotopy Parameter";
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>> ptr_h;
  ptr_h = Teuchos::rcp(new LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>(*p,dl));
  ptr_h->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Viscosity").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_h);

  // --- FO Stokes Resid --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("StokesFO Resid"));

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
  p->set<std::string>("Residual Variable Name", "StokesFO Residual");

  ev = Teuchos::rcp(new LandIce::StokesFOResid<EvalT,PHAL::AlbanyTraits>(*p, dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- FO Stokes Implicit Thickness Update Resid --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("StokesFOImplicitThicknessUpdate Resid"));

  //Input
  p->set<std::string>("Thickness Increment Variable Name", "ExtrudedThickness");
  p->set<std::string>("Gradient BF Name", Albany::grad_bf_name);
  p->set<std::string>("Weighted BF Name", Albany::weighted_bf_name);

  Teuchos::ParameterList& physParamList = params->sublist("LandIce Physical Parameters");
  p->set<Teuchos::ParameterList*>("Physical Parameter List", &physParamList);

  //Output
  p->set<std::string>("Residual Name", resid_names[0]);

  ev = Teuchos::rcp(new LandIce::StokesFOImplicitThicknessUpdateResid<EvalT,PHAL::AlbanyTraits>(*p, dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- LandIce viscosity ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Viscosity"));

  //Input
  p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name);
  p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
  p->set<std::string>("Velocity QP Variable Name", "Velocity");
  p->set<std::string>("Temperature Variable Name", "temperature");
  p->set<std::string>("Flow Factor Variable Name", "flow_factor");
  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Viscosity"));
  p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

  //Output
  p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
  p->set<std::string>("EpsilonSq QP Variable Name", "LandIce EpsilonSq");

  ev = Teuchos::rcp(new LandIce::ViscosityFO<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ParamScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Body Force ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Body Force"));

  //Input
  p->set<std::string>("LandIce Viscosity QP Variable Name", "LandIce Viscosity");
  p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name);
  p->set<std::string>("Surface Height Gradient Name", "surface_height Gradient");
  p->set<std::string>("Surface Height Name", "Surface Height");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Body Force"));
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

  //Output
  p->set<std::string>("Body Force Variable Name", "Body Force");

  ev = Teuchos::rcp(new LandIce::StokesFOBodyForce<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Scatter LandIce Stokes FO Residual ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Scatter StokesFOImplicitThicknessUpdate"));

  //Input
  p->set< Teuchos::ArrayRCP<std::string> >("Residual Names", resid_names);
  p->set<int>("Tensor Rank", 1);
  p->set<int>("Offset of First DOF", 0);
  p->set<int>("Offset 2D Field", 2);

  //Output
  p->set<std::string>("Scatter Field Name", "Scatter StokesFO Residual");

  ev = Teuchos::rcp(new PHAL::ScatterResidualWithExtrudedField<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> resStokesFO(scatter_names[0], dl->dummy);
    fm0.requireField<EvalT>(resStokesFO);
  }
}

template<typename EvalT>
void StokesFOThickness::constructThicknessEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0, 
                                                      const Albany::MeshSpecsStruct& meshSpecs,
                                                      Albany::FieldManagerChoice fieldManagerChoice)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  if (basalSideName!="__INVALID__")
  {
    // -------------------- Special evaluators for side handling ----------------- //

    //---- Restrict coordinate vector from cell-based to cell-side-based
    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,basalSideName,"Vertex Vector",cellType,"Coord Vec " + basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute side basis functions
    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis.at(basalSideName), sideCubature.at(basalSideName), basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Restrict velocity from cell-based to cell-side-based
    ev = evalUtils.constructDOFCellToSideEvaluator("Velocity",basalSideName,"Node Vector",cellType,"velocity_" + basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Restrict ice thickness from cell-based to cell-side-based
    ev = evalUtils.constructDOFCellToSideEvaluator("ice_thickness",basalSideName,"Node Scalar",cellType,"ice_thickness_" + basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Restrict surface height from cell-based to cell-side-based
    ev = evalUtils.constructDOFCellToSideEvaluator("surface_height",basalSideName,"Node Scalar",cellType,"surface_height_" + basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity on QP on side
    ev = evalUtils.constructDOFVecInterpolationSideEvaluator("velocity_" + basalSideName, basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate thickness on QP on side
    ev = evalUtils.constructDOFInterpolationSideEvaluator("ice_thickness_" + basalSideName, basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate basal_friction on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("basal_friction_" + basalSideName, basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface height on QP on side
    ev = evalUtils.constructDOFInterpolationSideEvaluator("surface_height_" + basalSideName, basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);
  }

  int offsetThickness = vecDimFO;

  //--- LandIce Gather Vertically Averaged Velocity ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Gather Averaged Velocity"));

  //Input
  p->set<std::string>("Averaged Velocity Name", "Averaged Velocity");
  p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));

  ev = Teuchos::rcp(new LandIce::GatherVerticallyAveragedVelocity<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- LandIce Gather 2D Field (Thickness) ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Gather Thickness"));

  //Input
  p->set<std::string>("2D Field Name", "Thickness2D");
  p->set<int>("Offset of First DOF", offsetThickness);
  p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));

  ev = Teuchos::rcp(new LandIce::Gather2DField<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- Thickness Resid --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Thickness Resid"));

  //Input
  p->set<std::string>("Averaged Velocity Variable Name", "Averaged Velocity");
  p->set<std::string>("Thickness Increment Variable Name", "Thickness2D");
  p->set<std::string>("Past Thickness Name", "ice_thickness");
  p->set<std::string>("Coordinate Vector Name", Albany::coord_vec_name);
  p->set<int>("Cubature Degree",3);
  if(this->params->isSublist("Parameter Fields")) {
    Teuchos::ParameterList& params_list =  this->params->sublist("Parameter Fields");
    if (params_list.get<int>("Register Surface Mass Balance",0)) {
      p->set<std::string>("SMB Name", "surface_mass_balance");
    }
  }
  p->set<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", Teuchos::rcpFromRef(meshSpecs));
  if(this->params->isParameter("Time Step Ptr")) {
    p->set<Teuchos::RCP<double> >("Time Step Ptr", this->params->get<Teuchos::RCP<double> >("Time Step Ptr"));
  } else {
    Teuchos::RCP<double> dt = Teuchos::rcp(new double(this->params->get<double>("Time Step")));
    p->set<Teuchos::RCP<double> >("Time Step Ptr", dt);
  }

  //Output
  p->set<std::string>("Residual Name", resid_names[1]);

  ev = Teuchos::rcp(new LandIce::ThicknessResid<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- LandIce Stokes FO Residual Thickness ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Scatter ResidualH"));

  //Input
  p->set< Teuchos::ArrayRCP<std::string> >("Residual Names", resid_names.persistingView(1,1));
  p->set<int>("Tensor Rank", 0);
  p->set<int>("Offset of First DOF", offsetThickness);
  p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));

  //Output
  p->set<std::string>("Scatter Field Name", scatter_names[1]);

  ev = Teuchos::rcp(new PHAL::ScatterResidual2D<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> resStokesFO(scatter_names[1], dl->dummy);
    fm0.requireField<EvalT>(resStokesFO);
  }
}

template <typename EvalT>
void StokesFOThickness::constructBasalBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  std::string param_name;

  for (auto pl : landice_bcs[LandIceBC::BasalFriction]) {
    const std::string& ssName = pl->get<std::string>("Side Set Name");

    //---- Restrict coordinate vector from cell-based to cell-side-based
    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,ssName,"Vertex Vector",cellType,"Coord Vec " + ssName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute side basis functions
    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis.at(ssName), sideCubature.at(ssName), ssName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Restrict velocity from cell-based to cell-side-based
    ev = evalUtils.constructDOFCellToSideEvaluator("Velocity",ssName,"Node Vector",cellType,"velocity_" + ssName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity on QP on side
    ev = evalUtils.constructDOFVecInterpolationSideEvaluator("velocity_" + ssName, ssName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate solution field gradient
    ev = evalUtils.constructDOFVecGradInterpolationSideEvaluator("velocity_" + ssName, ssName);
    fm0.template registerEvaluator<EvalT> (ev);

    // --- Basal Residual --- //
    p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Basal Residual"));

    //Input
    p->set<std::string>("BF Side Name", Albany::bf_name + " "+ssName);
    p->set<std::string>("Weighted Measure Name", Albany::weighted_measure_name + " "+ssName);
    p->set<std::string>("Basal Friction Coefficient Side QP Variable Name", "beta_" + ssName);
    p->set<std::string>("Velocity Side QP Variable Name", "velocity_" + ssName);
    p->set<std::string>("Side Set Name", ssName);
    p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<Teuchos::ParameterList*>("Parameter List", &pl->sublist("Basal Friction Coefficient"));
    p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

    //Output
    p->set<std::string>("Residual Variable Name", resid_names[0]);

    ev = Teuchos::rcp(new LandIce::StokesFOBasalResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Sliding velocity calculation ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Velocity Norm"));

    // Input
    p->set<std::string>("Field Name","velocity_" + ssName);
    p->set<std::string>("Field Layout","Cell Side QuadPoint Vector");
    p->set<std::string>("Side Set Name", ssName);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Field Norm"));

    // Output
    p->set<std::string>("Field Norm Name","sliding_velocity_" + ssName);

    ev = Teuchos::rcp(new PHAL::FieldFrobeniusNorm<EvalT,PHAL::AlbanyTraits>(*p,dl->side_layouts.at(ssName)));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Ice Overburden (QPs) ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Effective Pressure Surrogate"));

    // Input
    p->set<bool>("Nodal",false);
    p->set<std::string>("Side Set Name", ssName);
    p->set<std::string>("Ice Thickness Variable Name", "ice_thickness_" + ssName);
    p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

    // Output
    p->set<std::string>("Ice Overburden Variable Name", "ice_overburden_" + ssName);

    ev = Teuchos::rcp(new LandIce::IceOverburden<EvalT,PHAL::AlbanyTraits,true>(*p,dl->side_layouts.at(ssName)));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Ice Overburden (Nodes) ---//
    p->set<bool>("Nodal",true);
    ev = Teuchos::rcp(new LandIce::IceOverburden<EvalT,PHAL::AlbanyTraits,true>(*p,dl->side_layouts.at(ssName)));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Effective pressure surrogate (QPs) ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Effective Pressure Surrogate"));

    // Input
    p->set<bool>("Nodal",false);
    p->set<std::string>("Side Set Name", ssName);
    p->set<std::string>("Ice Overburden Variable Name", "ice_overburden_" + ssName);

    // Output
    p->set<std::string>("Effective Pressure Variable Name","effective_pressure_" + ssName);

    ev = Teuchos::rcp(new LandIce::EffectivePressure<EvalT,PHAL::AlbanyTraits,true,true>(*p,dl->side_layouts.at(ssName)));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Effective pressure surrogate (Nodes) ---//
    p->set<bool>("Nodal",true);
    ev = Teuchos::rcp(new LandIce::EffectivePressure<EvalT,PHAL::AlbanyTraits,true,true>(*p,dl->side_layouts.at(ssName)));
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

    //--- Shared Parameter for basal friction coefficient: lambda ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: lambda"));

    param_name = "Bed Roughness";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Lambda>> ptr_lambda;
    ptr_lambda = Teuchos::rcp(new LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Lambda>(*p,dl));
    ptr_lambda->setNominalValue(params->sublist("Parameters"),pl->sublist("Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_lambda);

    //--- Shared Parameter for basal friction coefficient: mu ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: mu"));

    param_name = "Coulomb Friction Coefficient";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Mu>> ptr_mu;
    ptr_mu = Teuchos::rcp(new LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Mu>(*p,dl));
    ptr_mu->setNominalValue(params->sublist("Parameters"),pl->sublist("Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_mu);

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
    p->set<std::string>("Sliding Velocity Variable Name", "sliding_velocity_" + ssName);
    p->set<std::string>("BF Variable Name", Albany::bf_name + " " + ssName);
    p->set<std::string>("Effective Pressure QP Variable Name", "effective_pressure_" + ssName);
    p->set<std::string>("Bed Roughness Variable Name", "bed_roughness_" + ssName);
    p->set<std::string>("Side Set Name", ssName);
    p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name + " " + ssName);
    p->set<Teuchos::ParameterList*>("Parameter List", &pl->sublist("Basal Friction Coefficient"));
    p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));
    p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
    p->set<std::string>("Bed Topography Variable Name", "bed_topography_" + ssName);
    p->set<std::string>("Ice Thickness Variable Name", "ice_thickness_" + ssName);

    //Output
    p->set<std::string>("Basal Friction Coefficient Variable Name", "beta_" + ssName);

    ev = Teuchos::rcp(new LandIce::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits,false,true,false>(*p,dl->side_layouts.at(ssName)));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- LandIce basal friction coefficient at nodes ---//
    p->set<bool>("Nodal",true);
    ev = Teuchos::rcp(new LandIce::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits,false,true,false>(*p,dl->side_layouts.at(ssName)));
    fm0.template registerEvaluator<EvalT>(ev);
  }
}

template <typename EvalT>
void StokesFOThickness::constructLateralBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  for (auto pl : landice_bcs[LandIceBC::Lateral]) {
    const std::string& ssName = pl->get<std::string>("Side Set Name");

    //---- Restrict vertex coordinates from cell-based to cell-side-based
    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,ssName,"Vertex Vector",cellType,Albany::coord_vec_name + " " + ssName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute Quad Points coordinates on the side set
    ev = evalUtils.constructMapToPhysicalFrameSideEvaluator(cellType,sideCubature.at(ssName),ssName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute side basis functions
    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis.at(ssName), sideCubature.at(ssName), ssName, false, true);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate ice thickness on QP on side
    ev = evalUtils.getPSTUtils().constructDOFCellToSideQPEvaluator("ice_thickness", ssName, "Node Scalar", cellType,"ice_thickness_"+ssName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface height on QP on side
    ev = evalUtils.getPSTUtils().constructDOFCellToSideQPEvaluator("surface_height", ssName, "Node Scalar", cellType,"surface_height_"+ssName);
    fm0.template registerEvaluator<EvalT>(ev);

    // -------------------------------- LandIce evaluators ------------------------- //

    // Lateral residual
    p = Teuchos::rcp( new Teuchos::ParameterList("Lateral Residual") );

    // Input
    p->set<std::string>("Ice Thickness Variable Name", "ice_thickness_"+ssName);
    p->set<std::string>("Ice Surface Elevation Variable Name", "surface_height_"+ssName);
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
void StokesFOThickness::constructSurfaceVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  if (surfaceSideName!="__INVALID__")
  {
    //---- Restrict coordinate vector from cell-based to cell-side-based
    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,surfaceSideName,"Vertex Vector",cellType,"Coord Vec " + surfaceSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute side basis functions
    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis.at(surfaceSideName), sideCubature.at(surfaceSideName), surfaceSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate surface velocity on QP on side
    ev = evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("observed_surface_velocity_" + surfaceSideName, surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface velocity rms on QP on side
    ev = evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("observed_surface_velocity_RMS_" + surfaceSideName, surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Restrict velocity (the solution) from cell-based to cell-side-based on upper side
    ev = evalUtils.constructDOFCellToSideEvaluator(dof_names[0],surfaceSideName,"Node Vector",cellType,"surface_velocity");
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity (the solution) on QP on side
    ev = evalUtils.constructDOFVecInterpolationSideEvaluator("surface_velocity", surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    // Surface Velocity Mismatch may require the gradient of the basal friction coefficient as a regularization.
    for (auto pl : landice_bcs[LandIceBC::BasalFriction]) {
      std::string ssName  = pl->get<std::string>("Side Set Name");

      std::string basal_velocity_side = "velocity_" + ssName;
      std::string basal_velocity_gradient_side = "velocity_" + ssName  + " Gradient";
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
      p->set<std::string>("Basal Velocity QP Name", basal_velocity_side);
      p->set<std::string>("Basal Velocity Gradient QP Name", basal_velocity_gradient_side);
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
}

template<typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
StokesFOThickness::constructResponsesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                                 const Albany::MeshSpecsStruct& meshSpecs,
                                                 Albany::StateManager& stateMgr,
                                                 Albany::FieldManagerChoice fieldManagerChoice,
                                                 const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    // ----------------------- Responses --------------------- //
    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));
    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
    paramList->set<std::string>("Surface Velocity Side QP Variable Name","surface_velocity");
    paramList->set<std::string>("Observed Surface Velocity Side QP Variable Name","observed_surface_velocity_" + surfaceSideName);
    paramList->set<std::string>("Observed Surface Velocity RMS Side QP Variable Name","observed_surface_velocity_RMS_" + surfaceSideName);
    paramList->set<std::string>("BF Surface Name",Albany::bf_name + " " + surfaceSideName);
    paramList->set<std::string>("Weighted Measure Basal Name",Albany::weighted_measure_name + " " + basalSideName);
    paramList->set<std::string>("Weighted Measure Surface Name",Albany::weighted_measure_name + " " + surfaceSideName);
    paramList->set<std::string>("Inverse Metric Basal Name",Albany::metric_inv_name + " " + basalSideName);
    paramList->set<std::string>("Metric Basal Name",Albany::metric_name + " " + basalSideName);
    paramList->set<std::string>("Metric Surface Name",Albany::metric_name + " " + surfaceSideName);
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

#endif // LANDICE_STOKES_FO_THICKNESS_PROBLEM_HPP
