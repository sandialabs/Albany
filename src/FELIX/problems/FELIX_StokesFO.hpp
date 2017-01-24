//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKES_FO_PROBLEM_HPP
#define FELIX_STOKES_FO_PROBLEM_HPP 1

#include <type_traits>

#include "Phalanx.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "PHAL_AddNoise.hpp"
#include "PHAL_LoadStateField.hpp"
#include "PHAL_SaveCellStateField.hpp"
#include "PHAL_DOFCellToSide.hpp"
#include "PHAL_DOFVecInterpolationSide.hpp"
#include "PHAL_LoadSideSetStateField.hpp"
#include "PHAL_SaveSideSetStateField.hpp"
#include "PHAL_SaveStateField.hpp"
#include "FELIX_SharedParameter.hpp"
#include "FELIX_ParamEnum.hpp"

#include "FELIX_EffectivePressure.hpp"
#include "FELIX_StokesFOResid.hpp"
#include "FELIX_StokesFOBasalResid.hpp"
#include "FELIX_L2ProjectedBoundaryLaplacianResidual.hpp"
#ifdef CISM_HAS_FELIX
#include "FELIX_CismSurfaceGradFO.hpp"
#endif
#include "FELIX_StokesFOBodyForce.hpp"
#include "FELIX_StokesFOStress.hpp"
#include "FELIX_ViscosityFO.hpp"
#include "PHAL_Field2Norm.hpp"
#include "FELIX_FluxDiv.hpp"
#include "FELIX_BasalFrictionCoefficient.hpp"
#include "FELIX_BasalFrictionCoefficientNode.hpp"
#include "FELIX_BasalFrictionCoefficientGradient.hpp"
#include "FELIX_BasalFrictionHeat.hpp"
#include "FELIX_Dissipation.hpp"
#include "FELIX_UpdateZCoordinate.hpp"
#include "FELIX_GatherVerticallyAveragedVelocity.hpp"
#include "FELIX_Time.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX
{

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
 */
class StokesFO : public Albany::AbstractProblem
{
public:

  //! Default constructor
  StokesFO (const Teuchos::RCP<Teuchos::ParameterList>& params,
            const Teuchos::RCP<Teuchos::ParameterList>& discParams,
            const Teuchos::RCP<ParamLib>& paramLib,
            const int numDim_);

  //! Destructor
  ~StokesFO();

  //! Return number of spatial dimensions
  virtual int spatialDimension() const { return numDim; }

  //! Build the PDE instantiations, boundary conditions, and initial solution
  virtual void buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                             Albany::StateManager& stateMgr);

  // Build evaluators
  virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
  buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                   const Albany::MeshSpecsStruct& meshSpecs,
                   Albany::StateManager& stateMgr,
                   Albany::FieldManagerChoice fmchoice,
                   const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Each problem must generate it's list of valide parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

private:

  //! Private to prohibit copying
  StokesFO(const StokesFO&);

  //! Private to prohibit copying
  StokesFO& operator=(const StokesFO&);

public:

  //! Main problem setup routine. Not directly called, but indirectly by following functions
  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
  void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

protected:

  Teuchos::RCP<shards::CellTopology> cellType;
  Teuchos::RCP<shards::CellTopology> basalSideType;
  Teuchos::RCP<shards::CellTopology> surfaceSideType;

  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  cellCubature;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  basalCubature;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  surfaceCubature;

  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > cellBasis;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > basalSideBasis;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > surfaceSideBasis;

  int numDim;
  int vecDimFO;
  Teuchos::RCP<Albany::Layouts> dl, dl_scalar, dl_side_scalar, dl_basal, dl_surface;

  //! Discretization parameters
  Teuchos::RCP<Teuchos::ParameterList> discParams;


  bool  sliding;
  std::string basalSideName;
  std::string surfaceSideName;

  std::string elementBlockName;
  std::string basalEBName;
  std::string surfaceEBName;
};

} // Namespace FELIX

// ================================ IMPLEMENTATION ============================ //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
FELIX::StokesFO::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                      const Albany::MeshSpecsStruct& meshSpecs,
                                      Albany::StateManager& stateMgr,
                                      Albany::FieldManagerChoice fieldManagerChoice,
                                      const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils_scalar(dl_scalar);

  int offset=0;

  Albany::StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;
  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels = Teuchos::rcp(new std::map<std::string, int> ());

  // ---------------------------- Registering state variables ------------------------- //

  std::string stateName, fieldName, param_name;

  // Getting the names of the distributed parameters (they won't have to be loaded as states)
  std::map<std::string,bool> is_dist_param;
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
      }
      else
      {
        // Legacy way to specify dist params: with parameter entries. Note: no mesh part can be specified.
        param_name = dist_params_list.get<std::string>(Albany::strint("Parameter", p_index));
        dist_params_name_to_mesh_part[param_name] = "";
      }
      is_dist_param[param_name] = true;
    }
  }

  if (discParams->isSublist("Required Fields Info")){
    Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
    int num_fields = req_fields_info.get<int>("Number Of Fields",0);

    std::string fieldType, fieldUsage, meshPart;
    bool nodal_state;
    for (int ifield=0; ifield<num_fields; ++ifield)
    {
      const Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));

      // Get current state specs
      stateName  = fieldName = thisFieldList.get<std::string>("Field Name");
      fieldType  = thisFieldList.get<std::string>("Field Type");
      fieldUsage = thisFieldList.get<std::string>("Field Usage");

      if (fieldUsage == "Unused")
        continue;

      meshPart = is_dist_param[stateName] ? dist_params_name_to_mesh_part[stateName] : "";

      if(fieldType == "Elem Scalar") {
        entity = Albany::StateStruct::ElemData;
        p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, elementBlockName, true, &entity, meshPart);
        nodal_state = false;
      }
      else if(fieldType == "Node Scalar") {
        entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, meshPart);
        nodal_state = true;
      }
      else if(fieldType == "Elem Vector") {
        entity = Albany::StateStruct::ElemData;
        p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, true, &entity, meshPart);
        nodal_state = false;
      }
      else if(fieldType == "Node Vector") {
        entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, true, &entity, meshPart);
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
        ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
        fm0.template registerEvaluator<EvalT>(ev);
      }
      else if (fieldUsage == "Input" || fieldUsage == "Input-Output")
      {
        // Not a parameter but still required as input: load it.
        p->set<std::string>("Field Name", fieldName);
        ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }
  }
  else {//temporary fix for non STK meshes..
    stateName = fieldName = "temperature";
    entity = Albany::StateStruct::NodalDataToElemNode;
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
    p->set<std::string>("Field Name", fieldName);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

#ifdef CISM_HAS_FELIX
    // Surface Gradient-x
    entity = Albany::StateStruct::NodalDataToElemNode;
    stateName = "xgrad_surface_height"; //ds/dx which can be passed from CISM (definened at nodes)
    fieldName = "CISM Surface Height Gradient X";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
    p->set<std::string>("Field Name", fieldName);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Surface Gradient-y
    entity = Albany::StateStruct::NodalDataToElemNode;
    stateName = "ygrad_surface_height"; //ds/dy which can be passed from CISM (defined at nodes)
    fieldName = "CISM Surface Height Gradient Y";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
    p->set<std::string>("Field Name", fieldName);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
#endif

    // Ice thickness
    if(is_dist_param["ice_thickness"])
    {
      // Thickness is a distributed parameter
      TEUCHOS_TEST_FOR_EXCEPTION (ss_requirements.find(basalSideName)==ss_requirements.end(), std::logic_error,
                                  "Error! 'ice_thickness' is a parameter, but there are no basal requirements.\n");
      const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);

      TEUCHOS_TEST_FOR_EXCEPTION (std::find(req.begin(), req.end(), stateName)==req.end(), std::logic_error,
                                  "Error! 'ice_thickness' is a parameter, but is not listed as basal requirements.\n");

      // ice_thickness is a distributed 3D parameter
      entity = Albany::StateStruct::NodalDistParameter;
      fieldName = "ice_thickness Param";
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, dist_params_name_to_mesh_part["ice_thickness"]);
      ev = evalUtils.constructGatherScalarExtruded2DNodalParameter(stateName,fieldName);
      fm0.template registerEvaluator<EvalT>(ev);

      std::stringstream key;
      key << stateName <<  "Is Distributed Parameter";
      this->params->set<int>(key.str(), 1);
    }
    else
    {
      // ice_thickness is just an input field
      fieldName = "ice_thickness";
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
      p->set<std::string>("Field Name", fieldName);
      ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Bed topography
    stateName = "bed_topography";
    entity= Albany::StateStruct::NodalDataToElemNode;
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

#if defined(CISM_HAS_FELIX) || defined(MPAS_HAS_FELIX)
    // Dirichelt field
    entity = Albany::StateStruct::NodalDistParameter;
    // Here is how to register the field for dirichlet condition.
    stateName = "dirichlet_field";
    p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, true, &entity, "");
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
#endif
  }

  if (discParams->isSublist("Side Set Discretizations"))
  {
    Teuchos::Array<std::string> ss_names = discParams->sublist("Side Set Discretizations").get<Teuchos::Array<std::string>>("Side Sets");

    for (int i=0; i<ss_names.size(); ++i)
    {
      const std::string& ss_name = ss_names[i];
      Teuchos::ParameterList& req_fields_info = discParams->sublist("Side Set Discretizations").sublist(ss_name).sublist("Required Fields Info");
      int num_fields = req_fields_info.get<int>("Number Of Fields",0);
      Teuchos::RCP<PHX::DataLayout> dl_temp;
      Teuchos::RCP<PHX::DataLayout> sns;
      std::string fieldType, fieldUsage, meshPart;
      bool nodal_state;
      int numLayers;

      const std::string& sideEBName = meshSpecs.sideSetMeshSpecs.at(ss_name)[0]->ebName;
      Teuchos::RCP<Albany::Layouts> ss_dl = dl->side_layouts.at(ss_name);
      for (int ifield=0; ifield<num_fields; ++ifield)
      {
        const Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));

        // Get current state specs
        stateName  = fieldName = thisFieldList.get<std::string>("Field Name");
        fieldUsage = thisFieldList.get<std::string>("Field Usage");

        if (fieldUsage == "Unused")
          continue;

        meshPart = is_dist_param[stateName] ? dist_params_name_to_mesh_part[stateName] : "";

        numLayers = thisFieldList.isParameter("Number Of Layers") ? thisFieldList.get<int>("Number Of Layers") : -1;
        fieldType  = thisFieldList.get<std::string>("Field Type");

        if(fieldType == "Elem Scalar") {
          entity = Albany::StateStruct::ElemData;
          p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->cell_scalar2, sideEBName, true, &entity, meshPart);
          nodal_state = false;
        }
        else if(fieldType == "Node Scalar") {
          entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
          p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->node_scalar, sideEBName, true, &entity, meshPart);
          nodal_state = true;
        }
        else if(fieldType == "Elem Vector") {
          entity = Albany::StateStruct::ElemData;
          p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->cell_vector, sideEBName, true, &entity, meshPart);
          nodal_state = false;
        }
        else if(fieldType == "Node Vector") {
          entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
          p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->node_vector, sideEBName, true, &entity, meshPart);
          nodal_state = true;
        }
        else if(fieldType == "Elem Layered Scalar") {
          entity = Albany::StateStruct::ElemData;
          sns = ss_dl->cell_scalar2;
          dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,LayerDim>(sns->dimension(0),sns->dimension(1),numLayers));
          stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
        }
        else if(fieldType == "Node Layered Scalar") {
          entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
          sns = ss_dl->node_scalar;
          dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,LayerDim>(sns->dimension(0),sns->dimension(1),sns->dimension(2),numLayers));
          stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
        }
        else if(fieldType == "Elem Layered Vector") {
          entity = Albany::StateStruct::ElemData;
          sns = ss_dl->cell_vector;
          dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Dim,LayerDim>(sns->dimension(0),sns->dimension(1),sns->dimension(2),numLayers));
          stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
        }
        else if(fieldType == "Node Layered Vector") {
          entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
          sns = ss_dl->node_vector;
          dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,Dim,LayerDim>(sns->dimension(0),sns->dimension(1),sns->dimension(2),
                                                                                 sns->dimension(3),numLayers));
          stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, dl_temp, sideEBName, true, &entity, meshPart);
        }

        // Get current state specs
        stateName  = fieldName = thisFieldList.get<std::string>("Field Name");
        fieldType  = thisFieldList.get<std::string>("Field Type");
        fieldUsage = thisFieldList.get<std::string>("Field Usage");

        if (fieldUsage == "Unused")
          continue;

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
  else
  {
    // Temporary fix for non STK meshes

    if (ss_requirements.find(basalSideName)!=ss_requirements.end())
    {
      stateName = fieldName = "ice_thickness";
      const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);
      if (std::find(req.begin(), req.end(), stateName)!=req.end())
      {
        // ...and thickness is one of them.
        if (std::find(requirements.begin(),requirements.end(),stateName)==requirements.end()) {
          entity = Albany::StateStruct::NodalDataToElemNode;
          p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
          ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
        }
      }
    }

    // Basal friction
    stateName = fieldName = "basal_friction";
    if(is_dist_param[stateName])
    {
      //basal friction is a distributed parameter
      TEUCHOS_TEST_FOR_EXCEPTION (ss_requirements.find(basalSideName)==ss_requirements.end(), std::logic_error,
                                  "Error! 'basal_friction' is a parameter, but there are no basal requirements.\n");
      const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);

      TEUCHOS_TEST_FOR_EXCEPTION (std::find(req.begin(), req.end(), stateName)==req.end(), std::logic_error,
                                  "Error! 'basal_friction' is a parameter, but is not listed as basal requirements.\n");

      //basal friction is a distributed 3D parameter
      entity = Albany::StateStruct::NodalDistParameter;
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, dist_params_name_to_mesh_part[stateName]);
      ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
      fm0.template registerEvaluator<EvalT>(ev);

      std::stringstream key;
      key << stateName <<  "Is Distributed Parameter";
      this->params->set<int>(key.str(), 1);

    }

    fieldName = stateName = "bed_roughness";
    if(is_dist_param[stateName])
    {
      //basal friction is a distributed parameter
      TEUCHOS_TEST_FOR_EXCEPTION (ss_requirements.find(basalSideName)==ss_requirements.end(), std::logic_error,
                                  "Error! 'bed_roughness' is a parameter, but there are no basal requirements.\n");
      const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);

      TEUCHOS_TEST_FOR_EXCEPTION (std::find(req.begin(), req.end(), stateName)==req.end(), std::logic_error,
                                  "Error! 'bed_roughness' is a parameter, but is not listed as basal requirements.\n");

      // bed_roughness is a distributed 3D parameter
      entity = Albany::StateStruct::NodalDistParameter;
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, dist_params_name_to_mesh_part[stateName]);
      ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
      fm0.template registerEvaluator<EvalT>(ev);

      // We save it, in case we optimize on it and it changes
      p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
      p->set<bool>("Is Vector Field", false);
      p->set<bool>("Nodal State", true);
      p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
      ev = Teuchos::rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
      fm0.template registerEvaluator<EvalT>(ev);

      // Only PHAL::AlbanyTraits::Residual evaluates something
      if (ev->evaluatedFields().size()>0)
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);

      std::stringstream key;
      key << stateName <<  "Is Distributed Parameter";
      this->params->set<int>(key.str(), 1);

      //---- Interpolate the 3D state on the side (the BasalFrictionCoefficient evaluator needs a side field)
      ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator(fieldName,basalSideName,"Node Scalar",cellType);
      fm0.template registerEvaluator<EvalT> (ev);

      //---- Interpolate on QP on side
      ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(fieldName, basalSideName);
      fm0.template registerEvaluator<EvalT>(ev);

      //---- Interpolate gradient on QP on side (in case it's a parameter and we want to add H1 regularization)
      ev = evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator(fieldName, basalSideName);
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Effective pressure
    stateName = fieldName = "effective_pressure";
    if(is_dist_param[stateName])
    {
      //basal friction is a distributed parameter
      TEUCHOS_TEST_FOR_EXCEPTION (ss_requirements.find(basalSideName)==ss_requirements.end(), std::logic_error,
                                  "Error! 'bed_roughness' is a parameter, but there are no basal requirements.\n");
      const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);

      TEUCHOS_TEST_FOR_EXCEPTION (std::find(req.begin(), req.end(), stateName)==req.end(), std::logic_error,
                                  "Error! 'bed_roughness' is a parameter, but is not listed as basal requirements.\n");

      //basal friction is a distributed 3D parameter
      entity = Albany::StateStruct::NodalDistParameter;
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, dist_params_name_to_mesh_part[stateName]);
      ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
      fm0.template registerEvaluator<EvalT>(ev);

      std::stringstream key;
      key << stateName <<  "Is Distributed Parameter";
      this->params->set<int>(key.str(), 1);

      //---- Interpolate the 3D state on the side (the BasalFrictionCoefficient evaluator needs a side field)
      ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator(fieldName,basalSideName,"Node Scalar",cellType);
      fm0.template registerEvaluator<EvalT> (ev);

      //---- Interpolate on QP on side
      ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(fieldName, basalSideName);
      fm0.template registerEvaluator<EvalT>(ev);

      //---- Interpolate gradient on QP on side (in case it's a parameter and we want to add H1 regularization)
      ev = evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator(fieldName, basalSideName);
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Remaining basal states
    if (ss_requirements.find(basalSideName)!=ss_requirements.end())
    {
      const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);

      stateName = fieldName = "basal_friction";
      if (std::find(req.begin(), req.end(), stateName)!=req.end())
      {
        // ...and basal_friction is one of them.
        entity = Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
        if (is_dist_param[stateName])
        {
          // basal friction is a distributed 3D parameter. We already took care of this case.
          // However, we may want to save it, since it may change if we're optimizing on it.
          p->set<bool>("Nodal State", true);
          p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
          ev = Teuchos::rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
          fm0.template registerEvaluator<EvalT>(ev);

          // Only PHAL::AlbanyTraits::Residual evaluates something
          if (ev->evaluatedFields().size()>0)
            fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
        }
        else if (std::find(requirements.begin(),requirements.end(),stateName)==requirements.end())
        {
          //---- Load the side state
          ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);

          //---- Interpolate Beta Given on QP on side (may be used by a response)
          ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(fieldName, basalSideName);
          fm0.template registerEvaluator<EvalT>(ev);
        }
      }

      stateName = fieldName = "beta";
      if (std::find(req.begin(), req.end(), stateName)!=req.end())
      {
        entity = Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
        p->set<bool>("Nodal State", true);
        p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
        ev = Teuchos::rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
        fm0.template registerEvaluator<EvalT>(ev);

        // Only PHAL::AlbanyTraits::Residual evaluates something
        if (ev->evaluatedFields().size()>0)
          fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
      }

      stateName = fieldName = "effective_pressure";
      if (std::find(req.begin(), req.end(), stateName)!=req.end())
      {
        entity = Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
        p->set<bool>("Nodal State", true);
        p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
        ev = Teuchos::rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
        fm0.template registerEvaluator<EvalT>(ev);

        // Only PHAL::AlbanyTraits::Residual evaluates something
        if (ev->evaluatedFields().size()>0)
          fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
      }

      stateName = fieldName = "basal_velocity";
      if (std::find(req.begin(), req.end(), stateName)!=req.end())
      {
        entity = Albany::StateStruct::NodalDataToElemNode;
        p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_vector, basalEBName, true, &entity);
        p->set<bool>("Nodal State", true);
        p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
        ev = Teuchos::rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
        fm0.template registerEvaluator<EvalT>(ev);

        // Only PHAL::AlbanyTraits::Residual evaluates something
        if (ev->evaluatedFields().size()>0)
          fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
      }
    }

    stateName = fieldName = "basal_friction";
    if (!is_dist_param[stateName] && std::find(requirements.begin(),requirements.end(),stateName)!=requirements.end())
    {
      entity = Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);

      // We are (for some mystic reason) extruding beta to the whole 3D mesh, even if it is not a parameter
      ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      // We restrict it back to the 2D mesh. Clearly, this is not optimal. Just add 'basal_friction' to the Basal Requirements!
      if(basalSideName!="INVALID") {
        ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator(fieldName,basalSideName,"Node Scalar",cellType);
        fm0.template registerEvaluator<EvalT> (ev);
      }
    }

    // Load surface mass balance
    entity= Albany::StateStruct::NodalDataToElemNode;
    stateName = fieldName = "surface_mass_balance";
    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Load surface mass balance RMS
    entity= Albany::StateStruct::NodalDataToElemNode;
    stateName = fieldName = "surface_mass_balance_RMS";
    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Load observed thickness
    entity = Albany::StateStruct::NodalDataToElemNode;
    stateName = fieldName = "observed_ice_thickness";
    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Load thickness RMS
    entity= Albany::StateStruct::NodalDataToElemNode;
    stateName = fieldName = "ice_thickness_RMS";
    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
 /*
if (basalSideName!="INVALID")
{
    entity = Albany::StateStruct::NodalDataToElemNode;
    stateName = fieldName = "surface_air_temperature";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

if (basalSideName!="INVALID")
{
    entity = Albany::StateStruct::NodalDataToElemNode;
    stateName = fieldName = "surface_air_enthalpy";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

if (basalSideName!="INVALID")
{
    entity = Albany::StateStruct::NodalDataToElemNode;
    stateName = fieldName ="basal_heat_flux";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }
*/

/*
  // Basal friction sensitivity
  stateName = "basal_friction_sensitivity";
  entity = Albany::StateStruct::NodalDistParameter;
  stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity,"bottom");

  // Thickness sensitivity
  stateName = "thickness_sensitivity";
  entity = Albany::StateStruct::NodalDistParameter;
  stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity,"bottom");
*/
  }

  // ----------  Define Field Names ----------- //
  Teuchos::ArrayRCP<std::string> dof_names(1), dof_name_auxiliary(1);
  Teuchos::ArrayRCP<std::string> resid_names(1);
  dof_names[0] = "Velocity";
  dof_name_auxiliary[0] = "L2 Projected Boundary Laplacian";
  resid_names[0] = "Stokes Residual";

  // ---------- Add time as a Sacado-ized parameter (only if specified) ------- //
  bool isTimeAParameter = false;
  if (params->isParameter("Use Time Parameter")) isTimeAParameter = params->get<bool>("Use Time Parameter");
  if (isTimeAParameter) {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("Time"));
    p->set<Teuchos::RCP<PHX::DataLayout>>("Workset Scalar Data Layout", dl->workset_scalar);
    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);
    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
    ev = Teuchos::rcp(new FELIX::Time<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Time", dl->workset_scalar, dl->dummy, elementBlockName, "scalar", 0.0, true);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // ------------------- Interpolations and utilities ------------------ //

  // Gather solution field
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names, offset);
  fm0.template registerEvaluator<EvalT> (ev);

  // Gather solution field
  if(neq > vecDimFO) {
    ev = evalUtils_scalar.constructGatherSolutionEvaluator_noTransient(false, dof_name_auxiliary, 2);
    fm0.template registerEvaluator<EvalT> (ev);

    ev = evalUtils.constructDOFCellToSideEvaluator(dof_name_auxiliary[0],basalSideName,"Node Scalar",cellType, dof_name_auxiliary[0]);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // Interpolate solution field
  ev = evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate solution gradient
  ev = evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter residual
  ev = evalUtils.constructScatterResidualEvaluatorWithExtrudedParams(true, resid_names, extruded_params_levels, offset, "Scatter Stokes");
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter residual
  Teuchos::ArrayRCP<std::string> resid2_name(1);
  resid2_name[0] = "L2 Projected Boundary Laplacian Residual";
  ev = evalUtils_scalar.constructScatterResidualEvaluatorWithExtrudedParams(false, resid2_name, extruded_params_levels, vecDimFO, "Auxiliary Residual");
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate temperature from nodes to cell
  ev = evalUtils.getPSTUtils().constructNodesToCellInterpolationEvaluator ("temperature",false);
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate flow factor from nodes to cell
  ev = evalUtils.getPSTUtils().constructNodesToCellInterpolationEvaluator ("flow_factor",false);
  fm0.template registerEvaluator<EvalT> (ev);

  if(!is_dist_param["thickness"])
  {
    //----- Gather Coordinate Vector (general parameters)
    ev = evalUtils.constructGatherCoordinateVectorEvaluator();
    fm0.template registerEvaluator<EvalT> (ev);
  }
  else
  {
#ifndef ALBANY_MESH_DEPENDS_ON_PARAMETERS
    //----- Gather Coordinate Vector (general parameters)
    ev = evalUtils.constructGatherCoordinateVectorEvaluator();
    fm0.template registerEvaluator<EvalT> (ev);
#else

    //----- Gather Coordinate Vector (ad hoc parameters)
    p = Teuchos::rcp(new Teuchos::ParameterList("Gather Coordinate Vector"));

    // Input: Periodic BC flag
    p->set<bool>("Periodic BC", false);

    // Output:: Coordindate Vector at vertices
    p->set<std::string>("Coordinate Vector Name", "Coord Vec Old");

    ev = Teuchos::rcp(new PHAL::GatherCoordinateVector<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    //------ Update Z Coordinate
    p = Teuchos::rcp(new Teuchos::ParameterList("Update Z Coordinate"));

    p->set<std::string>("Old Coords Name", "Coord Vec Old");
    p->set<std::string>("New Coords Name", "Coord Vec");
    if(is_dist_param["thickness"])
      p->set<std::string>("Thickness Name", "ice_thickness Param");
    else
      p->set<std::string>("Thickness Name", "ice_thickness");

    p->set<std::string>("Top Surface Name", "surface_height");

    ev = Teuchos::rcp(new FELIX::UpdateZCoordinateMovingBed<EvalT,PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
#endif
  }

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

  if (basalSideName!="INVALID")
  {
    // -------------------- Special evaluators for side handling ----------------- //

    //---- Restrict vertex coordinates from cell-based to cell-side-based
    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator("Coord Vec",basalSideName,"Vertex Vector",cellType,"Coord Vec " + basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute side basis functions
    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, basalSideBasis, basalCubature, basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Restrict velocity from cell-based to cell-side-based
    ev = evalUtils.constructDOFCellToSideEvaluator("Velocity",basalSideName,"Node Vector",cellType,"basal_velocity");
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity on QP on side
    ev = evalUtils.constructDOFVecInterpolationSideEvaluator("basal_velocity", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Compute Quad Points coordinates on the side set
    ev = evalUtils.constructMapToPhysicalFrameSideEvaluator(cellType,basalCubature,basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate stiffening gradient on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("stiffening_factor", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate stiffening gradient on QP on side
    ev = evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator("stiffening_factor", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    // Intepolate bed_topography
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("bed_topography", basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity gradient on QP on side
    ev = evalUtils.constructDOFVecGradInterpolationSideEvaluator("basal_velocity", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Restrict ice thickness from cell-based to cell-side-based
    ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("ice_thickness",basalSideName,"Node Scalar",cellType);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate thickness gradient on QP on side
    ev = evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator("ice_thickness", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate thickness on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("ice_thickness Param", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Restrict ice thickness (param) from cell-based to cell-side-based
    ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("ice_thickness Param",basalSideName,"Node Scalar",cellType);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate thickness (param) gradient on QP on side
    ev = evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator("ice_thickness Param", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate observed thickness on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("observed_ice_thickness", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate thickness on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("ice_thickness", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate thickness RMS on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("ice_thickness_RMS", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate effective pressure on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("effective_pressure", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate effective pressure gradient on QP on side
    ev = evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator("effective_pressure", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Restrict surface height from cell-based to cell-side-based
    ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("surface_height",basalSideName,"Node Scalar",cellType);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate surface height on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("surface_height", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    // Interpolate the 3D state on the side (the BasalFrictionCoefficient evaluator needs a side field)
    ev = evalUtils.constructDOFCellToSideEvaluator("Averaged Velocity",basalSideName,"Node Vector",cellType);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate surface height on QP on side
    ev = evalUtils.constructDOFDivInterpolationSideEvaluator("Averaged Velocity", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate velocity on QP on side
    ev = evalUtils.constructDOFVecInterpolationSideEvaluator("Averaged Velocity", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface velocity on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("surface_mass_balance", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface velocity on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("surface_mass_balance_RMS", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate basal_friction (if needed) on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("basal_friction", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate effective_pressure (if needed) on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("effective_pressure", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate bed_roughness (if needed) on QP on side
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("bed_roughness", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    // Parameters are 3D. If any field needed on basal side is a parameter, we must project it on side
    if (is_dist_param["basal_friction"])
    {
      // Interpolate the 3D state on the side (the BasalFrictionCoefficient evaluator needs a side field)
      ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("basal_friction",basalSideName,"Node Scalar",cellType);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (is_dist_param["effective_pressure"])
    {
      // Interpolate the 3D state on the side (the BasalFrictionCoefficient evaluator needs a side field)
      ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("effective_pressure",basalSideName,"Node Scalar",cellType);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (is_dist_param["bed_roughness"])
    {
      // Interpolate the 3D state on the side (the BasalFrictionCoefficient evaluator needs a side field)
      ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("bed_roughness",basalSideName,"Node Scalar",cellType);
      fm0.template registerEvaluator<EvalT> (ev);
    }
  }

  if (surfaceSideName!="INVALID")
  {
    //---- Restrict vertex coordinates from cell-based to cell-side-based
    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator("Coord Vec",surfaceSideName,"Vertex Vector",cellType,"Coord Vec " + surfaceSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute side basis functions
    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, surfaceSideBasis, surfaceCubature, surfaceSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate surface velocity on QP on side
    ev = evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("observed_surface_velocity", surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface velocity rms on QP on side
    ev = evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("observed_surface_velocity_RMS", surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Restrict velocity (the solution) from cell-based to cell-side-based on upper side
    ev = evalUtils.constructDOFCellToSideEvaluator("Velocity",surfaceSideName,"Node Vector",cellType,"Surface Velocity");
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity (the solution) on QP on side
    ev = evalUtils.constructDOFVecInterpolationSideEvaluator("surface_velocity", surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // -------------------------------- FELIX evaluators ------------------------- //

  // --- FO Stokes Stress --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Stress"));

  //Input
  p->set<std::string>("Velocity QP Variable Name", "Velocity");
  p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
  p->set<std::string>("Viscosity QP Variable Name", "FELIX Viscosity");
  p->set<std::string>("Surface Height QP Name", "Surface Height");
  p->set<std::string>("Coordinate Vector Name", "Coord Vec");
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string>("Stress Variable Name", "Stress Tensor");

  ev = Teuchos::rcp(new FELIX::StokesFOStress<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- FO Stokes Resid --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Resid"));

  //Input
  p->set<std::string>("Weighted BF Variable Name", "wBF");
  p->set<std::string>("Weighted Gradient BF Variable Name", "wGrad BF");
  p->set<std::string>("Velocity QP Variable Name", "Velocity");
  p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
  p->set<std::string>("Body Force Variable Name", "Body Force");
  p->set<std::string>("Viscosity QP Variable Name", "FELIX Viscosity");
  p->set<std::string>("Coordinate Vector Name", "Coord Vec");
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Equation Set"));
  p->set<std::string>("Basal Residual Variable Name", "Basal Residual");
  p->set<bool>("Needs Basal Residual", sliding);

  //Output
  p->set<std::string>("Residual Variable Name", "Stokes Residual");

  ev = Teuchos::rcp(new FELIX::StokesFOResid<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if(neq > vecDimFO) {

    p = Teuchos::rcp(new Teuchos::ParameterList("L2 Projected Boundary Laplacian Residual"));

   // const std::string& residual_name = params->get<std::string>("L2 Projected Boundary Laplacian Residual Name");

    //Input
    p->set<std::string>("Solution Variable Name", "L2 Projected Boundary Laplacian");
    p->set<std::string>("Field Name", "Beta Given");
    p->set<std::string>("Field Gradient Name", "Beta Gradient");
    p->set<std::string>("Gradient BF Side Name", "Grad BF "+basalSideName);
    p->set<std::string>("Weighted Measure Side Name", "Weighted Measure "+basalSideName);
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<double>("Mass Coefficient", params->sublist("FELIX L2 Projected Boundary Laplacian").get<double>("Mass Coefficient",1.0));
    p->set<double>("Laplacian Coefficient", params->sublist("FELIX L2 Projected Boundary Laplacian").get<double>("Laplacian Coefficient",1.0));
    p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));

    //Output
    p->set<std::string>("L2 Projected Boundary Laplacian Residual Name", "L2 Projected Boundary Laplacian Residual");

    ev = Teuchos::rcp(new FELIX::L2ProjectedBoundaryLaplacianResidual<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (sliding)
  {
    // --- Basal Residual --- //
    p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Basal Residual"));

    //Input
    p->set<std::string>("BF Side Name", "BF "+basalSideName);
    p->set<std::string>("Weighted Measure Name", "Weighted Measure "+basalSideName);
    p->set<std::string>("Basal Friction Coefficient Side QP Variable Name", "beta");
    p->set<std::string>("Velocity Side QP Variable Name", "basal_velocity");
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));

    //Output
    p->set<std::string>("Basal Residual Variable Name", "Basal Residual");

    ev = Teuchos::rcp(new FELIX::StokesFOBasalResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Sliding velocity calculation at nodes ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));

    // Input
    p->set<std::string>("Field Name","basal_velocity");
    p->set<std::string>("Field Layout","Cell Side Node Vector");
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Field Norm"));

    // Output
    p->set<std::string>("Field Norm Name","sliding_velocity");

    ev = Teuchos::rcp(new PHAL::Field2Norm<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Sliding velocity calculation ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));

    // Input
    p->set<std::string>("Field Name","basal_velocity");
    p->set<std::string>("Field Layout","Cell Side QuadPoint Vector");
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Field Norm"));

    // Output
    p->set<std::string>("Field Norm Name","sliding_velocity");

    ev = Teuchos::rcp(new PHAL::Field2Norm<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);

    if (!is_dist_param["effective_pressure"])
    {
      //--- Effective pressure (surrogate) calculation ---//
      p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Effective Pressure Surrogate"));

      // Input
      p->set<std::string>("Side Set Name", basalSideName);
      p->set<std::string>("Surface Height Variable Name","surface_height");
      p->set<std::string>("Ice Thickness Variable Name", "ice_thickness");
      p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));
      p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Effective Pressure Surrogate"));

      // Output
      p->set<std::string>("Effective Pressure Variable Name","effective_pressure");

      ev = Teuchos::rcp(new FELIX::EffectivePressure<EvalT,PHAL::AlbanyTraits,true,true>(*p,dl_basal));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    //--- Shared Parameter for basal friction coefficient: alpha ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: alpha"));

    param_name = "Hydraulic-Over-Hydrostatic Potential Ratio";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Alpha>> ptr_alpha;
    ptr_alpha = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Alpha>(*p,dl));
    ptr_alpha->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_alpha);

    //--- Shared Parameter for basal friction coefficient: lambda ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: lambda"));

    param_name = "Bed Roughness";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Lambda>> ptr_lambda;
    ptr_lambda = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Lambda>(*p,dl));
    ptr_lambda->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_lambda);

    //--- Shared Parameter for basal friction coefficient: mu ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: mu"));

    param_name = "Coulomb Friction Coefficient";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Mu>> ptr_mu;
    ptr_mu = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Mu>(*p,dl));
    ptr_mu->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_mu);

    //--- Shared Parameter for basal friction coefficient: power ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: power"));

    param_name = "Power Exponent";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Power>> ptr_power;
    ptr_power = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Power>(*p,dl));
    ptr_power->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_power);

    //--- FELIX basal friction coefficient ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient"));

    //Input
    p->set<std::string>("Sliding Velocity QP Variable Name", "sliding_velocity");
    p->set<std::string>("BF Variable Name", "BF " + basalSideName);
    p->set<std::string>("Effective Pressure QP Variable Name", "effective_pressure");
    p->set<std::string>("Bed Roughness Variable Name", "bed_roughness");
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec " + basalSideName);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));
    p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("FELIX Physical Parameters"));
    p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
    p->set<std::string>("Bed Topography QP Name", "bed_topography");
    p->set<std::string>("Thickness QP Name", "ice_thickness");

    //Output
    p->set<std::string>("Basal Friction Coefficient Variable Name", "beta");

    ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits,false,true>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- FELIX basal friction coefficient at nodes ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient Node"));

    //Input
    p->set<std::string>("Sliding Velocity Variable Name", "sliding_velocity");
    p->set<std::string>("Effective Pressure Variable Name", "effective_pressure");
    p->set<std::string>("Bed Roughness Variable Name", "bed_roughness");
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));

    //Output
    p->set<std::string>("Basal Friction Coefficient Variable Name", "beta");

    ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficientNode<EvalT,PHAL::AlbanyTraits,false,true>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (basalSideName!="INVALID")
  {
    fieldName = "Flux Divergence";
    stateName = "flux_divergence";
    p = Teuchos::rcp(new Teuchos::ParameterList("Flux Divergence"));

    //Input
    p->set<std::string>("Averaged Velocity Side QP Variable Name", "Averaged Velocity");
    p->set<std::string>("Averaged Velocity Side QP Divergence Name", "Averaged Velocity Divergence");
    if(is_dist_param["thickness"]) {
      p->set<std::string>("Thickness Side QP Variable Name", "ice_thickness Param");
      p->set<std::string>("Thickness Gradient Name", "ice_thickness Param Gradient");
    } else {
      p->set<std::string>("Thickness Side QP Variable Name", "ice_thickness");
      p->set<std::string>("Thickness Gradient Name", "ice_thickness Gradient");
    }

    p->set<std::string>("Field Name",  "Flux Divergence");
    p->set<std::string> ("Side Set Name", basalSideName);

    ev = Teuchos::rcp(new FELIX::FluxDiv<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);
  }

#ifdef ALBANY_EPETRA
  p = Teuchos::rcp(new Teuchos::ParameterList("Gather Averaged Velocity"));
  p->set<std::string>("Averaged Velocity Name", "Averaged Velocity");
  p->set<std::string>("Mesh Part", "basalside");
  p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));
  ev = Teuchos::rcp(new GatherVerticallyAveragedVelocity<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);
#endif

  //--- Shared Parameter for Continuation:  ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Homotopy Parameter"));

  param_name = "Glen's Law Homotopy Parameter";
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Homotopy>> ptr_homotopy;
  ptr_homotopy = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Homotopy>(*p,dl));
  ptr_homotopy->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Viscosity").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_homotopy);

  //--- FELIX viscosity ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Viscosity"));

  //Input
  p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");
  p->set<std::string>("Velocity QP Variable Name", "Velocity");
  p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
  p->set<std::string>("Temperature Variable Name", "temperature");
  p->set<std::string>("Flow Factor Variable Name", "flow_factor");
  p->set<std::string>("Stiffening Factor QP Name", "stiffening_factor");
  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Viscosity"));
  p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

  //Output
  p->set<std::string>("Viscosity QP Variable Name", "FELIX Viscosity");
  p->set<std::string>("EpsilonSq QP Variable Name", "FELIX EpsilonSq");

  ev = Teuchos::rcp(new FELIX::ViscosityFO<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ParamScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- Print FELIX Dissipation ---
  if(params->sublist("FELIX Viscosity").get("Extract Strain Rate Sq", false))
  {
    {
    p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Dissipation"));

    //Input
    p->set<std::string>("Viscosity QP Variable Name", "FELIX Viscosity");
    p->set<std::string>("EpsilonSq QP Variable Name", "FELIX EpsilonSq");

    //Output
    p->set<std::string>("Dissipation QP Variable Name", "FELIX Dissipation");

    ev = Teuchos::rcp(new FELIX::Dissipation<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
    }

    fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructQuadPointsToCellInterpolationEvaluator("FELIX Dissipation",false));

    // Saving the dissipation heat in the output mesh
    {
//         fm0.template registerEvaluator<EvalT> (evalUtils.constructNodesToCellInterpolationEvaluator("melting temp",false));

      std::string stateName = "dissipation_heat";
      p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, dl->dummy, elementBlockName, "scalar", 0.0, /* save state = */ false, /* write output = */ true);
      p->set<std::string>("Field Name", "FELIX Dissipation");
      p->set<std::string>("Weights Name","Weights");
      p->set("Weights Layout", dl->qp_scalar);
      p->set("Field Layout", dl->cell_scalar2);
      p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);
      ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT,PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
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

  // Saving the stress tensor in the output mesh
  if(params->get<bool>("Print Stress Tensor", false))
  {
    {
      std::string stateName = "Stress Tensor";
      p = stateMgr.registerStateVariable(stateName, dl->qp_tensor, dl->dummy, elementBlockName, "tensor", 0.0, /* save state = */ false, /* write output = */ true);
      p->set<std::string>("Field Name", "Stress Tensor");
      p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);
      ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
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

#ifdef CISM_HAS_FELIX
  //--- FELIX surface gradient from CISM ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Surface Gradient"));

  //Input
  p->set<std::string>("CISM Surface Height Gradient X Variable Name", "CISM Surface Height Gradient X");
  p->set<std::string>("CISM Surface Height Gradient Y Variable Name", "CISM Surface Height Gradient Y");
  p->set<std::string>("BF Variable Name", "BF");

  //Output
  p->set<std::string>("Surface Height Gradient QP Variable Name", "CISM Surface Height Gradient");
  ev = Teuchos::rcp(new FELIX::CismSurfaceGradFO<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);
#endif

  //--- Body Force ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Body Force"));

  //Input
  p->set<std::string>("FELIX Viscosity QP Variable Name", "FELIX Viscosity");
#ifdef CISM_HAS_FELIX
  p->set<std::string>("Surface Height Gradient QP Variable Name", "CISM Surface Height Gradient");
#endif
  p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");
  p->set<std::string>("Surface Height Gradient Name", "surface_height Gradient");
  p->set<std::string>("Surface Height Name", "Surface Height");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Body Force"));
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string>("Body Force Variable Name", "Body Force");

  ev = Teuchos::rcp(new FELIX::StokesFOBodyForce<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if (surfaceSideName!="INVALID")
  {
    // Load surface velocity
    entity= Albany::StateStruct::NodalDataToElemNode;
    stateName = "surface_velocity";
    fieldName = "Observed Surface Velocity";
    p = stateMgr.registerSideSetStateVariable(surfaceSideName, stateName, fieldName, dl_surface->node_vector, surfaceEBName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Load surface velocity rms
    entity= Albany::StateStruct::NodalDataToElemNode;
    stateName = "surface_velocity_rms";
    fieldName = "Observed Surface Velocity RMS";
    p = stateMgr.registerSideSetStateVariable(surfaceSideName, stateName, fieldName, dl_surface->node_vector, surfaceEBName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (basalSideName!="INVALID")
  {
    //--- FELIX basal friction coefficient gradient ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient Gradient"));

    // Input
    p->set<std::string>("Gradient BF Side Variable Name", "Grad BF "+basalSideName);
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<std::string>("Effective Pressure QP Name", "effective_pressure");
    p->set<std::string>("Effective Pressure Gradient QP Name", "effective_pressure Gradient");
    p->set<std::string>("Basal Velocity QP Name", "basal_velocity");
    p->set<std::string>("Basal Velocity Gradient QP Name", "basal_velocity Gradient");
    p->set<std::string>("Sliding Velocity QP Name", "sliding_velocity");
    p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec "+basalSideName);
    p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));

    // Output
    p->set<std::string>("Basal Friction Coefficient Gradient Name","beta Gradient");

    ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficientGradient<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  //--- FELIX noise (for synthetic inverse problem) ---//
  if (params->isSublist("FELIX Noise"))
  {
    if (params->sublist("FELIX Noise").isSublist("Observed Surface Velocity"))
    {
      // ---- Add noise to the measures ---- //
      p = Teuchos::rcp(new Teuchos::ParameterList("Noisy Observed Velocity"));

      //Input
      p->set<std::string>("Field Name", "observed_surface_velocity");
      p->set<Teuchos::RCP<PHX::DataLayout>>("Field Layout", dl_surface->qp_vector);
      p->set<Teuchos::ParameterList*>("PDF Parameters", &params->sublist("FELIX Noise").sublist("Observed Surface Velocity"));

      // Output
      p->set<std::string>("Noisy Field Name", "observed_surface_velocity_noisy");

      ev = Teuchos::rcp(new PHAL::AddNoiseParam<EvalT,PHAL::AlbanyTraits> (*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Stokes", dl->dummy);
    fm0.requireField<EvalT>(res_tag);

    if(neq > vecDimFO) {
      PHX::Tag<typename EvalT::ScalarT> res_tag2("Auxiliary Residual", dl_scalar->dummy);
      fm0.requireField<EvalT>(res_tag2);
    }
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    // ----------------------- Responses --------------------- //
    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));
    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
    paramList->set<std::string>("Basal Friction Coefficient Gradient Name","beta Gradient");
    paramList->set<std::string>("Stiffening Factor Gradient Name","stiffening_factor Gradient");
    paramList->set<std::string>("Stiffening Factor Name","stiffening_factor");
    if(is_dist_param["thickness"]) {
      paramList->set<std::string>("Thickness Gradient Name","ice_thickness Param Gradient");
      paramList->set<std::string>("Thickness Side QP Variable Name","ice_thickness Param");
    } else {
      paramList->set<std::string>("Thickness Gradient Name","ice_thickness Gradient");
      paramList->set<std::string>("Thickness Side QP Variable Name","ice_thickness");
    }
    paramList->set<std::string>("Surface Velocity Side QP Variable Name","surface_velocity");
    paramList->set<std::string>("SMB Side QP Variable Name","surface_mass_balance");
    paramList->set<std::string>("SMB RMS Side QP Variable Name","surface_mass_balance_RMS");
    paramList->set<std::string>("Flux Divergence Side QP Variable Name","Flux Divergence");
    paramList->set<std::string>("Thickness RMS Side QP Variable Name","ice_thickness_RMS");
    paramList->set<std::string>("Observed Thickness Side QP Variable Name","observed_ice_thickness");
    paramList->set<std::string>("Observed Surface Velocity Side QP Variable Name","observed_surface_velocity");
    paramList->set<std::string>("Observed Surface Velocity RMS Side QP Variable Name","observed_surface_velocity_RMS");
    paramList->set<std::string>("Weighted Measure Basal Name","Weighted Measure " + basalSideName);
    paramList->set<std::string>("Weighted Measure 2D Name","Weighted Measure " + basalSideName);
    paramList->set<std::string>("Weighted Measure Surface Name","Weighted Measure " + surfaceSideName);
    paramList->set<std::string>("Inverse Metric Basal Name","Inv Metric " + basalSideName);
    paramList->set<std::string>("Basal Side Name", basalSideName);
    paramList->set<std::string>("Surface Side Name", surfaceSideName);
    paramList->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));

    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

#endif // FELIX_STOKES_FO_PROBLEM_HPP
