//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKES_FO_HYDROLOGY_PROBLEM_HPP
#define FELIX_STOKES_FO_HYDROLOGY_PROBLEM_HPP 1

#include <type_traits>

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
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

#include "PHAL_LoadStateField.hpp"
#include "PHAL_DOFCellToSide.hpp"
#include "PHAL_LoadSideSetStateField.hpp"
#include "PHAL_SaveSideSetStateField.hpp"

#ifdef CISM_HAS_FELIX
#include "FELIX_CismSurfaceGradFO.hpp"
#endif
#include "FELIX_EffectivePressure.hpp"
#include "FELIX_StokesFOResid.hpp"
#include "FELIX_StokesFOBasalResid.hpp"
#include "FELIX_StokesFOBodyForce.hpp"
#include "FELIX_ViscosityFO.hpp"
#include "PHAL_Field2Norm.hpp"
#include "FELIX_BasalFrictionCoefficient.hpp"
#include "FELIX_BasalFrictionCoefficientNode.hpp"
#include "FELIX_HydrologyWaterDischarge.hpp"
#include "FELIX_HydrologyResidualPotentialEqn.hpp"
#include "FELIX_HydrologyResidualThicknessEqn.hpp"
#include "FELIX_HydrologyMeltingRate.hpp"
#include "FELIX_SharedParameter.hpp"
#include "FELIX_ParamEnum.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX
{

/*!
 * \brief The coupled problem StokesFO+Hydrology
 */
class StokesFOHydrology : public Albany::AbstractProblem
{
public:

  //! Default constructor
  StokesFOHydrology (const Teuchos::RCP<Teuchos::ParameterList>& params,
                     const Teuchos::RCP<ParamLib>& paramLib,
                     const int numDim_);

  //! Destructor
  ~StokesFOHydrology();

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
  StokesFOHydrology(const StokesFOHydrology&) = delete;

  //! Private to prohibit copying
  StokesFOHydrology& operator=(const StokesFOHydrology&) = delete;

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

  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>  cellCubature;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>  basalCubature;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>  surfaceCubature;

  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> cellBasis;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> basalSideBasis;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> surfaceSideBasis;

  int numDim;
  int stokes_neq;
  int hydro_neq;

  bool has_h_equation;
  bool unsteady;

  Teuchos::RCP<Albany::Layouts> dl,dl_basal,dl_surface;

  std::string basalSideName;
  std::string surfaceSideName;

  std::string elementBlockName;
  std::string basalEBName;
  std::string surfaceEBName;

  Teuchos::ArrayRCP<std::string> stokes_dof_names;
  Teuchos::ArrayRCP<std::string> stokes_resid_names;

  Teuchos::ArrayRCP<std::string> hydro_dof_names;
  Teuchos::ArrayRCP<std::string> hydro_dof_names_dot;
  Teuchos::ArrayRCP<std::string> hydro_resid_names;
};

} // Namespace FELIX

// ================================ IMPLEMENTATION ============================ //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
FELIX::StokesFOHydrology::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                               const Albany::MeshSpecsStruct& meshSpecs,
                                               Albany::StateManager& stateMgr,
                                               Albany::FieldManagerChoice fieldManagerChoice,
                                               const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  int offset=0;

  Albany::StateStruct::MeshFieldEntity entity;
  RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  RCP<Teuchos::ParameterList> p;

  // ---------------------------- Registering state variables ------------------------- //

  std::string stateName, fieldName, param_name, mesh_part;
  std::vector<std::string>& basalReq = this->ss_requirements.at(basalSideName);

  // Surface height (2D & 3D)
  entity = Albany::StateStruct::NodalDataToElemNode;
  stateName = "surface_height";
  fieldName = "Surface Height";
  p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
  p->set<std::string>("Field Name", fieldName);
  ev = rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
  p->set<std::string>("Field Name", fieldName);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Ice thickness (2D & 3D)
  entity = Albany::StateStruct::NodalDataToElemNode;
  stateName = "ice_thickness";
  fieldName = "Ice Thickness";
  p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
  p->set<std::string>("Field Name", fieldName);
  ev = rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
  p->set<std::string>("Field Name", fieldName);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Bed roughness
  stateName = "bed_roughness";
  bool isLambdaAParameter = false; // Determining whether bed_roughness is a distributed parameter
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
        param_list = &dist_params_list.sublist(parameter_sublist_name);
        if (param_list->get<std::string>("Name", "") == stateName)
        {
          mesh_part = param_list->get<std::string>("Mesh Part","");
          isLambdaAParameter = true;
          break;
        }
      }
      else
      {
        if (stateName == dist_params_list.get(Albany::strint("Parameter", p_index), ""))
        {
          isLambdaAParameter = true;
          mesh_part = "";
          break;
        }
      }
    }
  }

  if(isLambdaAParameter)
  {
    // bed_roughness is a distributed parameter
    TEUCHOS_TEST_FOR_EXCEPTION (ss_requirements.find(basalSideName)==ss_requirements.end(), std::logic_error,
                                "Error! 'bed_roughness' is a parameter, but there are no basal requirements.\n");
    const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);

    TEUCHOS_TEST_FOR_EXCEPTION (std::find(req.begin(), req.end(), stateName)==req.end(), std::logic_error,
                                "Error! 'bed_roughness' is a parameter, but is not listed as basal requirements.\n");

    // bed_roughness is a distributed 3D parameter
    entity = Albany::StateStruct::NodalDistParameter;
    fieldName = "Bed Roughness";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity,mesh_part);
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

  // Remaining optional basal output states
  if (ss_requirements.find(basalSideName)!=ss_requirements.end())
  {
    const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);

    stateName = "basal_friction";
    if (std::find(req.begin(), req.end(), stateName)!=req.end())
    {
      // basal_friction is one of them (perhaps for comparison purposes)
      entity = Albany::StateStruct::NodalDataToElemNode;
      fieldName = "Beta Given";
      if (std::find(requirements.begin(),requirements.end(),stateName)==requirements.end())
      {
        p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);

        //---- Load the side state
        ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);

        //---- Interpolate Beta Given on QP on side (may be used by a response)
        ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(fieldName, basalSideName);
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }

    stateName = "beta";
    if (std::find(req.begin(), req.end(), stateName)!=req.end())
    {
      entity = Albany::StateStruct::NodalDataToElemNode;
      fieldName = "Beta";
      p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
      p->set<bool>("Is Vector Field", false);
      p->set<bool>("Nodal State", true);
      p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
      ev = Teuchos::rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
      fm0.template registerEvaluator<EvalT>(ev);

      // Only PHAL::AlbanyTraits::Residual evaluates something
      if (ev->evaluatedFields().size()>0)
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }

    stateName = "effective_pressure";
    if (std::find(req.begin(), req.end(), stateName)!=req.end())
    {
      entity = Albany::StateStruct::NodalDataToElemNode;
      fieldName = "Effective Pressure";
      p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
      p->set<bool>("Is Vector Field", false);
      p->set<bool>("Nodal State", true);
      p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
      ev = Teuchos::rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
      fm0.template registerEvaluator<EvalT>(ev);

      // Only PHAL::AlbanyTraits::Residual evaluates something
      if (ev->evaluatedFields().size()>0)
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }

    stateName = "basal_velocity";
    if (std::find(req.begin(), req.end(), stateName)!=req.end())
    {
      entity = Albany::StateStruct::NodalDataToElemNode;
      fieldName = "Basal Velocity";
      p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_vector, basalEBName, true, &entity);
      p->set<bool>("Is Vector Field", true);
      p->set<bool>("Nodal State", true);
      p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
      ev = Teuchos::rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
      fm0.template registerEvaluator<EvalT>(ev);

      // Only PHAL::AlbanyTraits::Residual evaluates something
      if (ev->evaluatedFields().size()>0)
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
  }

  // Bed topography
  stateName = "bed_topography";
  stateName = "Bed Topography";
  entity= Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
  p->set<const std::string>("Field Name",fieldName);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  if (params->sublist("FELIX Hydrology").get<bool>("Use SMB To Approximate Water Input",false))
  {
    // Surface Mass Balance
    entity = Albany::StateStruct::NodalDataToElemNode;
    stateName = "surface_mass_balance";
    fieldName = "Surface Mass Balance";
    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
    p->set<const std::string>("Field Name",fieldName);
    ev = rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  else
  {
    // Surface water input
    entity = Albany::StateStruct::NodalDataToElemNode;
    stateName = "surface_water_input";
    fieldName = "Surface Water Input";
    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
    p->set<const std::string>("Field Name",fieldName);
    ev = rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Hydrology geothermal flux
  stateName = "geothermal_flux";
  fieldName = "Geothermal Flux";
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerSideSetStateVariable(basalSideName,stateName,fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
  p->set<const std::string>("Field Name",fieldName);
  ev = rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  if (!has_h_equation)
  {
    // Since we don't have the water thickness eqn, we need to load h as a state
    stateName = "water_thickness";
    fieldName = "Water Thickness";
    entity = Albany::StateStruct::NodalDataToElemNode;
    p = stateMgr.registerSideSetStateVariable(basalSideName,stateName,fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
    p->set<const std::string>("Field Name",fieldName);
    ev = rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (surfaceSideName!="INVALID")
  {
    // Load surface velocity
    entity= Albany::StateStruct::NodalDataToElemNode;
    stateName = "surface_velocity";
    fieldName = "Observed Surface Velocity";
    p = stateMgr.registerSideSetStateVariable(surfaceSideName, stateName, fieldName, dl_surface->node_vector, surfaceEBName, true, &entity);
    ev = rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Load surface velocity rms
    entity= Albany::StateStruct::NodalDataToElemNode;
    stateName = "surface_velocity_rms";
    fieldName = "Observed Surface Velocity RMS";
    p = stateMgr.registerSideSetStateVariable(surfaceSideName, stateName, fieldName, dl_surface->node_vector, surfaceEBName, true, &entity);
    ev = rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // ------------------- Interpolations and utilities ------------------ //

  int offsetStokes = 0;
  int offsetHydro  = stokes_neq;

  // Gather stokes solution field
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(true, stokes_dof_names, offsetStokes);
  fm0.template registerEvaluator<EvalT> (ev);

  // Gather hydrology solution field
  if (has_h_equation && unsteady)
  {
    ev = evalUtils.constructGatherSolutionEvaluator (false, hydro_dof_names, hydro_dof_names_dot, offsetHydro);
    fm0.template registerEvaluator<EvalT> (ev);
  }
  else
  {
    ev = evalUtils.constructGatherSolutionEvaluator_noTransient (false, hydro_dof_names, offsetHydro);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // Scatter stokes residual
  ev = evalUtils.constructScatterResidualEvaluator(true, stokes_resid_names, offsetStokes, "Scatter Stokes");
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter hydrology residual
  ev = evalUtils.constructScatterResidualEvaluator(false, hydro_resid_names, offsetHydro, "Scatter Hydrology");
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate stokes solution field
  ev = evalUtils.constructDOFVecInterpolationEvaluator("Velocity");
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate stokes solution gradient
  ev = evalUtils.constructDOFVecGradInterpolationEvaluator("Velocity");
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate effective pressure
  ev = evalUtils.constructDOFInterpolationSideEvaluator("Effective Pressure", basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  // Compute basis funcitons
  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Gather coordinates
  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);

  // Map to physical frame
  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Intepolate surface height
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("Surface Height");
  fm0.template registerEvaluator<EvalT> (ev);

  // Intepolate surface height gradient
  ev = evalUtils.getPSTUtils().constructDOFGradInterpolationEvaluator("Surface Height");
  fm0.template registerEvaluator<EvalT> (ev);

  // -------------------- Special evaluators for basal side handling ----------------- //

  //---- Restrict vertex coordinates from cell-based to cell-side-based on basalside
  ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator("Coord Vec",basalSideName,"Vertex Vector",cellType,"Coord Vec " + basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Restrict ice velocity from cell-based to cell-side-based on basal side
  ev = evalUtils.constructDOFCellToSideEvaluator("Velocity",basalSideName,"Node Vector", cellType,"Basal Velocity");
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Restrict hydraulic potential from cell-based to cell-side-based on basal side
  ev = evalUtils.constructDOFCellToSideEvaluator(hydro_dof_names[0],basalSideName,"Node Scalar", cellType);
  fm0.template registerEvaluator<EvalT> (ev);

  if (has_h_equation)
  {
    //---- Restrict water thickness from cell-based to cell-side-based on basal side
    //TODO: need to write GatherSolutionSide evaluator
    ev = evalUtils.constructDOFCellToSideEvaluator(hydro_dof_names[1],basalSideName,"Node Scalar", cellType);
    fm0.template registerEvaluator<EvalT> (ev);

    // Interpolate drainage sheet depth
    ev = evalUtils.constructDOFInterpolationSideEvaluator("Water Thickness", basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    if (unsteady)
    {
      // Interpolate drainage sheet depth time derivative
      ev = evalUtils.constructDOFInterpolationSideEvaluator("Water Thickness Dot", basalSideName);
      fm0.template registerEvaluator<EvalT> (ev);
    }
  }
  else
  {
    // Interpolate water thickness (no need to restrict it to the side, cause we loaded it as a side state already)
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("Water Thickness", basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  //---- Compute side basis functions
  ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, basalSideBasis, basalCubature, basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Restrict ice velocity from cell-based to cell-side-based and interpolate on quad points
  ev = evalUtils.constructDOFVecInterpolationSideEvaluator("Basal Velocity", basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate ice thickness on QP on side
  ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("Ice Thickness", basalSideName);
  fm0.template registerEvaluator<EvalT>(ev);

  //---- Interpolate surface height on QP on side
  ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("Surface Height", basalSideName);
  fm0.template registerEvaluator<EvalT>(ev);

  //---- Interpolate hydraulic potential gradient
  ev = evalUtils.constructDOFGradInterpolationSideEvaluator("Hydraulic Potential", basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate surface water input
  ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("Surface Water Input", basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate geothermal flux
  ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("Geothermal Flux", basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Extend hydrology potential equation residual from cell-side-based to cell-based
  ev = evalUtils.constructDOFSideToCellEvaluator(hydro_resid_names[0],basalSideName,"Node Scalar", cellType);
  fm0.template registerEvaluator<EvalT> (ev);

  if (has_h_equation)
  {
    //---- Extend hydrology thickness equation residual from cell-side-based to cell-based
    ev = evalUtils.constructDOFSideToCellEvaluator(hydro_resid_names[1],basalSideName,"Node Scalar", cellType);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // -------------------- Special evaluators for surface side handling ----------------- //

  if (surfaceSideName!="INVALID")
  {
    //---- Compute side basis functions
    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, surfaceSideBasis, surfaceCubature, surfaceSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Restrict velocity (the solution) from cell-based to cell-side-based on upper side and interpolate on quad points
    ev = evalUtils.constructDOFCellToSideEvaluator("Velocity",surfaceSideName,"Node Vector", cellType,"Surface Velocity");
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate surface velocity on QP on side
    ev = evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("Observed Surface Velocity", surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface velocity rms on QP on side
    ev = evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("Observed Surface Velocity RMS", surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // -------------------------------- FELIX evaluators ------------------------- //

  // --- FO Stokes Resid --- //
  p = rcp(new ParameterList("Stokes Resid"));

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
  p->set<bool>("Needs Basal Residual", true);

  //Output
  p->set<std::string>("Residual Variable Name", stokes_resid_names[0]);

  ev = rcp(new FELIX::StokesFOResid<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- Basal Stokes Residual --- //
  p = rcp(new ParameterList("Stokes Basal Resid"));

  //Input
  p->set<std::string>("BF Side Name", "BF "+basalSideName);
  p->set<std::string>("Weighted Measure Name", "Weighted Measure "+basalSideName);
  p->set<std::string>("Basal Friction Coefficient Side QP Variable Name", "Beta");
  p->set<std::string>("Velocity Side QP Variable Name", "Basal Velocity");
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));

  //Output
  p->set<std::string>("Basal Residual Variable Name", "Basal Residual");

  ev = rcp(new FELIX::StokesFOBasalResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Sliding velocity calculation ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));

  // Input
  p->set<std::string>("Field Name","Basal Velocity");
  p->set<std::string>("Field Layout","Cell Side QuadPoint Vector");
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Field Norm"));

  // Output
  p->set<std::string>("Field Norm Name","Sliding Velocity");

  ev = Teuchos::rcp(new PHAL::Field2Norm<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Effective pressure calculation ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Effective Pressure"));

  // Input
  p->set<std::string>("Ice Thickness Variable Name","Ice Thickness");
  p->set<std::string>("Surface Height Variable Name","Surface Height");
  p->set<std::string>("Hydraulic Potential Variable Name",hydro_dof_names[0]);
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

  // Output
  p->set<std::string>("Effective Pressure Variable Name","Effective Pressure");

  ev = Teuchos::rcp(new FELIX::EffectivePressure<EvalT,PHAL::AlbanyTraits,true,false>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- FELIX basal friction coefficient ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient"));

  //Input
  p->set<std::string>("Sliding Velocity QP Variable Name", "Sliding Velocity");
  p->set<std::string>("BF Variable Name", "BF "+basalSideName);
  p->set<std::string>("Effective Pressure QP Variable Name", "Effective Pressure");
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));
  p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));

  //Output
  p->set<std::string>("Basal Friction Coefficient Variable Name", "Beta");

  ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits,true,true>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

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

  if (std::find(ss_requirements.at(basalSideName).begin(), ss_requirements.at(basalSideName).end(), "beta")!=ss_requirements.at(basalSideName).end())
  {
    // We are trying to export beta at nodes, so we need to compute it.

    //--- FELIX basal friction coefficient at nodes ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient Node"));

    //Input
    p->set<std::string>("Sliding Velocity Variable Name", "Sliding Velocity");
    p->set<std::string>("Effective Pressure Variable Name", "Effective Pressure");
    p->set<std::string>("Bed Roughness Variable Name", "Bed Roughness");
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));

    //Output
    p->set<std::string>("Basal Friction Coefficient Variable Name", "Beta");

    ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficientNode<EvalT,PHAL::AlbanyTraits,false,true>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Sliding velocity calculation ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));

    // Input
    p->set<std::string>("Field Name","Basal Velocity");
    p->set<std::string>("Field Layout","Cell Side Node Vector");
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Field Norm"));

    // Output
    p->set<std::string>("Field Norm Name","Sliding Velocity");

    ev = Teuchos::rcp(new PHAL::Field2Norm<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  //--- FELIX viscosity ---//
  p = rcp(new ParameterList("FELIX Viscosity"));

  //Input
  p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");
  p->set<std::string>("Velocity QP Variable Name", "Velocity");
  p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
  p->set<std::string>("Temperature Variable Name", "temperature");
  p->set<std::string>("Flow Factor Variable Name", "flow_factor");
  p->set<RCP<ParamLib> >("Parameter Library", paramLib);
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Viscosity"));

  //Output
  p->set<std::string>("Viscosity QP Variable Name", "FELIX Viscosity");
  p->set<std::string>("EpsilonSq QP Variable Name", "FELIX EpsilonSq");

  ev = rcp(new FELIX::ViscosityFO<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT, typename EvalT::ParamScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Shared Parameter for Continuation:  ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Homotopy Parameter"));

  param_name = "Glen's Law Homotopy Parameter";
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Homotopy>> ptr_homotopy;
  ptr_homotopy = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Homotopy>(*p,dl));
  ptr_homotopy->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Viscosity").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_homotopy);

#ifdef CISM_HAS_FELIX
  //--- FELIX surface gradient from CISM ---//
  p = rcp(new ParameterList("FELIX Surface Gradient"));

  //Input
  p->set<std::string>("CISM Surface Height Gradient X Variable Name", "CISM Surface Height Gradient X");
  p->set<std::string>("CISM Surface Height Gradient Y Variable Name", "CISM Surface Height Gradient X");
  p->set<std::string>("BF Variable Name", "BF");

  //Output
  p->set<std::string>("Surface Height Gradient QP Variable Name", "CISM Surface Height Gradient");
  ev = rcp(new FELIX::CismSurfaceGradFO<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);
#endif

  //--- Body Force ---//
  p = rcp(new ParameterList("Body Force"));

  //Input
  p->set<std::string>("FELIX Viscosity QP Variable Name", "FELIX Viscosity");
#ifdef CISM_HAS_FELIX
  p->set<std::string>("Surface Height Gradient QP Variable Name", "CISM Surface Height Gradient");
#endif
  p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");
  p->set<std::string>("Surface Height Gradient Name", "Surface Height Gradient");
  p->set<std::string>("Surface Height Name", "Surface Height");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Body Force"));
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string>("Body Force Variable Name", "Body Force");

  ev = rcp(new FELIX::StokesFOBodyForce<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Water Discharge -------- //
  p = rcp(new Teuchos::ParameterList("Hydrology Water Discharge"));

  //Input
  p->set<std::string> ("Water Thickness QP Variable Name","Water Thickness");
  p->set<std::string> ("Hydraulic Potential Gradient QP Variable Name","Hydraulic Potential Gradient");
  p->set<std::string> ("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*> ("FELIX Hydrology",&params->sublist("FELIX Hydrology"));
  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string> ("Water Discharge QP Variable Name","Water Discharge");

  if (has_h_equation)
    ev = rcp(new FELIX::HydrologyWaterDischarge<EvalT,PHAL::AlbanyTraits,true,true>(*p,dl_basal));
  else
    ev = rcp(new FELIX::HydrologyWaterDischarge<EvalT,PHAL::AlbanyTraits,false,true>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Melting Rate -------- //
  p = rcp(new Teuchos::ParameterList("Hydrology Melting Rate"));

  //Input
  p->set<std::string> ("Geothermal Heat Source QP Variable Name","Geothermal Flux");
  p->set<std::string> ("Sliding Velocity QP Variable Name","Sliding Velocity");
  p->set<std::string> ("Basal Friction Coefficient QP Variable Name","Beta");
  p->set<std::string> ("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string> ("Melting Rate QP Variable Name","Melting Rate");

  ev = rcp(new FELIX::HydrologyMeltingRate<EvalT,PHAL::AlbanyTraits,true>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);


  // ------- Hydrology Residual Potential Eqn-------- //
  p = rcp(new Teuchos::ParameterList("Hydrology Residual Potential Eqn"));

  //Input
  p->set<std::string> ("BF Name", "BF " + basalSideName);
  p->set<std::string> ("Gradient BF Name", "Grad BF " + basalSideName);
  p->set<std::string> ("Weighted Measure Name", "Weights");
  p->set<std::string> ("Water Discharge QP Variable Name", "Water Discharge");
  p->set<std::string> ("Effective Pressure QP Variable Name", "Effective Pressure");
  p->set<std::string> ("Water Thickness QP Variable Name", "Water Thickness");
  p->set<std::string> ("Melting Rate QP Variable Name","Melting Rate");
  p->set<std::string> ("Surface Water Input QP Variable Name","Surface Water Input");
  p->set<std::string> ("Sliding Velocity QP Variable Name","Sliding Velocity");
  p->set<std::string> ("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));
  p->set<Teuchos::ParameterList*> ("FELIX Hydrology Parameters",&params->sublist("FELIX Hydrology"));

  //Output
  p->set<std::string> ("Potential Eqn Residual Name",hydro_resid_names[0]);

  if (has_h_equation)
    ev = rcp(new FELIX::HydrologyResidualPotentialEqn<EvalT,PHAL::AlbanyTraits,true,true>(*p,dl_basal));
  else
    ev = rcp(new FELIX::HydrologyResidualPotentialEqn<EvalT,PHAL::AlbanyTraits,false,true>(*p,dl_basal));

  fm0.template registerEvaluator<EvalT>(ev);

  if (has_h_equation)
  {
    // ------- Hydrology Evolution Residual -------- //
    p = rcp(new Teuchos::ParameterList("Hydrology Residual Evolution"));

    //Input
    p->set<std::string> ("BF Name", "BF " + basalSideName);
    p->set<std::string> ("Weighted Measure Name", "Weighted Measure " + basalSideName);
    p->set<std::string> ("Water Thickness QP Variable Name","Water Thickness");
    p->set<std::string> ("Water Thickness Dot QP Variable Name","Water Thickness Dot");
    p->set<std::string> ("Effective Pressure QP Variable Name","Effective Pressure");
    p->set<std::string> ("Melting Rate QP Variable Name","Melting Rate");
    p->set<std::string> ("Sliding Velocity QP Variable Name","Sliding Velocity");
    p->set<std::string> ("Side Set Name", basalSideName);
    p->set<Teuchos::ParameterList*> ("FELIX Hydrology",&params->sublist("FELIX Hydrology"));
    p->set<Teuchos::ParameterList*> ("FELIX Physics",&params->sublist("FELIX Physics"));

    //Output
    p->set<std::string> ("Thickness Eqn Residual Name", hydro_resid_names[1]);

    ev = rcp(new FELIX::HydrologyResidualThicknessEqn<EvalT,PHAL::AlbanyTraits,true>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    // Require scattering of stokes residual
    PHX::Tag<typename EvalT::ScalarT> stokes_res_tag("Scatter Stokes", dl->dummy);
    fm0.requireField<EvalT>(stokes_res_tag);

    // Require scattering of hydrology residual
    PHX::Tag<typename EvalT::ScalarT> hydro_res_tag("Scatter Hydrology", dl->dummy);
    fm0.requireField<EvalT>(hydro_res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    // ----------------------- Responses --------------------- //
    RCP<ParameterList> paramList = rcp(new ParameterList("Param List"));
    RCP<const Albany::MeshSpecsStruct> meshSpecsPtr = Teuchos::rcpFromRef(meshSpecs);
    paramList->set<RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", meshSpecsPtr);
    paramList->set<RCP<ParamLib> >("Parameter Library", paramLib);
    paramList->set<std::string>("Surface Velocity Side QP Variable Name","Surface Velocity");
    paramList->set<std::string>("Observed Surface Velocity Side QP Variable Name","Observed Surface Velocity");
    paramList->set<std::string>("Observed Surface Velocity RMS Side QP Variable Name","Observed Surface Velocity RMS");
    paramList->set<std::string>("BF Surface Name","BF " + surfaceSideName);
    paramList->set<std::string>("Weighted Measure Surface Name","Weighted Measure " + surfaceSideName);
    paramList->set<std::string>("Surface Side Name", surfaceSideName);
    // The regularization with the gradient of beta is available only for GIVEN_FIELD or GIVEN_CONSTANT, which do not apply here
    paramList->set<std::string>("Basal Friction Coefficient Gradient Name","ERROR! REGULARIZATION SHOULD BE DISABLED.");
    paramList->set<std::string>("Basal Side Name","ERROR! REGULARIZATION SHOULD BE DISABLED.");
    paramList->set<std::string>("Inverse Metric Basal Name","ERROR! REGULARIZATION SHOULD BE DISABLED.");
    paramList->set<std::string>("Weighted Measure Basal Name","ERROR! REGULARIZATION SHOULD BE DISABLED.");

    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

#endif // FELIX_STOKES_FO_HYDROLOGY_PROBLEM_HPP
