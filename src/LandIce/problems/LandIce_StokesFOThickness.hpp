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
#include "PHAL_DOFCellToSide.hpp"
#include "PHAL_LoadSideSetStateField.hpp"
#include "PHAL_SaveSideSetStateField.hpp"
#include "PHAL_DOFVecInterpolationSide.hpp"

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
     const Teuchos::RCP<ParamLib>& paramLib,
     const int numDim_);

    //! Destructor
    ~StokesFOThickness();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Get boolean telling code if SDBCs are utilized
    virtual bool useSDBCs() const {return use_sdbcs_; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
      Albany::StateManager& stateMgr);

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
    buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! Each problem must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  private:

    //! Private to prohibit copying
    StokesFOThickness(const StokesFOThickness&);

    //! Private to prohibit copying
    StokesFOThickness& operator=(const StokesFOThickness&);

  public:

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
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
    Teuchos::RCP<shards::CellTopology> lateralSideType;

    Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  cellCubature;
    Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  basalCubature;
    Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  surfaceCubature;
    Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  lateralCubature;

    Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > cellBasis;
    Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > basalSideBasis;
    Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > surfaceSideBasis;
    Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > lateralSideBasis;

    int numDim, vecDimFO;
    Teuchos::RCP<Albany::Layouts> dl,dl_full,dl_basal,dl_surface,dl_lateral;

    bool  sliding;
    bool  lateral_resid;
    std::string basalSideName;
    std::string surfaceSideName;
    std::string lateralSideName;

    std::string elementBlockName;
    std::string basalEBName;
    std::string surfaceEBName;
    /// Boolean marking whether SDBCs are used
    bool use_sdbcs_;
  };

}

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
LandIce::StokesFOThickness::constructEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils_full(dl_full);

  int offset=0;

  // Temporary variables used numerous times below
  Albany::StateStruct::MeshFieldEntity entity;
  RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  RCP<Teuchos::ParameterList> p;

  // ---------------------------- Registering state variables ------------------------- //

  std::string stateName, fieldName, param_name;

  // Temperature
  entity = Albany::StateStruct::NodalDataToElemNode;
  stateName = fieldName = "temperature";
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);
  if (ss_requirements.find(basalSideName)!=ss_requirements.end())
  {
    const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);
    auto it = std::find(req.begin(), req.end(), stateName);
    if (it!=req.end())
    {
      entity = Albany::StateStruct::NodalDataToElemNode;
      // Note: temperature on side registered as node vector, since we store all layers values in the nodes on the basal mesh.
      //       However, we need to create the layout, since the vector length is the number of layers.
      //       I don't have a clean solution now, so I just ask the user to pass me the number of layers in the temperature file.
      int numLayers = params->get<int>("Layered Data Length",11); // Default 11 layers: 0, 0.1, ...,1.0
      Teuchos::RCP<PHX::DataLayout> dl_temp;
      Teuchos::RCP<PHX::DataLayout> sns = dl_basal->node_scalar;
      dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,LayerDim>(sns->dimension(0),sns->dimension(1),sns->dimension(2),numLayers));
      p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_temp, basalEBName, true, &entity);
    }
  }

  // Flow factor
  entity = Albany::StateStruct::ElemData;
  stateName = fieldName = "flow_factor";
  p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, elementBlockName, true, &entity);
  p->set<std::string>("Field Name", fieldName);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Surface height
  entity = Albany::StateStruct::NodalDataToElemNode;
  stateName = fieldName = "surface_height";
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
  p->set<std::string>("Field Name", fieldName);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);
  if (ss_requirements.find(basalSideName)!=ss_requirements.end())
  {
    // If requested, we also add it as side set state (e.g., needed if we load from file on the side mesh)
    const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);
    auto it = std::find(req.begin(), req.end(), stateName);
    if (it!=req.end())
      p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
  }

  // Ice ice_thickness
  entity = Albany::StateStruct::NodalDataToElemNode;
  stateName = fieldName = "ice_thickness";
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
  p->set<std::string>("Field Name", fieldName);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);
  if (ss_requirements.find(basalSideName)!=ss_requirements.end())
  {
    // If requested, we also add it as side set state (e.g., needed if we load from file on the side mesh)
    const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);
    auto it = std::find(req.begin(), req.end(), stateName);
    if (it!=req.end())
      p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
  }

  bool have_SMB=false;
  if(this->params->isSublist("Parameter Fields"))
  {
    Teuchos::ParameterList& params_list =  this->params->sublist("Parameter Fields");
    if(params_list.get<int>("Register Surface Mass Balance",0))
    {
      entity = Albany::StateStruct::NodalDataToElemNode;
      stateName = "surface_mass_balance";
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
      ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      have_SMB=true;
    }
  }

  // Basal friction
  stateName = fieldName = "basal_friction";
  bool isStateAParameter = false; // Determining whether basal friction is a distributed parameter
  const std::string* meshPart;
  const std::string emptyString("");
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
        if (param_list->get<std::string>("Name", emptyString) == stateName)
        {
          meshPart = &param_list->get<std::string>("Mesh Part",emptyString);
          isStateAParameter = true;
          break;
        }
      }
      else
      {
        if (stateName == dist_params_list.get(Albany::strint("Parameter", p_index), emptyString))
        {
          isStateAParameter = true;
          meshPart = &emptyString;
          break;
        }
      }
    }
  }

  if(isStateAParameter)
  {
    //basal friction is a distributed parameter
    TEUCHOS_TEST_FOR_EXCEPTION (ss_requirements.find(basalSideName)==ss_requirements.end(), std::logic_error,
                                "Error! 'basal_friction' is a parameter, but there are no basal requirements.\n");
    const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);

    TEUCHOS_TEST_FOR_EXCEPTION (std::find(req.begin(), req.end(), stateName)==req.end(), std::logic_error,
                                "Error! 'basal_friction' is a parameter, but is not listed as basal requirements.\n");

    //basal friction is a distributed 3D parameter
    entity= Albany::StateStruct::NodalDistParameter;
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, *meshPart);
    ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
    fm0.template registerEvaluator<EvalT>(ev);

    std::stringstream key;
    key << stateName <<  "Is Distributed Parameter";
    this->params->set<int>(key.str(), 1);

    if (basalSideName!="INVALID")
    {
      // Interpolate the 3D state on the side (the BasalFrictionCoefficient evaluator needs a side field)
      ev = evalUtils_full.getPSTUtils().constructDOFCellToSideEvaluator(fieldName,basalSideName,"Node Scalar",cellType);
      fm0.template registerEvaluator<EvalT> (ev);
    }
  }
  if (ss_requirements.find(basalSideName)!=ss_requirements.end())
  {
    const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);
    if (std::find(req.begin(), req.end(), stateName)!=req.end())
    {
      // ...and basal_friction is one of them.
      entity = Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
      if (isStateAParameter)
      {
        //basal friction is a distributed 3D parameter. We already took care of this case
      }
      else if (std::find(requirements.begin(),requirements.end(),stateName)==requirements.end()) //otherwise see below
      {
        ev = rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }
    else if (isStateAParameter)
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! basal_friction is a parameter, but is not listed as a basal requirement.\n");
    }
  }
  else if (isStateAParameter)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! basal_friction is a parameter, but there are no basal requirements.\n");
  }

  if (!isStateAParameter && std::find(requirements.begin(),requirements.end(),"basal_friction")!=requirements.end())
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
    p->set<std::string>("Field Name", fieldName);

    // We are (for some mystic reason) extruding beta to the whole 3D mesh
    ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // We restrict it back to the 2D mesh. Clearly, this is not optimal. Just add 'basal_friction' to the Basal Requirements!
    if(basalSideName!="INVALID") {
      ev = evalUtils_full.constructDOFCellToSideEvaluator("Beta",basalSideName,"Node Scalar",cellType);
      fm0.template registerEvaluator<EvalT> (ev);
    }
  }

  if (ss_requirements.find(basalSideName)!=ss_requirements.end())
  {
    stateName = fieldName = "effective_pressure";
    const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);

    auto it = std::find(req.begin(), req.end(), stateName);
    if (it!=req.end())
    {
      // We interpolate the effective pressure from quad point to cell
      ev = evalUtils_full.constructSideQuadPointsToSideInterpolationEvaluator (fieldName, basalSideName, false);
      fm0.template registerEvaluator<EvalT>(ev);

      // We register the state and build the loader
      p = stateMgr.registerSideSetStateVariable(basalSideName,stateName,fieldName, dl_basal->cell_scalar2, basalEBName, true);
      p->set<bool>("Is Vector Field", false);
      ev = rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
      fm0.template registerEvaluator<EvalT>(ev);
      if (fieldManagerChoice == Albany::BUILD_RESID_FM)
      {
        // Only PHAL::AlbanyTraits::Residual evaluates something
        if (ev->evaluatedFields().size()>0)
        {
          // Require save beta
          fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
        }
      }
    }
  }

  // Bed topography
  stateName = "bed_topography";
  entity= Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

#ifdef MPAS_HAS_LANDICE
  // Dirichelt field
  entity = Albany::StateStruct::NodalDistParameter;
  // Here is how to register the field for dirichlet condition.
  stateName = "dirichlet_field";
  p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, true, &entity, "");
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);
#endif

  // Define Field Names
  Teuchos::ArrayRCP<std::string> dof_names(1);
  Teuchos::ArrayRCP<std::string> resid_names(1);
  dof_names[0] = "Velocity";
  resid_names[0] = "Stokes Residual";

  // ------------------- Interpolations and utilities ------------------ //

  //---- Gather solution field (whole)
  ev = evalUtils_full.constructGatherSolutionEvaluator_noTransient(true, dof_names, offset);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate solution field
  ev = evalUtils_full.constructDOFVecInterpolationEvaluator(dof_names[0], offset);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate solution field gradient
  ev = evalUtils_full.constructDOFVecGradInterpolationEvaluator(dof_names[0], offset);
  fm0.template registerEvaluator<EvalT> (ev);

  dof_names[0] = "Velocity Reduced";

  //---- Gather solution field (velocity only)
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names, offset);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate solution field
  ev = evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0], offset);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate solution field gradient
  ev = evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0], offset);
  fm0.template registerEvaluator<EvalT> (ev);

  dof_names[0] = "U";

  //---- Gather solution field (velocity as part of whole solution)
  ev = evalUtils_full.constructGatherSolutionEvaluator_noTransient(true, dof_names, 0);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate solution field gradient
  ev = evalUtils_full.constructDOFVecGradInterpolationEvaluator(dof_names[0], 0);
  fm0.template registerEvaluator<EvalT> (ev);

#ifndef ALBANY_MESH_DEPENDS_ON_SOLUTION
  //---- Gather coordinates
  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);
#endif

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

  if (basalSideName!="INVALID")
  {
    // -------------------- Special evaluators for side handling ----------------- //

    //---- Restrict coordinate vector from cell-based to cell-side-based
    ev = evalUtils_full.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,basalSideName,"Vertex Vector",cellType,"Coord Vec " + basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute side basis functions
    ev = evalUtils_full.constructComputeBasisFunctionsSideEvaluator(cellType, basalSideBasis, basalCubature, basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Restrict velocity from cell-based to cell-side-based
    ev = evalUtils_full.constructDOFCellToSideEvaluator("Velocity",basalSideName,"Node Vector",cellType,"basal_velocity");
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Restrict ice thickness from cell-based to cell-side-based
    ev = evalUtils_full.constructDOFCellToSideEvaluator("ice_thickness",basalSideName,"Node Scalar",cellType);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Restrict surface height from cell-based to cell-side-based
    ev = evalUtils_full.constructDOFCellToSideEvaluator("surface_height",basalSideName,"Node Scalar",cellType);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity on QP on side
    ev = evalUtils_full.constructDOFVecInterpolationSideEvaluator("basal_velocity", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate thickness on QP on side
    ev = evalUtils_full.constructDOFInterpolationSideEvaluator("ice_thickness", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate basal_friction on QP on side
    ev = evalUtils_full.getPSTUtils().constructDOFInterpolationSideEvaluator("basal_friction", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface height on QP on side
    ev = evalUtils_full.constructDOFInterpolationSideEvaluator("surface_height", basalSideName);
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (surfaceSideName!="INVALID")
  {
    //---- Restrict coordinate vector from cell-based to cell-side-based
    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,surfaceSideName,"Vertex Vector",cellType,"Coord Vec " + surfaceSideName);
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
    ev = evalUtils.constructDOFCellToSideEvaluator("Velocity Reduced",surfaceSideName,"Node Vector",cellType,"surface_velocity");
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity (the solution) on QP on side
    ev = evalUtils.constructDOFVecInterpolationSideEvaluator("surface_velocity", surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (lateralSideName!="INVALID")
  {
    //---- Restrict vertex coordinates from cell-based to cell-side-based
    ev = evalUtils_full.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,lateralSideName,"Vertex Vector",cellType,Albany::coord_vec_name + " " + lateralSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute Quad Points coordinates on the side set
    ev = evalUtils_full.constructMapToPhysicalFrameSideEvaluator(cellType,lateralCubature,lateralSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute side basis functions
    ev = evalUtils_full.constructComputeBasisFunctionsSideEvaluator(cellType, lateralSideBasis, lateralCubature, lateralSideName, false, true);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate ice thickness on QP on side
    ev = evalUtils_full.getPSTUtils().constructDOFCellToSideQPEvaluator("ice_thickness", lateralSideName, "Node Scalar", cellType,"ice_thickness_"+lateralSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface height on QP on side
    ev = evalUtils_full.getPSTUtils().constructDOFCellToSideQPEvaluator("surface_height", lateralSideName, "Node Scalar", cellType,"surface_height_"+lateralSideName);
    fm0.template registerEvaluator<EvalT>(ev);
  }


  // -------------------------------- LandIce/PHAL evaluators ------------------------- //

  //--- LandIce Stokes FO Residual With Extruded Field ---//
  p = rcp(new ParameterList("Scatter StokesFO"));

  //Input
  resid_names[0] = "StokesFO Residual";
  p->set< Teuchos::ArrayRCP<std::string> >("Residual Names", resid_names);
  p->set<int>("Tensor Rank", 1);
  p->set<int>("Offset of First DOF", 0);
  p->set<int>("Offset 2D Field", 2);

  //Output
  p->set<std::string>("Scatter Field Name", "Scatter StokesFO");

  ev = rcp(new PHAL::ScatterResidualWithExtrudedField<EvalT,PHAL::AlbanyTraits>(*p,dl_full));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- LandIce Stokes FO Residual Implicit Thickness Update ---//
  p = rcp(new ParameterList("Scatter StokesFOImplicitThicknessUpdate"));

  //Input
  resid_names[0] = "StokesFOImplicitThicknessUpdate Residual";
  p->set< Teuchos::ArrayRCP<std::string> >("Residual Names", resid_names);
  p->set<int>("Tensor Rank", 1);
  p->set<int>("Offset of First DOF", 0);
  p->set<int>("Offset 2D Field", 2);

  //Output
  p->set<std::string>("Scatter Field Name", "Scatter StokesFOImplicitThicknessUpdate");

  ev = rcp(new PHAL::ScatterResidualWithExtrudedField<EvalT,PHAL::AlbanyTraits>(*p,dl_full));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- LandIce Stokes FO Residual Thickness ---//
  p = rcp(new ParameterList("Scatter ResidualH"));

  //Input
  offset = vecDimFO;
  resid_names[0] = "Thickness Residual";
  p->set< Teuchos::ArrayRCP<std::string> >("Residual Names", resid_names);
  p->set<int>("Tensor Rank", 0);
  p->set<int>("Offset of First DOF", offset);
  p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",rcp(new CellTopologyData(meshSpecs.ctd)));

  //Output
  p->set<std::string>("Scatter Field Name", "Scatter Thickness");

  ev = rcp(new PHAL::ScatterResidual2D<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- Thickness Resid --- //
  p = rcp(new ParameterList("Thickness Resid"));

  //Input
  p->set<std::string>("Averaged Velocity Variable Name", "Averaged Velocity");
  p->set<std::string>("Thickness Increment Variable Name", "Thickness2D");
  p->set<std::string>("Past Thickness Name", "ice_thickness");
  p->set<std::string>("Coordinate Vector Name", Albany::coord_vec_name);
  p->set<int>("Cubature Degree",3);
  if(have_SMB)
    p->set<std::string>("SMB Name", "surface_mass_balance");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Equation Set"));
  p->set<RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", Teuchos::rcpFromRef(meshSpecs));
  if(this->params->isParameter("Time Step Ptr"))
    p->set<RCP<double> >("Time Step Ptr", this->params->get<Teuchos::RCP<double> >("Time Step Ptr"));
  else {
    RCP<double> dt = rcp(new double(this->params->get<double>("Time Step")));
    p->set<RCP<double> >("Time Step Ptr", dt);
  }

  //Output
  p->set<std::string>("Residual Name", "Thickness Residual");

  ev = rcp(new LandIce::ThicknessResid<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- LandIce Gather Extruded 2D Field (Thickness) ---//
  p = rcp(new ParameterList("Gather ExtrudedThickness"));

  //Input
  offset = vecDimFO;
  p->set<std::string>("2D Field Name", "ExtrudedThickness");
  p->set<int>("Offset of First DOF", offset);
  p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",rcp(new CellTopologyData(meshSpecs.ctd)));

  ev = rcp(new LandIce::GatherExtruded2DField<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- LandIce Gather 2D Field (Thickness) ---//
  p = rcp(new ParameterList("Gather Thickness"));

  //Input
  offset = vecDimFO;
  p->set<std::string>("2D Field Name", "Thickness2D");
  p->set<int>("Offset of First DOF", offset);
  p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",rcp(new CellTopologyData(meshSpecs.ctd)));

  ev = rcp(new LandIce::Gather2DField<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- LandIce Gather Vertically Averaged Velocity ---//
  p = rcp(new ParameterList("Gather Averaged Velocity"));

  //Input
  p->set<std::string>("Averaged Velocity Name", "Averaged Velocity");
  p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",rcp(new CellTopologyData(meshSpecs.ctd)));

  ev = rcp(new LandIce::GatherVerticallyAveragedVelocity<EvalT,PHAL::AlbanyTraits>(*p,dl));
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

  if (sliding)
  {
    // --- Basal Residual --- //
    p = rcp(new ParameterList("Stokes Basal Residual"));

    //Input
    p->set<std::string>("BF Side Name", Albany::bf_name + " "+basalSideName);
    p->set<std::string>("Weighted Measure Name", Albany::weighted_measure_name + " "+basalSideName);
    p->set<std::string>("Basal Friction Coefficient Side QP Variable Name", "beta");
    p->set<std::string>("Velocity Side QP Variable Name", "basal_velocity");
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Basal Friction Coefficient"));
    p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

    //Output
    p->set<std::string>("Basal Residual Variable Name", "Basal Residual");

    ev = rcp(new LandIce::StokesFOBasalResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl_full));
    fm0.template registerEvaluator<EvalT>(ev);

    // --- Prolongate Stokes FO Basal Residual --- //
    p = rcp(new ParameterList("Prolongate Stokes FO Basal Resid"));

    //Input
    p->set<std::string>("Field Name", "Basal Residual");
    p->set<std::string>("Field Layout", "Cell Node Vector");
    p->set<bool>("Pad Back", true);
    p->set<double>("Padding Value", 0.);

    ev = rcp(new LandIce::ProlongateVector<EvalT,PHAL::AlbanyTraits>(*p, dl, dl_full));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Sliding velocity calculation ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Velocity Norm"));

    // Input
    p->set<std::string>("Field Name","basal_velocity");
    p->set<std::string>("Field Layout","Cell Side QuadPoint Vector");
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Field Norm"));

    // Output
    p->set<std::string>("Field Norm Name","sliding_velocity");

    ev = Teuchos::rcp(new PHAL::FieldFrobeniusNorm<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Ice Overburden (QPs) ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Effective Pressure Surrogate"));

    // Input
    p->set<bool>("Nodal",false);
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<std::string>("Ice Thickness Variable Name", "ice_thickness");
    p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

    // Output
    p->set<std::string>("Ice Overburden Variable Name", "ice_overburden");

    ev = Teuchos::rcp(new LandIce::IceOverburden<EvalT,PHAL::AlbanyTraits,true>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Ice Overburden (Nodes) ---//
    p->set<bool>("Nodal",true);
    ev = Teuchos::rcp(new LandIce::IceOverburden<EvalT,PHAL::AlbanyTraits,true>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Effective pressure surrogate (QPs) ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Effective Pressure Surrogate"));

    // Input
    p->set<bool>("Nodal",false);
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<std::string>("Ice Overburden Variable Name", "ice_overburden");

    // Output
    p->set<std::string>("Effective Pressure Variable Name","effective_pressure");

    ev = Teuchos::rcp(new LandIce::EffectivePressure<EvalT,PHAL::AlbanyTraits,true,true>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Effective pressure surrogate (Nodes) ---//
    p->set<bool>("Nodal",true);
    ev = Teuchos::rcp(new LandIce::EffectivePressure<EvalT,PHAL::AlbanyTraits,true,true>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Shared Parameter for basal friction coefficient: alpha ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: alpha"));

    param_name = "Hydraulic-Over-Hydrostatic Potential Ratio";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Alpha>> ptr_alpha;
    ptr_alpha = Teuchos::rcp(new LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Alpha>(*p,dl));
    ptr_alpha->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_alpha);

    //--- Shared Parameter for basal friction coefficient: lambda ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: lambda"));

    param_name = "Bed Roughness";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Lambda>> ptr_lambda;
    ptr_lambda = Teuchos::rcp(new LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Lambda>(*p,dl));
    ptr_lambda->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_lambda);

    //--- Shared Parameter for basal friction coefficient: mu ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: mu"));

    param_name = "Coulomb Friction Coefficient";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Mu>> ptr_mu;
    ptr_mu = Teuchos::rcp(new LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Mu>(*p,dl));
    ptr_mu->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_mu);

    //--- Shared Parameter for basal friction coefficient: power ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: power"));

    param_name = "Power Exponent";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Power>> ptr_power;
    ptr_power = Teuchos::rcp(new LandIce::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Power>(*p,dl));
    ptr_power->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_power);

    //--- LandIce basal friction coefficient ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Basal Friction Coefficient"));

    //Input
    p->set<std::string>("Sliding Velocity Variable Name", "sliding_velocity");
    p->set<std::string>("BF Variable Name", Albany::bf_name + " " + basalSideName);
    p->set<std::string>("Effective Pressure Variable Name", "effective_pressure");
    p->set<std::string>("Bed Roughness Variable Name", "bed_roughness");
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec " + basalSideName);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Basal Friction Coefficient"));
    p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));
    p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
    params->sublist("LandIce Basal Friction Coefficient").set<std::string>("Beta Given Variable Name", "basal_friction");
    p->set<std::string>("Bed Topography Variable Name", "bed_topography");
    p->set<std::string>("Ice Thickness Variable Name", "ice_thickness");

    //Output
    p->set<std::string>("Basal Friction Coefficient Variable Name", "beta");

    ev = Teuchos::rcp(new LandIce::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits,false,true,false>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);
  }

#ifdef ALBANY_MESH_DEPENDS_ON_SOLUTION
  {
    //--- Gather Coordinates ---//
    p = rcp(new ParameterList("Gather Coordinate Vector"));

    // Input: Periodic BC flag
    p->set<bool>("Periodic BC", false);

    // Output:: Coordindate Vector at vertices
    p->set<std::string>("Coordinate Vector Name", "Coord Vec Old");

    ev =  rcp(new PHAL::GatherCoordinateVector<EvalT,PHAL::AlbanyTraits>(*p,dl_full));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  {
    //--- Update Z Coordinate ---//
    p = rcp(new ParameterList("Update Z Coordinate"));

    // Input
    p->set<std::string>("Old Coords Name", "Coord Vec Old");
    p->set<std::string>("New Coords Name", Albany::coord_vec_name);
    p->set<std::string>("Thickness Increment Name", "ExtrudedThickness");
    p->set<std::string>("Past Thickness Name", "ice_thickness");
    p->set<std::string>("Top Surface Name", "surface_height");

    ev = rcp(new LandIce::UpdateZCoordinateMovingTop<EvalT,PHAL::AlbanyTraits>(*p, dl_full));
    fm0.template registerEvaluator<EvalT>(ev);
  }
#endif

  // --- FO Stokes Resid --- //
  p = rcp(new ParameterList("StokesFO Resid"));

  //Input
  p->set<std::string>("Weighted BF Variable Name", Albany::weighted_bf_name);
  p->set<std::string>("Weighted Gradient BF Variable Name", Albany::weighted_grad_bf_name);
  p->set<std::string>("Velocity QP Variable Name", "Velocity");
  p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
  p->set<std::string>("Body Force Variable Name", "Body Force");
  p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
  p->set<std::string>("Coordinate Vector Name", Albany::coord_vec_name);
  p->set<std::string>("Basal Residual Variable Name", "Basal Residual");
  p->set<std::string>("Lateral Residual Variable Name", "Lateral Residual");
  p->set<bool>("Needs Basal Residual", sliding);
  p->set<bool>("Needs Lateral Residual", lateral_resid);
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Equation Set"));

  //Output
  p->set<std::string>("Residual Variable Name", "StokesFO Residual");

  ev = rcp(new LandIce::StokesFOResid<EvalT,PHAL::AlbanyTraits>(*p, dl_full));
  fm0.template registerEvaluator<EvalT>(ev);

  if (lateral_resid) {
    p = Teuchos::rcp( new Teuchos::ParameterList("Lateral Residual") );

    // Input
    p->set<std::string>("Ice Thickness Variable Name", "ice_thickness_"+lateralSideName);
    p->set<std::string>("Ice Surface Elevation Variable Name", "surface_height_"+lateralSideName);
    p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name + " " + lateralSideName);
    p->set<std::string>("BF Side Name", Albany::bf_name + " " + lateralSideName);
    p->set<std::string>("Weighted Measure Name", Albany::weighted_measure_name + " " + lateralSideName);
    p->set<std::string>("Side Normal Name", Albany::normal_name + " " + lateralSideName);
    p->set<std::string>("Side Set Name", lateralSideName);
    p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
    p->set<Teuchos::ParameterList*>("Lateral BC Parameters",&params->sublist("LandIce Lateral BC"));
    p->set<Teuchos::ParameterList*>("Physical Parameters",&params->sublist("LandIce Physical Parameters"));
    p->set<Teuchos::ParameterList*>("Stereographic Map",&params->sublist("Stereographic Map"));

    // Output
    p->set<std::string>("Lateral Residual Variable Name", "Lateral Residual");

    ev = Teuchos::rcp( new LandIce::StokesFOLateralResid<EvalT,PHAL::AlbanyTraits,false>(*p,dl_full) );
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // --- FO Stokes Implicit Thickness Update Resid --- //
  p = rcp(new ParameterList("StokesFOImplicitThicknessUpdate Resid"));

  //Input
  p->set<std::string>("Input Residual Name", "StokesFO Residual");
  p->set<std::string>("Thickness Increment Variable Name", "ExtrudedThickness");
  p->set<std::string>("Gradient BF Name", Albany::grad_bf_name);
  p->set<std::string>("Weighted BF Name", Albany::weighted_bf_name);

  Teuchos::ParameterList& physParamList = params->sublist("LandIce Physical Parameters");
  p->set<Teuchos::ParameterList*>("Physical Parameter List", &physParamList);

  //Output
  p->set<std::string>("Residual Name", "StokesFOImplicitThicknessUpdate Residual");

  ev = rcp(new LandIce::StokesFOImplicitThicknessUpdateResid<EvalT,PHAL::AlbanyTraits>(*p, dl_full, dl_full));
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

  //--- LandIce viscosity ---//
  p = rcp(new ParameterList("LandIce Viscosity"));

  //Input
  p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name);
  p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
  p->set<std::string>("Velocity QP Variable Name", "Velocity");
  p->set<std::string>("Temperature Variable Name", "temperature");
  p->set<std::string>("Flow Factor Variable Name", "flow_factor");
  p->set<RCP<ParamLib> >("Parameter Library", paramLib);
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Viscosity"));
  p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

  //Output
  p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
  p->set<std::string>("EpsilonSq QP Variable Name", "LandIce EpsilonSq");

  ev = rcp(new LandIce::ViscosityFO<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ParamScalarT>(*p,dl_full));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Body Force ---//
  p = rcp(new ParameterList("Body Force"));

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

  ev = rcp(new LandIce::StokesFOBodyForce<EvalT,PHAL::AlbanyTraits>(*p,dl_full));
  fm0.template registerEvaluator<EvalT>(ev);

  if (surfaceSideName!="INVALID")
  {
    // Load surface velocity
    entity= Albany::StateStruct::NodalDataToElemNode;
    stateName = fieldName = "observed_surface_velocity";
    p = stateMgr.registerSideSetStateVariable(surfaceSideName, stateName, fieldName, dl_surface->node_vector, surfaceEBName, true, &entity);
    ev = rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Load surface velocity rms
    entity= Albany::StateStruct::NodalDataToElemNode;
    stateName = fieldName = "observed_surface_velocity_RMS";
    p = stateMgr.registerSideSetStateVariable(surfaceSideName, stateName, fieldName, dl_surface->node_vector, surfaceEBName, true, &entity);
    ev = rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Build beta gradient on QP on basal side
    p = rcp(new ParameterList("Basal Friction Coefficient Gradient"));

    // Input
    p->set<std::string>("Beta Given Variable Name", "basal_friction");
    p->set<std::string>("Gradient BF Side Variable Name", Albany::grad_bf_name + " "+basalSideName);
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<std::string>("Effective Pressure QP Name", "effective_pressure");
    p->set<std::string>("Effective Pressure Gradient QP Name", "effective_pressure Gradient");
    p->set<std::string>("Basal Velocity QP Name", "basal_velocity");
    p->set<std::string>("Basal Velocity Gradient QP Name", "basal_velocity Gradient");
    p->set<std::string>("Sliding Velocity QP Name", "sliding_velocity");
    p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec "+basalSideName);
    p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Basal Friction Coefficient"));

    // Output
    p->set<std::string>("Basal Friction Coefficient Gradient Name","beta Gradient");

    ev = rcp(new LandIce::BasalFrictionCoefficientGradient<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> resProH_tag("Scatter StokesFOImplicitThicknessUpdate", dl_full->dummy);
    fm0.requireField<EvalT>(resProH_tag);
    PHX::Tag<typename EvalT::ScalarT> resThick_tag("Scatter Thickness", dl->dummy);
    fm0.requireField<EvalT>(resThick_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    // ----------------------- Responses --------------------- //
    RCP<ParameterList> paramList = rcp(new ParameterList("Param List"));
    paramList->set<RCP<ParamLib> >("Parameter Library", paramLib);
    paramList->set<std::string>("Basal Friction Coefficient Gradient Name","beta Gradient");
    paramList->set<std::string>("Surface Velocity Side QP Variable Name","surface_velocity");
    paramList->set<std::string>("Observed Surface Velocity Side QP Variable Name","observed_surface_velocity");
    paramList->set<std::string>("Observed Surface Velocity RMS Side QP Variable Name","observed_surface_velocity_RMS");
    paramList->set<std::string>("BF Surface Name",Albany::bf_name + " " + surfaceSideName);
    paramList->set<std::string>("Weighted Measure Basal Name",Albany::weighted_measure_name + " " + basalSideName);
    paramList->set<std::string>("Weighted Measure Surface Name",Albany::weighted_measure_name + " " + surfaceSideName);
    paramList->set<std::string>("Inverse Metric Basal Name",Albany::metric_inv_name + " " + basalSideName);
    paramList->set<std::string>("Metric Basal Name",Albany::metric_name + " " + basalSideName);
    paramList->set<std::string>("Metric Surface Name",Albany::metric_name + " " + surfaceSideName);
    paramList->set<std::string>("Basal Side Name", basalSideName);
    paramList->set<std::string>("Surface Side Name", surfaceSideName);
    paramList->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",rcp(new CellTopologyData(meshSpecs.ctd)));

    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

#endif // LANDICE_STOKES_FO_THICKNESS_PROBLEM_HPP
