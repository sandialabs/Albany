//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKES_FO_HYDROLOGY_PROBLEM_HPP
#define FELIX_STOKES_FO_HYDROLOGY_PROBLEM_HPP 1

#include <type_traits>

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
#include "FELIX_FieldNorm.hpp"
#include "FELIX_BasalFrictionCoefficient.hpp"
#include "FELIX_BasalFrictionCoefficientGradient.hpp"
#include "FELIX_HydrologyWaterDischarge.hpp"
#include "FELIX_HydrologyResidualEllipticEqn.hpp"
#include "FELIX_HydrologyResidualEvolutionEqn.hpp"
#include "FELIX_HydrologyMeltingRate.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX
{

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
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

  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  cellCubature;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  basalCubature;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  surfaceCubature;

  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > cellBasis;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > basalSideBasis;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > surfaceSideBasis;

  int numDim;
  int vecDimStokesFO;
  bool has_evolution_equation;
  Teuchos::RCP<Albany::Layouts> dl,dl_basal,dl_surface;

  std::string basalSideName;
  std::string surfaceSideName;

  std::string elementBlockName;
  std::string basalEBName;
  std::string surfaceEBName;

  std::vector<std::string> basalReq;
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

  std::string stateName, fieldName;

  // Temperature
  entity = Albany::StateStruct::NodalDataToElemNode;
  stateName = fieldName = "temperature";
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Note: temperature on side registered as node vector, since we store all layers values in the nodes on the basal mesh.
  //       However, we need to create the layout, since the vector length is the value of the number of layers.
  //       I don't have a clean solution now, so I just ask the user to pass me the number of layers in the temperature file.
  int numLayers = params->get<int>("Layered Data Length",11); // Default 11 layers: 0, 0.1, ...,1.0
  Teuchos::RCP<PHX::DataLayout> dl_temp;
  Teuchos::RCP<PHX::DataLayout> sns = dl_basal->node_scalar;
  dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,VecDim>(sns->dimension(0),sns->dimension(1),sns->dimension(2),numLayers));
  p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_temp, basalEBName, true, &entity);

  // Flow factor
  entity = Albany::StateStruct::ElemData;
  stateName = fieldName = "flow_factor";
  p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, elementBlockName, true, &entity);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Surface height
  entity = Albany::StateStruct::NodalDataToElemNode;
  stateName = "surface_height";
  fieldName = "Surface Height";
  p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
  p->set<std::string>("Field Name", fieldName);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

#ifdef CISM_HAS_FELIX
  // Surface Gradient-x
  entity = Albany::StateStruct::NodalDataToElemNode;
  stateName = "xgrad_surface_height"; //ds/dx which can be passed from CISM (definened at nodes)
  fieldName = "CISM Surface Height Gradient X";
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
  p->set<std::string>("Field Name", fieldName);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Surface Gradient-y
  entity = Albany::StateStruct::NodalDataToElemNode;
  stateName = "ygrad_surface_height"; //ds/dy which can be passed from CISM (defined at nodes)
  fieldName = "CISM Surface Height Gradient Y";
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
  p->set<std::string>("Field Name", fieldName);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);
#endif

  // Ice thickness
  entity = Albany::StateStruct::NodalDataToElemNode;
  stateName = "thickness";
  fieldName = "Ice Thickness";
  p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
  p->set<std::string>("Field Name", fieldName);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Check if the user also wants to save a side-averaged beta
  stateName = "beta_side_avg";
  fieldName = "Beta";
  if (std::find(basalReq.begin(), basalReq.end(), stateName) != basalReq.end())
  {
    // We interpolate beta from quad point to cell
    ev = evalUtils.constructSideQuadPointsToSideInterpolationEvaluator (fieldName, basalSideName, false);
    fm0.template registerEvaluator<EvalT>(ev);

    // We save it on the basal mesh
    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->cell_scalar2, basalEBName, true);
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

  // Check if the user also wants to save the effective pressure
  stateName = "effective_pressure";
  fieldName = "Effective Pressure";
  if (std::find(basalReq.begin(), basalReq.end(), stateName)!=basalReq.end());
  {
    // We interpolate the effective pressure from quad point to cell
    ev = evalUtils.constructSideQuadPointsToSideInterpolationEvaluator (fieldName, basalSideName, false);
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

  // Bed topography
  stateName = "bed_topography";
  entity= Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Hydrology surface water input
  stateName = "surface_water_input";
  fieldName = "Surface Water Input";
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerSideSetStateVariable(basalSideName,stateName,fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
  p->set<const std::string>("Field Name",fieldName);
  ev = rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Hydrology geothermal flux
  stateName = "geothermal flux";
  fieldName = "Geothermal Flux";
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerSideSetStateVariable(basalSideName,stateName,fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
  p->set<const std::string>("Field Name",fieldName);
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  if (!has_evolution_equation)
  {
    // Drainage sheet depth
    stateName = "drainage_sheet_depth";
    fieldName = "Drainage Sheet Depth";
    entity = Albany::StateStruct::NodalDataToElemNode;
    p = stateMgr.registerSideSetStateVariable(basalSideName,stateName,fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
    p->set<const std::string>("Field Name",fieldName);
    ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
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
  int offsetHydro  = vecDimStokesFO;

  // Gather stokes solution field
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(true, stokes_dof_names, offsetStokes);
  fm0.template registerEvaluator<EvalT> (ev);

  // Gather hydrology solution field
  if (has_evolution_equation)
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

  if (has_evolution_equation)
  {
    // Interpolate drainage sheet depth
    ev = evalUtils.constructDOFInterpolationSideEvaluator("Drainage Sheet Depth", basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    // Interpolate drainage sheet depth time derivative
    ev = evalUtils.constructDOFInterpolationSideEvaluator("Drainage Sheet Depth Dot", basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);
  }
  else
  {
    // Interpolate drainage sheet depth
    ev = evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("Drainage Sheet Depth", basalSideName);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // Compute basis funcitons
  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Gather coordinates
  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);

  // Map to physical frame
  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate temperature from nodes to cell
  ev = evalUtils.getPSTUtils().constructNodesToCellInterpolationEvaluator ("temperature",false);
  fm0.template registerEvaluator<EvalT> (ev);

  // Intepolate surface height
  ev = evalUtils.constructDOFInterpolationEvaluator_noDeriv("Surface Height");
  fm0.template registerEvaluator<EvalT> (ev);

  // Intepolate surface height gradient
  ev = evalUtils.constructDOFGradInterpolationEvaluator_noDeriv("Surface Height");
  fm0.template registerEvaluator<EvalT> (ev);

  // -------------------- Special evaluators for basal side handling ----------------- //

  //---- Compute side basis functions
  ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, basalSideBasis, basalCubature, basalSideName);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Restrict ice velocity from cell-based to cell-side-based
  ev = evalUtils.constructDOFCellToSideEvaluator("Velocity",basalSideName,"Node Vector",cellType,"Basal Velocity");
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Restrict ice thickness from cell-based to cell-side-based
  ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator ("Ice Thickness",basalSideName,"Node Scalar",cellType);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Restrict surface height from cell-based to cell-side-based
  ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator ("Surface Height",basalSideName,"Node Scalar",cellType);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate velocity on QP on side
  ev = evalUtils.constructDOFVecInterpolationSideEvaluator("Basal Velocity", basalSideName);
  fm0.template registerEvaluator<EvalT>(ev);

  //---- Interpolate thickness on QP on side
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

  // -------------------- Special evaluators for surface side handling ----------------- //

  if (surfaceSideName!="INVALID")
  {
    //---- Compute side basis functions
    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, surfaceSideBasis, surfaceCubature, surfaceSideName);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate surface velocity on QP on side
    ev = evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("Observed Surface Velocity", surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate surface velocity rms on QP on side
    ev = evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("Observed Surface Velocity RMS", surfaceSideName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Restrict velocity (the solution) from cell-based to cell-side-based on upper side
    ev = evalUtils.constructDOFCellToSideEvaluator("Velocity",surfaceSideName,"Node Vector", cellType,"Surface Velocity");
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity (the solution) on QP on side
    ev = evalUtils.constructDOFVecInterpolationSideEvaluator("Surface Velocity", surfaceSideName);
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
  p->set<std::string>("Residual Variable Name", "Stokes Residual");

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

  ev = rcp(new FELIX::StokesFOBasalResid<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Sliding velocity calculation ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));

  // Input
  p->set<std::string>("Field Name","Basal Velocity");
  p->set<std::string>("Field Layout","Cell Side QuadPoint");
  p->set<std::string>("Side Set Name", basalSideName);

  // Output
  p->set<std::string>("Field Norm Name","Sliding Velocity");

  ev = Teuchos::rcp(new FELIX::FieldNorm<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Effective pressure calculation ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Effective Pressure Surrogate"));

  // Input
  p->set<std::string>("Hydrostatic Potential Variable Name","Subglacial Hydrostatic Potential");
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<bool>("Surrogate", false);
  p->set<bool>("Stokes", true);

  // Output
  p->set<std::string>("Effective Pressure Variable Name","Effective Pressure");

  ev = Teuchos::rcp(new FELIX::EffectivePressure<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- FELIX basal friction coefficient ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient"));

  //Input
  p->set<std::string>("Sliding Velocity Side QP Variable Name", "Sliding Velocity");
  p->set<std::string>("BF Variable Name", "BF "+basalSideName);
  p->set<std::string>("Effective Pressure QP Variable Name", "Effective Pressure");
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));
  p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string>("Basal Friction Coefficient Variable Name", "Beta");

  ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits>(*p,dl_basal));
  fm0.template registerEvaluator<EvalT>(ev);

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

  ev = rcp(new FELIX::ViscosityFO<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);


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
  p->set<std::string> ("Drainage Sheet Depth QP Variable Name","Drainage Sheet Depth");
  p->set<std::string> ("Hydraulic Potential Gradient QP Variable Name","Hydraulic Potential Gradient");
  p->set<std::string> ("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*> ("FELIX Hydrology",&params->sublist("FELIX Hydrology"));
  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string> ("Water Discharge QP Variable Name","Water Discharge");

  ev = rcp(new FELIX::HydrologyWaterDischarge<EvalT,PHAL::AlbanyTraits>(*p,dl));
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

  ev = rcp(new FELIX::HydrologyMeltingRate<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);


  // ------- Hydrology Residual Elliptic Eqn-------- //
  p = rcp(new Teuchos::ParameterList("Hydrology Residual Elliptic Eqn"));

  //Input
  p->set<std::string> ("BF Name", "BF " + basalSideName);
  p->set<std::string> ("Gradient BF Name", "Grad BF " + basalSideName);
  p->set<std::string> ("Weighted Measure Name", "Weights");
  p->set<std::string> ("Water Discharge QP Variable Name", "Water Discharge");
  p->set<std::string> ("Effective Pressure QP Variable Name", "Effective Pressure");
  p->set<std::string> ("Drainage Sheet Depth QP Variable Name", "Drainage Sheet Depth");
  p->set<std::string> ("Melting Rate QP Variable Name","Melting Rate");
  p->set<std::string> ("Surface Water Input QP Variable Name","Surface Water Input");
  p->set<std::string> ("Sliding Velocity QP Variable Name","Sliding Velocity");
  p->set<std::string> ("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));
  p->set<Teuchos::ParameterList*> ("FELIX Hydrology Parameters",&params->sublist("FELIX Hydrology"));

  //Output
  p->set<std::string> ("Hydrology Elliptic Eqn Residual Name",hydro_resid_names[0]);

  ev = rcp(new FELIX::HydrologyResidualEllipticEqn<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if (has_evolution_equation)
  {
    // ------- Hydrology Evolution Residual -------- //
    p = rcp(new Teuchos::ParameterList("Hydrology Residual Evolution"));

    //Input
    p->set<std::string> ("BF Name", "BF " + basalSideName);
    p->set<std::string> ("Weighted Measure Name", "Weighted Measure " + basalSideName);
    p->set<std::string> ("Drainage Sheet Depth QP Variable Name","Drainage Sheet Depth");
    p->set<std::string> ("Drainage Sheet Depth Dot QP Variable Name","Drainage Sheet Depth Dot");
    p->set<std::string> ("Effective Pressure QP Variable Name","Effective Pressure");
    p->set<std::string> ("Melting Rate QP Variable Name","Melting Rate");
    p->set<std::string> ("Sliding Velocity QP Variable Name","Sliding Velocity");
    p->set<std::string> ("Side Set Name", basalSideName);
    p->set<Teuchos::ParameterList*> ("FELIX Hydrology",&params->sublist("FELIX Hydrology"));
    p->set<Teuchos::ParameterList*> ("FELIX Physics",&params->sublist("FELIX Physics"));

    //Output
    p->set<std::string> ("Residual Evolution Eqn Variable Name", hydro_resid_names[1]);

    ev = rcp(new FELIX::HydrologyResidualEvolutionEqn<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Stokes", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
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
