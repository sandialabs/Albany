/*
 * FELIX_Enthalpy.hpp
 *
 *  Created on: May 10, 2016
 *      Author: abarone
 */

#ifndef FELIX_ENTHALPY_PROBLEM_HPP
#define FELIX_ENTHALPY_PROBLEM_HPP

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Phalanx.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_SaveCellStateField.hpp"
#include "FELIX_SharedParameter.hpp"
#include "FELIX_ParamEnum.hpp"

#include "FELIX_EnthalpyResid.hpp"
#include "FELIX_w_ZResid.hpp"
#include "FELIX_ViscosityFO.hpp"
#include "FELIX_Dissipation.hpp"
#include "FELIX_BasalFrictionHeat.hpp"
#include "FELIX_GeoFluxHeat.hpp"
#include "FELIX_HydrostaticPressure.hpp"
#include "FELIX_LiquidWaterFraction.hpp"
#include "FELIX_PressureMeltingEnthalpy.hpp"
#include "FELIX_PressureMeltingTemperature.hpp"
#include "FELIX_Temperature.hpp"
#include "FELIX_Integral1Dw_Z.hpp"
#include "FELIX_VerticalVelocity.hpp"
#include "FELIX_BasalMeltRate.hpp"


namespace FELIX
{

  class Enthalpy : public Albany::AbstractProblem
  {
  public:
    //! Default constructor
    Enthalpy(const Teuchos::RCP<Teuchos::ParameterList>& params,
             const Teuchos::RCP<ParamLib>& paramLib,
             const int numDim_);

    //! Destructor
    ~Enthalpy();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                              Albany::StateManager& stateMgr);

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> > buildEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                                                                const Albany::MeshSpecsStruct& meshSpecs,
                                                                                Albany::StateManager& stateMgr,
                                                                                Albany::FieldManagerChoice fmchoice,
                                                                                const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! Each problem must generate its list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                        const Albany::MeshSpecsStruct& meshSpecs,
                        Albany::StateManager& stateMgr,
                        Albany::FieldManagerChoice fmchoice,
                        const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

  protected:
    Teuchos::RCP<shards::CellTopology> cellType;
    Teuchos::RCP<shards::CellTopology> basalSideType;

    Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  cellCubature;
    Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  basalCubature;

    Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > cellBasis;
    Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > basalSideBasis;

    int numDim;
    Teuchos::RCP<Albany::Layouts> dl, dl_basal;
    std::string elementBlockName;

    bool needsDiss, needsBasFric;
    bool isGeoFluxConst;

    std::string basalSideName, basalEBName;
  };

} // end of the namespace FELIX

// ================================ IMPLEMENTATION ============================ //
template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
FELIX::Enthalpy::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                      const Albany::MeshSpecsStruct& meshSpecs,
                                      Albany::StateManager& stateMgr,
                                      Albany::FieldManagerChoice fieldManagerChoice,
                                      const Teuchos::RCP<Teuchos::ParameterList>& responseList)
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

  Albany::StateStruct::MeshFieldEntity entity;

  Teuchos::RCP<ParameterList> p;

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;

  // Here is how to register the field for dirichlet condition.
  // Enthalpy Dirichlet field on the surface
  {
    entity = Albany::StateStruct::NodalDistParameter;
    std::string stateName = "surface_air_enthalpy";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, "");
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Velocity
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    std::string stateName = "velocity";
    p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, true, &entity, "");
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Flow factor - actually, this is not used if viscosity is temperature based
  {
    entity = Albany::StateStruct::ElemData;
    std::string stateName = "flow_factor";
    p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, elementBlockName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Basal friction
  if(needsBasFric)
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    std::string stateName = "basal_friction";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Geothermal flux
  if(!isGeoFluxConst)
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    std::string stateName = "basal_heat_flux";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Thickness
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    std::string stateName = "thickness";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Surface Height
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    std::string stateName = "surface_height";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  int offset = 0;

  // --- Interpolation and utilities ---
  // Enthalpy
  {
    Teuchos::ArrayRCP<string> dof_names(1);
    Teuchos::ArrayRCP<string> resid_names(1);
    dof_names[0] = "Enthalpy";
    resid_names[0] = "Enthalpy Residual";

    // no transient
    fm0.template registerEvaluator<EvalT> (evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names, offset));

    fm0.template registerEvaluator<EvalT> (evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Enthalpy"));

    fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

    fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

    // --- Restrict enthalpy from cell-based to cell-side-based
    fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFCellToSideEvaluator(dof_names[0],basalSideName,"Node Scalar",cellType));

    //fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator(dof_names[0], basalSideName));

    offset++;
  }

  // w_z
  {
    Teuchos::ArrayRCP<string> dof_names(1);
    Teuchos::ArrayRCP<string> resid_names(1);
    dof_names[0] = "w_z";
    resid_names[0] = "w_z Residual";

    // no transient
    fm0.template registerEvaluator<EvalT> (evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names, offset));

    fm0.template registerEvaluator<EvalT> (evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter w_z"));

    fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));
  }

  fm0.template registerEvaluator<EvalT> (evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT> (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature));

  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFVecInterpolationEvaluator("velocity"));

  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFVecGradInterpolationEvaluator("velocity"));

  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFGradInterpolationEvaluator("melting temp"));

  // Interpolate temperature from nodes to cell
  if(needsDiss)
    fm0.template registerEvaluator<EvalT> (evalUtils.constructNodesToCellInterpolationEvaluator("Temperature",false));

  // Interpolate pressure melting temperature gradient from nodes to QPs
  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator("melting temp",basalSideName));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("melting temp"));

  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("melting enthalpy"));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("phi"));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFGradInterpolationEvaluator("phi"));

  // --- Special evaluators for side handling --- //

  // --- Restrict vertex coordinates from cell-based to cell-side-based
  fm0.template registerEvaluator<EvalT> (evalUtils.getMSTUtils().constructDOFCellToSideEvaluator("Coord Vec",basalSideName,"Vertex Vector",cellType,
                                                                                                 "Coord Vec " + basalSideName));

  // --- Compute side basis functions
  fm0.template registerEvaluator<EvalT> (evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, basalSideBasis, basalCubature, basalSideName));

  // --- Compute Quad Points coordinates on the side set
  fm0.template registerEvaluator<EvalT> (evalUtils.constructMapToPhysicalFrameSideEvaluator(cellType,basalCubature,basalSideName));

  // --- Restrict basal velocity from cell-based to cell-side-based
  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("velocity",basalSideName,"Node Vector",cellType));

  // --- Restrict vertical velocity from cell-based to cell-side-based
  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFCellToSideEvaluator("w",basalSideName,"Node Scalar",cellType));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationSideEvaluator("w", basalSideName));

  // --- Restrict enthalpy Hs from cell-based to cell-side-based
  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("melting enthalpy",basalSideName,"Node Scalar",cellType));

  // --- Interpolate velocity on QP on side
  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("velocity", basalSideName));

  if(needsBasFric)
  {
    // --- Restrict basal friction from cell-based to cell-side-based
    fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("basal_friction",basalSideName,"Node Scalar",cellType));

    // --- Interpolate Beta Given on QP on side
    fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("basal_friction", basalSideName));

    fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("Basal Heat"));

    fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("Basal Heat SUPG"));
  }

  // --- Utilities for Basal Melt Rate
  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("melting temp",basalSideName,"Node Scalar",cellType));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFCellToSideEvaluator("phi",basalSideName,"Node Scalar",cellType));

  // --- Utilities for Geothermal flux
  if(!isGeoFluxConst)
  {
    // --- Restrict geothermal flux from cell-based to cell-side-based
    fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("basal_heat_flux",basalSideName,"Node Scalar",cellType));

    // --- Interpolate geothermal_flux on QP on side
    fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("basal_heat_flux", basalSideName));

    fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("Geo Flux Heat"));

    fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("Geo Flux Heat SUPG"));
  }

  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("w"));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFSideToCellEvaluator("basal_melt_rate",basalSideName,"Node Scalar",cellType,"basal_melt_rate"));

  // -------------------------------- FELIX evaluators ------------------------- //

  // --- Enthalpy Residual ---
  {
    p = rcp(new ParameterList("Enthalpy Resid"));

    //Input
    p->set<string>("Weighted BF Variable Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);

    p->set<string>("Weighted Gradient BF Variable Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<string>("Enthalpy QP Variable Name", "Enthalpy");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Enthalpy Gradient QP Variable Name", "Enthalpy Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_gradient);

    p->set<std::string>("Enthalpy Hs QP Variable Name", "melting enthalpy");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<std::string>("Coordinate Vector Name", "Coord Vec");

    // Velocity field for the convective term (read from the mesh)
    p->set<string>("Velocity QP Variable Name", "velocity");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    // Vertical velocity derived from the continuity equation
    p->set<string>("Vertical Velocity QP Variable Name", "w");

    p->set<string>("Geothermal Flux Heat QP Variable Name","Geo Flux Heat");
    p->set<string>("Geothermal Flux Heat QP SUPG Variable Name","Geo Flux Heat SUPG");

    p->set<string>("Melting Temperature Gradient QP Variable Name","melting temp Gradient");

    if(needsDiss)
    {
      p->set<std::string>("Dissipation QP Variable Name", "FELIX Dissipation");
    }

    if(needsBasFric)
    {
      p->set<std::string>("Basal Friction Heat QP Variable Name", "Basal Heat");
      p->set<std::string>("Basal Friction Heat QP SUPG Variable Name", "Basal Heat SUPG");
    }

    p->set<string>("Water Content QP Variable Name","phi");
    p->set<string>("Water Content Gradient QP Variable Name","phi Gradient");

    p->set<bool>("Needs Dissipation", needsDiss);
    p->set<bool>("Needs Basal Friction", needsBasFric);
    p->set<bool>("Constant Geothermal Flux", isGeoFluxConst);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

    p->set<ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));
    if(params->isSublist("FELIX Enthalpy Stabilization"))
      p->set<ParameterList*>("FELIX Enthalpy Stabilization", &params->sublist("FELIX Enthalpy Stabilization"));

    p->set<std::string>("Velocity Gradient QP Variable Name", "velocity Gradient");

    //Output
    p->set<string>("Residual Variable Name", "Enthalpy Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new FELIX::EnthalpyResid<EvalT,AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // --- w_z Residual ---
  {
    p = rcp(new ParameterList("w_z Resid"));

    //Input
    p->set<string>("Weighted BF Variable Name", "wBF");
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);

    p->set<string>("w_z QP Variable Name", "w_z");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<std::string>("Velocity Gradient QP Variable Name", "velocity Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vecgradient);

    //Output
    p->set<string>("Residual Variable Name", "w_z Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new FELIX::w_ZResid<EvalT,AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // --- FELIX Dissipation ---
  if(needsDiss)
  {
    {
      p = rcp(new ParameterList("FELIX Dissipation"));

      //Input
      p->set<std::string>("Viscosity QP Variable Name", "FELIX Viscosity");
      p->set<std::string>("EpsilonSq QP Variable Name", "FELIX EpsilonSq");

      //Output
      p->set<std::string>("Dissipation QP Variable Name", "FELIX Dissipation");

      ev = Teuchos::rcp(new FELIX::Dissipation<EvalT,PHAL::AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // --- FELIX Viscosity ---
    {
      p = rcp(new ParameterList("FELIX Viscosity"));

      //Input
      p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");
      p->set<std::string>("Velocity QP Variable Name", "velocity");
      p->set<std::string>("Velocity Gradient QP Variable Name", "velocity Gradient");
      p->set<std::string>("Temperature Variable Name", "Temperature");
      p->set<std::string>("Flow Factor Variable Name", "flow_factor");

      p->set<ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
      p->set<ParameterList*>("Parameter List", &params->sublist("FELIX Viscosity"));
      p->set<RCP<ParamLib> >("Parameter Library", paramLib);
      p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

      //Output
      p->set<std::string>("Viscosity QP Variable Name", "FELIX Viscosity");
      p->set<std::string>("EpsilonSq QP Variable Name", "FELIX EpsilonSq");

      ev = Teuchos::rcp(new FELIX::ViscosityFO<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::ScalarT>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // --- FELIX Basal friction heat ---
  if(needsBasFric)
  {
    p = rcp(new ParameterList("FELIX Basal Friction Heat"));
    //Input
    p->set<std::string>("BF Side Name", "BF "+basalSideName);
    p->set<std::string>("Gradient BF Side Name", "Grad BF "+basalSideName);
    p->set<std::string>("Weighted Measure Name", "Weighted Measure "+basalSideName);
    p->set<std::string>("Velocity Side QP Variable Name", "velocity");
    p->set<std::string>("Basal Friction Coefficient Side QP Variable Name", "basal_friction");

    p->set<std::string>("Side Set Name", basalSideName);

    if(params->isSublist("FELIX Enthalpy Stabilization"))
      p->set<ParameterList*>("FELIX Enthalpy Stabilization", &params->sublist("FELIX Enthalpy Stabilization"));

    p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);

    //Output
    p->set<std::string>("Basal Friction Heat Variable Name", "Basal Heat");
    p->set<std::string>("Basal Friction Heat SUPG Variable Name", "Basal Heat SUPG");

    ev = Teuchos::rcp(new FELIX::BasalFrictionHeat<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // --- FELIX Geothermal flux heat
  {
    p = rcp(new ParameterList("FELIX Geothermal Flux Heat"));
    //Input
    p->set<std::string>("BF Side Name", "BF "+basalSideName);
    p->set<std::string>("Gradient BF Side Name", "Grad BF "+basalSideName);
    p->set<std::string>("Weighted Measure Name", "Weighted Measure "+basalSideName);
    p->set<std::string>("Velocity Side QP Variable Name", "velocity");
    p->set<std::string>("Vertical Velocity Side QP Variable Name", "w");

    p->set<std::string>("Side Set Name", basalSideName);
    p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);

    if(params->isSublist("FELIX Enthalpy Stabilization"))
      p->set<ParameterList*>("FELIX Enthalpy Stabilization", &params->sublist("FELIX Enthalpy Stabilization"));

    if(!isGeoFluxConst)
      p->set<std::string>("Geothermal Flux Side QP Variable Name", "basal_heat_flux");

    p->set<bool>("Constant Geothermal Flux", isGeoFluxConst);

    //Output
    p->set<std::string>("Geothermal Flux Heat Variable Name", "Geo Flux Heat");
    p->set<std::string>("Geothermal Flux Heat SUPG Variable Name", "Geo Flux Heat SUPG");

    ev = Teuchos::rcp(new FELIX::GeoFluxHeat<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // --- FELIX hydrostatic pressure
  {
    p = rcp(new ParameterList("FELIX Hydrostatic Pressure"));

    //Input
    p->set<std::string>("Surface Height Variable Name", "surface_height");
    p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");

    p->set<ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

    //Output
    p->set<std::string>("Hydrostatic Pressure Variable Name", "Hydrostatic Pressure");

    ev = Teuchos::rcp(new FELIX::HydrostaticPressure<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // --- FELIX pressure-melting temperature
  {
    p = rcp(new ParameterList("FELIX Pressure Melting Temperature"));

    //Input
    p->set<std::string>("Hydrostatic Pressure Variable Name", "Hydrostatic Pressure");

    p->set<ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

    //Output
    p->set<std::string>("Melting Temperature Variable Name", "melting temp");

    ev = Teuchos::rcp(new FELIX::PressureMeltingTemperature<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    // Saving the melting temperature in the output mesh
    {
      fm0.template registerEvaluator<EvalT> (evalUtils.constructNodesToCellInterpolationEvaluator("melting temp",false));

      std::string stateName = "MeltingTemperature_Cell";
      p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, dl->dummy, elementBlockName, "scalar", 0.0, /* save state = */ false, /* write output = */ true);
      p->set<std::string>("Field Name", "melting temp");
      p->set<std::string>("Weights Name","Weights");
      p->set("Weights Layout", dl->qp_scalar);
      p->set("Field Layout", dl->cell_scalar2);
      p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);
      ev = rcp(new PHAL::SaveCellStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // --- FELIX pressure-melting enthalpy
  {
    p = rcp(new ParameterList("FELIX Pressure Melting Enthalpy"));

    //Input
    p->set<std::string>("Melting Temperature Variable Name", "melting temp");

    p->set<ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

    //Output
    p->set<std::string>("Enthalpy Hs Variable Name", "melting enthalpy");
    ev = Teuchos::rcp(new FELIX::PressureMeltingEnthalpy<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // --- FELIX Temperature: diff enthalpy is h - hs.
  {
    p = rcp(new ParameterList("FELIX Temperature"));

    //Input
    p->set<std::string>("Melting Temperature Variable Name", "melting temp");
    p->set<std::string>("Enthalpy Hs Variable Name", "melting enthalpy");
    p->set<std::string>("Enthalpy Variable Name", "Enthalpy");

    p->set<ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

    //Output
    p->set<std::string>("Temperature Variable Name", "Temperature");
    p->set<std::string>("Diff Enthalpy Variable Name", "Diff Enth");

    ev = Teuchos::rcp(new FELIX::Temperature<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    // Saving the temperature in the output mesh
    {
      std::string stateName = "Temperature_Cell";
      p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, dl->dummy, elementBlockName, "scalar", 0.0, /* save state = */ false, /* write output = */ true);
      p->set<std::string>("Field Name", "Temperature");
      p->set<std::string>("Weights Name","Weights");
      p->set("Weights Layout", dl->qp_scalar);
      p->set("Field Layout", dl->cell_scalar2);
      p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);

      ev = rcp(new PHAL::SaveCellStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      // Forcing the execution of the evaluator
      if (fieldManagerChoice == Albany::BUILD_RESID_FM)
      {
        if (ev->evaluatedFields().size()>0)
        {
          fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
        }
      }
    }

    // Saving the diff enthalpy field in the output mesh
    {
      fm0.template registerEvaluator<EvalT> (evalUtils.constructNodesToCellInterpolationEvaluator("Diff Enth",false));

      std::string stateName = "h-hs_Cell";
      p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, dl->dummy, elementBlockName, "scalar", 0.0, /* save state = */ false, /* write output = */ true);
      p->set<std::string>("Field Name", "Diff Enth");
      p->set<std::string>("Weights Name","Weights");
      p->set("Weights Layout", dl->qp_scalar);
      p->set("Field Layout", dl->cell_scalar2);
      p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);

      ev = rcp(new PHAL::SaveCellStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // --- FELIX Liquid Water Fraction
  {
    p = rcp(new ParameterList("FELIX Liquid Water Fraction"));

    //Input
    p->set<std::string>("Enthalpy Hs Variable Name", "melting enthalpy");
    p->set<std::string>("Enthalpy Variable Name", "Enthalpy");

    p->set<ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

    //Output
    p->set<std::string>("Water Content Variable Name", "phi");
    ev = Teuchos::rcp(new FELIX::LiquidWaterFraction<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    // Saving phi in the output mesh
    {
      fm0.template registerEvaluator<EvalT> (evalUtils.constructNodesToCellInterpolationEvaluator("phi",false));

      std::string stateName = "phi";
      p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, dl->dummy, elementBlockName, "scalar", 0.0, /* save state = */ false, /* write output = */ true);

      p->set<std::string>("Weights Name","Weights");
      p->set("Weights Layout", dl->qp_scalar);
      p->set("Field Layout", dl->cell_scalar2);
      p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);

      ev = rcp(new PHAL::SaveCellStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Forcing the execution of the evaluator
    if (fieldManagerChoice == Albany::BUILD_RESID_FM)
    {
      if (ev->evaluatedFields().size()>0)
      {
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
      }
    }
  }

  // --- FELIX Integral 1D w_z
  {
    p = rcp(new ParameterList("FELIX Integral 1D w_z"));

    p->set<std::string>("Basal Melt Rate Variable Name", "basal_melt_rate");
    p->set<std::string>("Thickness Variable Name", "thickness");

    p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));

    p->set<bool>("Stokes and Thermo coupled", false);

    //Output
    p->set<std::string>("Integral1D w_z Variable Name", "w");
    ev = Teuchos::rcp(new FELIX::Integral1Dw_Z<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // --- FELIX Vertical Velocity
  {
    /*    p = rcp(new ParameterList("FELIX Vertical Velocity"));

      //Input
      p->set<std::string>("Thickness Variable Name", "thickness");
      p->set<std::string>("Integral1D w_z Variable Name", "int1Dw_z");

      //Output
      p->set<std::string>("Vertical Velocity Variable Name", "w");
      ev = Teuchos::rcp(new FELIX::VerticalVelocity<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);
     */
    {
      fm0.template registerEvaluator<EvalT> (evalUtils.constructNodesToCellInterpolationEvaluator("w",false));

      std::string stateName = "w";
      entity = Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, dl->dummy, elementBlockName, "scalar", 0.0, /* save state = */ false, /* write output = */ true);

      p->set<std::string>("Weights Name","Weights");
      p->set("Weights Layout", dl->qp_scalar);
      p->set("Field Layout", dl->cell_scalar2);
      p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);

      ev = rcp(new PHAL::SaveCellStateField<EvalT,PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // --- FELIX Basal Melt Rate
  {
    p = rcp(new ParameterList("FELIX Basal Melt Rate"));

    //Input
    p->set<std::string>("Water Content Side Variable Name", "phi");
    p->set<std::string>("Geothermal Flux Side Variable Name", "basal_heat_flux");
    p->set<std::string>("Velocity Side Variable Name", "velocity");
    p->set<std::string>("Basal Friction Coefficient Side Variable Name", "basal_friction");
    p->set<std::string>("Enthalpy Hs Side Variable Name", "melting enthalpy");
    p->set<std::string>("Enthalpy Side Variable Name", "Enthalpy");
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

    p->set<ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

    p->set<std::string>("Side Set Name", basalSideName);

    //Output
    p->set<std::string>("Basal Melt Rate Variable Name", "basal_melt_rate");
    ev = Teuchos::rcp(new FELIX::BasalMeltRate<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);

    {
      fm0.template registerEvaluator<EvalT> (evalUtils.constructNodesToCellInterpolationEvaluator("basal_melt_rate",false));

      std::string stateName = "basal_melt_rate";
      entity = Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, dl->dummy, elementBlockName, "scalar", 0.0, /* save state = */ false, /* write output = */ true);

      p->set<std::string>("Weights Name","Weights");
      p->set("Weights Layout", dl->qp_scalar);
      p->set("Field Layout", dl->cell_scalar2);
      p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);
      ev = rcp(new PHAL::SaveCellStateField<EvalT,PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  {
    //--- Shared Parameter for homotopy parameter: h ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Glen's Law Homotopy Parameter"));

    std::string param_name = "Glen's Law Homotopy Parameter";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Homotopy>> ptr_h;
    ptr_h = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Homotopy>(*p,dl));
    ptr_h->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Viscosity").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_h);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    PHX::Tag<typename EvalT::ScalarT> res_tag0("Scatter Enthalpy", dl->dummy);
    fm0.requireField<EvalT>(res_tag0);
    PHX::Tag<typename EvalT::ScalarT> res_tag1("Scatter w_z", dl->dummy);
    fm0.requireField<EvalT>(res_tag1);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }

  return Teuchos::null;
}

#endif /* FELIX_ENTHALPY_PROBLEM_HPP */
