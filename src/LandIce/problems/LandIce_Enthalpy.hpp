/*
 * LandIce_Enthalpy.hpp
 *
 *  Created on: May 10, 2016
 *      Author: abarone
 */

#ifndef LANDICE_ENTHALPY_PROBLEM_HPP
#define LANDICE_ENTHALPY_PROBLEM_HPP

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"
#include "LandIce_ResponseUtilities.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_SaveCellStateField.hpp"
#include "PHAL_SaveStateField.hpp"
#include "PHAL_LoadSideSetStateField.hpp"
#include "PHAL_ScatterScalarNodalParameter.hpp"
#include "PHAL_SharedParameter.hpp"
#include "LandIce_ParamEnum.hpp"

#include "LandIce_EnthalpyResid.hpp"
#include "LandIce_EnthalpyBasalResid.hpp"
#include "LandIce_w_Resid.hpp"
#include "LandIce_ViscosityFO.hpp"
#include "LandIce_Dissipation.hpp"
#include "LandIce_HydrostaticPressure.hpp"
#include "LandIce_LiquidWaterFraction.hpp"
#include "LandIce_PressureMeltingEnthalpy.hpp"
#include "LandIce_Temperature.hpp"
#include "LandIce_VerticalVelocity.hpp"
#include "LandIce_BasalMeltRate.hpp"
#include "LandIce_SurfaceAirEnthalpy.hpp"


namespace LandIce
{

  class Enthalpy : public Albany::AbstractProblem
  {
  public:
    //! Default constructor
    Enthalpy(const Teuchos::RCP<Teuchos::ParameterList>& params,
             const Teuchos::RCP<Teuchos::ParameterList>& discParams,
             const Teuchos::RCP<ParamLib>& paramLib,
             const int numDim_);

    //! Destructor
    ~Enthalpy();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Get boolean telling code if SDBCs are utilized
    virtual bool useSDBCs() const {return use_sdbcs_; }

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

    unsigned int numDim;
    Teuchos::RCP<Albany::Layouts> dl, dl_basal;
    std::string elementBlockName;

    //! Discretization parameters
    Teuchos::RCP<Teuchos::ParameterList> discParams;

    std::set<std::string> volumeFields;

    bool needsDiss, needsBasFric;

    std::string basalSideName, basalEBName;

    /// Boolean marking whether SDBCs are used
    bool use_sdbcs_;
  };

} // end of the namespace LandIce

// ================================ IMPLEMENTATION ============================ //
template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
LandIce::Enthalpy::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                      const Albany::MeshSpecsStruct& /* meshSpecs */,
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

  using FRT = Albany::FieldRankType;

  Albany::StateStruct::MeshFieldEntity entity;

  Teuchos::RCP<ParameterList> p;

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;

  // Here is how to register the field for dirichlet condition.
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    std::string stateName = "surface_air_temperature";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, "");
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, stateName, dl_basal->node_scalar, basalEBName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Velocity
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    std::string stateName = "velocity";
    p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, true, &entity, "");
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    if (discParams->isSublist("Side Set Discretizations") &&
        discParams->sublist("Side Set Discretizations").isSublist("basalside") &&
        discParams->sublist("Side Set Discretizations").sublist("basalside").isSublist("Required Fields Info")){
      Teuchos::ParameterList& req_fields_info = discParams->sublist("Side Set Discretizations").sublist("basalside").sublist("Required Fields Info");
      unsigned int num_fields = req_fields_info.get<int>("Number Of Fields",0);
      for (unsigned int ifield=0; ifield<num_fields; ++ifield)
      {
        const Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));
        if(thisFieldList.get<std::string>("Field Name") ==  stateName){
          unsigned int numLayers = thisFieldList.get<int>("Number Of Layers");
          auto sns = dl_basal->node_vector;
          auto dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,VecDim, LayerDim>(sns->extent(0),sns->extent(1),sns->extent(2),2, numLayers));
          stateMgr.registerSideSetStateVariable(basalSideName, stateName, stateName, dl_temp, basalEBName, true, &entity);
        }
      }
    }
  }

  // Vertical Velocity
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    std::string stateName = "vertical_velocity";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, "");
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    if (discParams->isSublist("Side Set Discretizations") &&
        discParams->sublist("Side Set Discretizations").isSublist("basalside") &&
        discParams->sublist("Side Set Discretizations").sublist("basalside").isSublist("Required Fields Info")){
      Teuchos::ParameterList& req_fields_info = discParams->sublist("Side Set Discretizations").sublist("basalside").sublist("Required Fields Info");
      unsigned int num_fields = req_fields_info.get<int>("Number Of Fields",0);
      for (unsigned int ifield=0; ifield<num_fields; ++ifield)
      {
        const Teuchos::ParameterList& thisFieldList =  req_fields_info.sublist(Albany::strint("Field", ifield));
        if(thisFieldList.get<std::string>("Field Name") ==  stateName){
          unsigned int numLayers = thisFieldList.get<int>("Number Of Layers");
          auto sns = dl_basal->node_vector;
          auto dl_temp = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,LayerDim>(sns->extent(0),sns->extent(1),sns->extent(2), numLayers));
          stateMgr.registerSideSetStateVariable(basalSideName, stateName, stateName, dl_temp, basalEBName, true, &entity);
        }
      }
    }
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

    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, stateName, dl_basal->node_scalar_sideset, basalEBName, true, &entity, "", true);
    ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Geothermal flux
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    std::string stateName = "heat_flux";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, stateName, dl_basal->node_scalar_sideset, basalEBName, true, &entity, "", true);
    ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Thickness
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    std::string stateName = "ice_thickness";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, stateName, dl_basal->node_scalar, basalEBName, true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Surface Height
  {
    entity = Albany::StateStruct::NodalDataToElemNode;
    std::string stateName = "surface_height";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, stateName, dl_basal->node_scalar_sideset, basalEBName, true, &entity, "", true);
    ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
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
    fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFCellToSideEvaluator(dof_names[0],basalSideName,"Node Scalar Sideset",cellType));

    fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationSideEvaluator(dof_names[0], basalSideName));
  }

  fm0.template registerEvaluator<EvalT> (evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT> (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature));

  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFVecInterpolationEvaluator("velocity"));

  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFVecGradInterpolationEvaluator("velocity"));

  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("vertical_velocity"));

  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFGradInterpolationEvaluator("melting_temperature"));

  // Interpolate temperature from nodes to cell
  fm0.template registerEvaluator<EvalT> (evalUtils.constructBarycenterEvaluator("temperature", cellBasis, FRT::Scalar));

  // Interpolate pressure melting temperature gradient from nodes to QPs
  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator("melting temp",basalSideName));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("melting_temperature"));

  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("melting enthalpy"));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("phi"));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFGradInterpolationEvaluator("phi"));

  // --- Special evaluators for side handling --- //

  // --- Restrict vertex coordinates from cell-based to cell-side-based
  fm0.template registerEvaluator<EvalT> (evalUtils.getMSTUtils().constructDOFCellToSideEvaluator("Coord Vec",basalSideName,"Vertex Vector Sideset",cellType,
                                                                                                 "Coord Vec " + basalSideName));

  // --- Compute side basis functions
  fm0.template registerEvaluator<EvalT> (evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, basalSideBasis, basalCubature, basalSideName));

  // --- Compute Quad Points coordinates on the side set
  fm0.template registerEvaluator<EvalT> (evalUtils.constructMapToPhysicalFrameSideEvaluator(cellType,basalCubature,basalSideName));

  // --- Restrict basal velocity from cell-based to cell-side-based
  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("velocity",basalSideName,"Node Vector Sideset",cellType));

  // --- Restrict vertical velocity from cell-based to cell-side-based
  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFCellToSideEvaluator("vertical_velocity",basalSideName,"Node Scalar",cellType));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationSideEvaluator("basal_dTdz", basalSideName));

  // --- Restrict enthalpy Hs from cell-based to cell-side-based
  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("melting enthalpy",basalSideName,"Node Scalar Sideset",cellType));
  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("melting enthalpy", basalSideName));

  // --- Interpolate velocity on QP on side
  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("velocity", basalSideName));

  if(needsBasFric)
  {
    // --- Interpolate Beta Given on QP on side
    fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("basal_friction", basalSideName));
  }

  // --- Utilities for Basal Melt Rate
  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("melting temp",basalSideName,"Node Scalar",cellType));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFCellToSideEvaluator("phi",basalSideName,"Node Scalar Sideset",cellType));
  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationSideEvaluator("phi",basalSideName));

  // --- Utilities for Geothermal flux

  // --- Interpolate geothermal_flux on QP on side
  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("heat_flux", basalSideName));

  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFSideToCellEvaluator("basal_vert_velocity",basalSideName,"Node Scalar",cellType,"basal_vert_velocity"));

  // -------------------------------- LandIce evaluators ------------------------- //

  // --- Enthalpy Residual ---
  {
    p = rcp(new ParameterList("Enthalpy Resid"));

    //Input
    p->set<string>("Weighted BF Variable Name", Albany::weighted_bf_name);
    p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);

    p->set<string>("Weighted Gradient BF Variable Name", Albany::weighted_grad_bf_name);
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    p->set<string>("Enthalpy QP Variable Name", "Enthalpy");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<string>("Enthalpy Gradient QP Variable Name", "Enthalpy Gradient");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_gradient);

    p->set<std::string>("Enthalpy Hs QP Variable Name", "melting enthalpy");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<std::string>("Diff Enthalpy Variable Name", "Diff Enth");

    p->set<std::string>("Coordinate Vector Name", "Coord Vec");

    // Velocity field for the convective term (read from the mesh)
    p->set<string>("Velocity QP Variable Name", "velocity");
    p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

    // Vertical velocity derived from the continuity equation
    p->set<string>("Vertical Velocity QP Variable Name", "vertical_velocity");

    p->set<string>("Melting Temperature Gradient QP Variable Name","melting_temperature Gradient");

    if(needsDiss)
    {
      p->set<std::string>("Dissipation QP Variable Name", "LandIce Dissipation");
    }

    p->set<string>("Water Content QP Variable Name","phi");
    p->set<string>("Water Content Gradient QP Variable Name","phi Gradient");

    p->set<bool>("Needs Dissipation", needsDiss);
    p->set<bool>("Needs Basal Friction", needsBasFric);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

    p->set<ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));
    p->set<Teuchos::ParameterList*>("LandIce Enthalpy Regularization", &params->sublist("LandIce Enthalpy",false).sublist("Regularization", false));
    if(params->isSublist("LandIce Enthalpy") &&  params->sublist("LandIce Enthalpy").isParameter("Stabilization"))
      p->set<ParameterList*>("LandIce Enthalpy Stabilization", &params->sublist("LandIce Enthalpy").sublist("Stabilization"));

    p->set<std::string>("Velocity Gradient QP Variable Name", "velocity Gradient");

    p->set<std::string>("Enthalpy Basal Residual Variable Name", "Enthalpy Basal Residual");

    //Output
    p->set<string>("Residual Variable Name", "Enthalpy Residual");
    p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

    ev = rcp(new LandIce::EnthalpyResid<EvalT,AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // --- Enthalpy Basal Residual ---
  {
    p = rcp(new ParameterList("Enthalpy Basal Resid"));

    //Input

    p->set<std::string>("BF Side Name", Albany::bf_name +" "+basalSideName);
    p->set<std::string>("Weighted Measure Side Name", Albany::weighted_measure_name+" "+basalSideName);
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<std::string>("Basal Melt Rate Side QP Variable Name", "basal_neumann_term");

    //Output
    p->set<std::string>("Enthalpy Basal Residual Variable Name", "Enthalpy Basal Residual");

    ev = rcp(new LandIce::EnthalpyBasalResid<EvalT,AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  // --- LandIce Dissipation ---
  if(needsDiss)
  {
    {
      p = rcp(new ParameterList("LandIce Dissipation"));

      //Input
      p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
      p->set<std::string>("EpsilonSq QP Variable Name", "LandIce EpsilonSq");
      p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

      //Output
      p->set<std::string>("Dissipation QP Variable Name", "LandIce Dissipation");

      ev = Teuchos::rcp(new LandIce::Dissipation<EvalT,PHAL::AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    std::string stateName = "heat_dissipation";
    if(volumeFields.find(stateName) != volumeFields.end())
    { //save
      std::string fieldName = "LandIce Dissipation";
      entity = Albany::StateStruct::QuadPoint;
      p = stateMgr.registerStateVariable(stateName, dl->qp_scalar, elementBlockName, true, &entity, "");
      p->set<std::string>("Field Name", fieldName);
      p->set("Field Layout", dl->qp_scalar);
      p->set<bool>("Nodal State", false);

      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((fieldManagerChoice == Albany::BUILD_RESID_FM)&&(ev->evaluatedFields().size()>0))
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }

    // --- LandIce Viscosity ---
    {
      p = rcp(new ParameterList("LandIce Viscosity"));

      //Input
      p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");
      p->set<std::string>("Velocity QP Variable Name", "velocity");
      p->set<std::string>("Velocity Gradient QP Variable Name", "velocity Gradient");
      p->set<std::string>("Temperature Variable Name", "temperature");
      p->set<std::string>("Flow Factor Variable Name", "flow_factor");

      p->set<ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
      p->set<ParameterList*>("Parameter List", &params->sublist("LandIce Viscosity"));
      p->set<RCP<ParamLib> >("Parameter Library", paramLib);
      p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

      //Output
      p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
      p->set<std::string>("EpsilonSq QP Variable Name", "LandIce EpsilonSq");

      ev = Teuchos::rcp(new LandIce::ViscosityFO<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::ScalarT>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // --- LandIce hydrostatic pressure
  {
    p = rcp(new ParameterList("LandIce Hydrostatic Pressure"));

    //Input
    p->set<std::string>("Surface Height Variable Name", "surface_height");
    p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");

    p->set<ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

    //Output
    p->set<std::string>("Hydrostatic Pressure Variable Name", "Hydrostatic Pressure");

    ev = Teuchos::rcp(new LandIce::HydrostaticPressure<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  // --- LandIce pressure-melting enthalpy
  {
    p = rcp(new ParameterList("LandIce Pressure Melting Enthalpy"));

    //Input
    p->set<std::string>("Hydrostatic Pressure Variable Name", "Hydrostatic Pressure");

    p->set<ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

    //Output
    p->set<std::string>("Melting Temperature Variable Name", "melting_temperature");

    p->set<std::string>("Enthalpy Hs Variable Name", "melting enthalpy");

    ev = Teuchos::rcp(new LandIce::PressureMeltingEnthalpy<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    std::string stateName = "melting_temperature";
    if(volumeFields.find(stateName) != volumeFields.end())
    { // Saving the melting temperature in the output mesh

      std::string fieldName = "melting_temperature";
      entity = Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, "");
      p->set<std::string>("Field Name", fieldName);
      p->set("Field Layout", dl->node_scalar);
      p->set<bool>("Nodal State", true);

      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((fieldManagerChoice == Albany::BUILD_RESID_FM)&&(ev->evaluatedFields().size()>0))
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
  }

  // --- LandIce Surface Air Enthalpy
  {
    p = rcp(new ParameterList("LandIce surface Air Enthalpy"));

    //Input
    p->set<ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));
    p->set<std::string>("Surface Air Temperature Name", "surface_air_temperature");

    //Output
    p->set<std::string>("Surface Air Enthalpy Name", "surface_enthalpy");

    ev = Teuchos::rcp(new LandIce::SurfaceAirEnthalpy<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // --- LandIce Temperature: diff enthalpy is h - hs.
  {
    p = rcp(new ParameterList("LandIce Temperature"));

    //Input
    p->set<std::string>("Melting Temperature Variable Name", "melting_temperature");
    p->set<std::string>("Enthalpy Hs Variable Name", "melting enthalpy");
    p->set<std::string>("Enthalpy Variable Name", "Enthalpy");
    p->set<std::string>("Thickness Variable Name", "ice_thickness");

    p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);

    p->set<ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

    p->set<std::string>("Side Set Name", basalSideName);

    //Output
    p->set<std::string>("Temperature Variable Name", "temperature");
    p->set<std::string>("Corrected Temperature Variable Name", "corrected_temperature");
    p->set<std::string>("Basal dTdz Variable Name", "basal_dTdz");
    p->set<std::string>("Diff Enthalpy Variable Name", "Diff Enth");

    ev = Teuchos::rcp(new LandIce::Temperature<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    // Saving the temperature in the output mesh
    std::string stateName = "temperature";
    if(volumeFields.find(stateName) != volumeFields.end())
    {
      std::string fieldName = "temperature";
      entity = Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, "");
      p->set<std::string>("Field Name", fieldName);
      p->set("Field Layout", dl->node_scalar);
      p->set<bool>("Nodal State", true);

      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((fieldManagerChoice == Albany::BUILD_RESID_FM)&&(ev->evaluatedFields().size()>0))
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }

    {
      std::string stateName = "surface_enthalpy";
      entity = Albany::StateStruct::NodalDistParameter;
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, "");
      p->set<std::string>("Parameter Name", stateName);

      ev = rcp(new PHAL::ScatterScalarNodalParameter<EvalT,PHAL::AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((fieldManagerChoice == Albany::BUILD_RESID_FM)&&(ev->evaluatedFields().size()>0))
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }

    // Saving the diff enthalpy field in the output mesh
    stateName = "Diff Enth";
    if(volumeFields.find(stateName) != volumeFields.end())
    {
      std::string fieldName = "Diff Enth";
      entity = Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, "");
      p->set<std::string>("Field Name", fieldName);
      p->set("Field Layout", dl->node_scalar);
      p->set<bool>("Nodal State", true);

      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((fieldManagerChoice == Albany::BUILD_RESID_FM)&&(ev->evaluatedFields().size()>0))
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
  }

  // --- LandIce Liquid Water Fraction
  {
    p = rcp(new ParameterList("LandIce Liquid Water Fraction"));

    //Input
    p->set<std::string>("Enthalpy Hs Variable Name", "melting enthalpy");
    p->set<std::string>("Enthalpy Variable Name", "Enthalpy");

    p->set<ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Water Content Variable Name", "phi");
    ev = Teuchos::rcp(new LandIce::LiquidWaterFraction<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    std::string stateName = "phi";
    if(volumeFields.find(stateName) != volumeFields.end())
    { // Saving the water fraction phi in the output mesh
      std::string fieldName = "phi";
      entity = Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, "");
      p->set<std::string>("Field Name", fieldName);
      p->set("Field Layout", dl->node_scalar);
      p->set<bool>("Nodal State", true);

      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      if ((fieldManagerChoice == Albany::BUILD_RESID_FM)&&(ev->evaluatedFields().size()>0))
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
  }


  // --- LandIce Basal Melt Rate
  {
    p = rcp(new ParameterList("LandIce Basal Melt Rate"));

    //Input
    p->set<bool>("Nodal", false);
    p->set<std::string>("Water Content Side Variable Name", "phi");
    p->set<std::string>("Geothermal Flux Side Variable Name", "heat_flux");
    p->set<std::string>("Velocity Side Variable Name", "velocity");
    p->set<std::string>("Basal Friction Coefficient Side Variable Name", "basal_friction");
    p->set<std::string>("Enthalpy Hs Side Variable Name", "melting enthalpy");
    p->set<std::string>("Enthalpy Side Variable Name", "Enthalpy");
    p->set<std::string>("Basal dTdz Variable Name", "basal_dTdz");
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

    p->set<ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));
    p->set<Teuchos::ParameterList*>("LandIce Enthalpy", &params->sublist("LandIce Enthalpy",false));

    p->set<std::string>("Side Set Name", basalSideName);

    //Output
    p->set<std::string>("Basal Vertical Velocity Variable Name", "basal_vert_velocity");
    p->set<std::string>("Basal Melt Rate Variable Name", "basal_neumann_term");
    ev = Teuchos::rcp(new LandIce::BasalMeltRate<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT, typename EvalT::ParamScalarT>(*p,dl_basal));
    fm0.template registerEvaluator<EvalT>(ev);
  }


  p = Teuchos::rcp(new Teuchos::ParameterList("Glen's Law Homotopy Parameter"));

  std::string param_name = "Glen's Law Homotopy Parameter";
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>> ptr_homotopy;
  ptr_homotopy = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>(*p,dl));
  ptr_homotopy->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Viscosity").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_homotopy);

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    PHX::Tag<typename EvalT::ScalarT> res_tag0("Scatter Enthalpy", dl->dummy);
    fm0.requireField<EvalT>(res_tag0);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    LandIce::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
  }

  return Teuchos::null;
}

#endif /* LandIce_ENTHALPY_PROBLEM_HPP */
