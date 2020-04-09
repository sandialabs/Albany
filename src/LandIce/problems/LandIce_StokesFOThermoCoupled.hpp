//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_FO_THERMO_COUPLED_PROBLEM_HPP
#define LANDICE_STOKES_FO_THERMO_COUPLED_PROBLEM_HPP

#include "LandIce_StokesFOBase.hpp"

#include "LandIce_BasalFrictionHeat.hpp"
#include "LandIce_BasalMeltRate.hpp"
#include "LandIce_EnthalpyBasalResid.hpp"
#include "LandIce_EnthalpyResid.hpp"
#include "LandIce_GeoFluxHeat.hpp"
#include "LandIce_HydrostaticPressure.hpp"
#include "LandIce_Integral1Dw_Z.hpp"
#include "LandIce_LiquidWaterFraction.hpp"
#include "LandIce_PressureMeltingEnthalpy.hpp"
#include "LandIce_Temperature.hpp"
#include "LandIce_w_Resid.hpp"
#include "LandIce_w_ZResid.hpp"
#include "LandIce_SurfaceAirEnthalpy.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce
{

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
 */
class StokesFOThermoCoupled : public StokesFOBase
{
public:

  //! Default constructor
  StokesFOThermoCoupled (const Teuchos::RCP<Teuchos::ParameterList>& params,
                         const Teuchos::RCP<Teuchos::ParameterList>& discParams,
                         const Teuchos::RCP<ParamLib>& paramLib,
                         const int numDim_);

  //! Destructor
  ~StokesFOThermoCoupled() = default;

  // Build evaluators
  virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
  buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                   const Albany::MeshSpecsStruct& meshSpecs,
                   Albany::StateManager& stateMgr,
                   Albany::FieldManagerChoice fmchoice,
                   const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Each problem must generate it's list of valide parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

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

  template<typename EvalT>
  void constructEnthalpyEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                    Albany::FieldManagerChoice fieldManagerChoice);

  template<typename EvalT>
  void constructVerticalVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0, 
                                            Albany::FieldManagerChoice fieldManagerChoice,
                                            const Albany::MeshSpecsStruct& meshSpecs);

  void setupEvaluatorRequests ();
  void setFieldsProperties ();

  bool needsDiss;
  bool needsBasFric;
  bool isGeoFluxConst;
  bool compute_w;

  bool adjustBedTopo;
  bool adjustSurfaceHeight;

  std::string hydrostatic_pressure_name;
  std::string melting_enthalpy_name;
  std::string melting_temperature_name;
  std::string surface_enthalpy_name;
  std::string water_content_name;
  std::string geothermal_flux_name;
};

// ================================ IMPLEMENTATION ============================ //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
StokesFOThermoCoupled::
constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                     const Albany::MeshSpecsStruct& meshSpecs,
                     Albany::StateManager& stateMgr,
                     Albany::FieldManagerChoice fieldManagerChoice,
                     const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // --- StokesFOBase evaluators --- //
  constructStokesFOBaseEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice);

  // Gather velocity field
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names[0], dof_offsets[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter velocity residual
  ev = evalUtils.constructScatterResidualEvaluatorWithExtrudedParams(true, resid_names[0], Teuchos::rcpFromRef(extruded_params_levels), dof_offsets[0], scatter_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);
  if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> res_tag(scatter_names[0], dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }

  // Geometry
  // If the mesh depends on parameters AND the thickness is a parameter,
  // after gathering the coordinates, we modify the z coordinate of the mesh.
  if (Albany::mesh_depends_on_parameters() && is_dist_param[ice_thickness_name]) {
    if(adjustBedTopo && !adjustSurfaceHeight) {
      //----- Gather Coordinate Vector (ad hoc parameters)
      p = Teuchos::rcp(new Teuchos::ParameterList("Gather Coordinate Vector"));

      // Output:: Coordindate Vector at vertices
      p->set<std::string>("Coordinate Vector Name", "Coord Vec Old");

      ev = Teuchos::rcp(new PHAL::GatherCoordinateVector<EvalT,PHAL::AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);

      //------ Update Z Coordinate
      p = Teuchos::rcp(new Teuchos::ParameterList("Update Z Coordinate"));
      p->set<std::string>("Old Coords Name",  "Coord Vec Old");
      p->set<std::string>("New Coords Name",  Albany::coord_vec_name);
      p->set<std::string>("Thickness Name",   ice_thickness_name);
      p->set<std::string>("Thickness Lower Bound Name",   ice_thickness_name + "_lowerbound");
      p->set<std::string>("Thickness Upper Bound Name",   ice_thickness_name + "_upperbound");
      p->set<std::string>("Top Surface Name", "observed_surface_height");
      p->set<std::string>("Updated Top Surface Name", surface_height_name);
      p->set<std::string>("Bed Topography Name", "observed_bed_topography");
      p->set<std::string>("Updated Bed Topography Name", bed_topography_name);
      p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

      ev = Teuchos::rcp(new LandIce::UpdateZCoordinateMovingBed<EvalT,PHAL::AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    } else if(adjustSurfaceHeight && !adjustBedTopo) {
      //----- Gather Coordinate Vector (ad hoc parameters)
      p = Teuchos::rcp(new Teuchos::ParameterList("Gather Coordinate Vector"));

      // Output:: Coordindate Vector at vertices
      p->set<std::string>("Coordinate Vector Name", "Coord Vec Old");

      ev = Teuchos::rcp(new PHAL::GatherCoordinateVector<EvalT,PHAL::AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);

      //------ Update Z Coordinate
      p = Teuchos::rcp(new Teuchos::ParameterList("Update Z Coordinate"));
      p->set<std::string>("Old Coords Name",  "Coord Vec Old");
      p->set<std::string>("New Coords Name",  Albany::coord_vec_name);
      p->set<std::string>("Thickness Name",   ice_thickness_name);
      p->set<std::string>("Top Surface Name", surface_height_name);
      p->set<std::string>("Bed Topography Name", bed_topography_name);
      p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

      ev = Teuchos::rcp(new LandIce::UpdateZCoordinateMovingTop<EvalT,PHAL::AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    } else {
          TEUCHOS_TEST_FOR_EXCEPTION(adjustBedTopo == adjustSurfaceHeight, std::logic_error, "Error! When the ice thickness is a parameter,\n "
              "either 'Adjust Bed Topography to Account for Thickness Changes' or\n"
              " 'Adjust Surface Height to Account for Thickness Changes' needs to be true.\n");
    }
  } else {
    //----- Gather Coordinate Vector (general parameters)
    ev = evalUtils.constructGatherCoordinateVectorEvaluator();
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // --- Vertical velocity equation evaluators --- //
  constructVerticalVelocityEvaluators<EvalT> (fm0, fieldManagerChoice, meshSpecs);

  // --- Enthalpy equation evaluators --- //
  constructEnthalpyEvaluators<EvalT> (fm0, fieldManagerChoice);

  // Finally, construct responses, and return the tags
  return constructStokesFOBaseResponsesEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice, responseList);
}

template <typename EvalT>
void StokesFOThermoCoupled::
constructVerticalVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                     Albany::FieldManagerChoice fieldManagerChoice,
                                     const Albany::MeshSpecsStruct& meshSpecs)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // Gather solution
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names[1], dof_offsets[1]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter residual
  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names[1], dof_offsets[1], scatter_names[1]);
  fm0.template registerEvaluator<EvalT> (ev);
  if (fieldManagerChoice == Albany::BUILD_RESID_FM) {

    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> res_tag(scatter_names[1], dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }

  if(!compute_w) {
    // Compute W integrating W_z
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Integral 1D W_z"));

    // Input
    p->set<std::string>("Basal Vertical Velocity Variable Name", "basal_vert_velocity");
    p->set<std::string>("Thickness Variable Name", ice_thickness_name);
    p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));
    p->set<bool>("Stokes and Thermo coupled", true);

    // Output
    p->set<std::string>("Integral1D w_z Variable Name", "W");

    ev = createEvaluatorWithOneScalarType<LandIce::Integral1Dw_Z,EvalT>(p,dl,field_scalar_type[ice_thickness_name]);
    fm0.template registerEvaluator<EvalT>(ev);

    // --- W_z Residual ---
    p = Teuchos::rcp(new Teuchos::ParameterList(resid_names[1]));

    //Input
    p->set<std::string>("Weighted BF Variable Name", Albany::weighted_bf_name);
    p->set<std::string>("w_z QP Variable Name", "W_z");
    p->set<std::string>("Velocity Gradient QP Variable Name", dof_names[0] + " Gradient");

    //Output
    p->set<std::string>("Residual Variable Name", resid_names[1]);

    ev = Teuchos::rcp(new LandIce::w_ZResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  } else {


    ev = evalUtils.constructDOFGradInterpolationEvaluator ("W", dof_offsets[1]);
    fm0.template registerEvaluator<EvalT> (ev);

    // --- W Residual ---
    p = Teuchos::rcp(new Teuchos::ParameterList(resid_names[1]));

    //Input
    p->set<std::string>("Velocity QP Variable Name", dof_names[0]);
    p->set<std::string>("Weighted BF Variable Name", Albany::weighted_bf_name);
    p->set<std::string>("BF Side Name", Albany::bf_name + " "+basalSideName);
    p->set<std::string>("Weighted Gradient BF Variable Name", Albany::weighted_grad_bf_name);
    p->set<std::string>("Weighted Measure Side Name", Albany::weighted_measure_name + " "+basalSideName);
    p->set<std::string>("Side Normal Name", Albany::normal_name + " " + basalSideName);
    p->set<std::string>("w Side QP Variable Name", "W_" + basalSideName);
    p->set<std::string>("w Gradient QP Variable Name", "W Gradient");
    p->set<std::string>("Basal Vertical Velocity Side QP Variable Name", "basal_vert_velocity_" + basalSideName);
    p->set<std::string>("Velocity Gradient QP Variable Name", dof_names[0] + " Gradient");
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<std::string>("Coordinate Vector Name", Albany::coord_vec_name);

    //Output
    p->set<std::string>("Residual Variable Name", resid_names[1]);

    ev = Teuchos::rcp(new LandIce::w_Resid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
}

template <typename EvalT>
void StokesFOThermoCoupled::
constructEnthalpyEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                             Albany::FieldManagerChoice fieldManagerChoice)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // Gather solution
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names[2], dof_offsets[2]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter residual
  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names[2], dof_offsets[2], scatter_names[2]);
  fm0.template registerEvaluator<EvalT> (ev);
  if (fieldManagerChoice == Albany::BUILD_RESID_FM) {

    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> res_tag(scatter_names[2], dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }

  // --- LandIce Basal Melt Rate
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Basal Melt Rate"));

  //Input
  p->set<bool>("Nodal", !compute_w);
  p->set<std::string>("Water Content Side Variable Name", water_content_name + "_" + basalSideName);
  p->set<std::string>("Geothermal Flux Side Variable Name", geothermal_flux_name + "_" + basalSideName);
  p->set<std::string>("Velocity Side Variable Name", dof_names[0] + "_" + basalSideName);
  p->set<std::string>("Basal Friction Coefficient Side Variable Name", "beta_" + basalSideName);
  p->set<std::string>("Enthalpy Hs Side Variable Name", melting_enthalpy_name + "_" + basalSideName);
  p->set<std::string>("Enthalpy Side Variable Name", dof_names[2] + "_" + basalSideName);
  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));
  p->set<Teuchos::ParameterList*>("LandIce Enthalpy", &params->sublist("LandIce Enthalpy", false));

  p->set<std::string>("Side Set Name", basalSideName);

  //Output
  p->set<std::string>("Basal Vertical Velocity Variable Name", "basal_vert_velocity_" + basalSideName);
  p->set<std::string>("Basal Melt Rate Variable Name", "basal_melt_rate_" + basalSideName);
  ev = createEvaluatorWithTwoScalarTypes<LandIce::BasalMeltRate,EvalT>(p,dl->side_layouts[basalSideName],
                                                                         FieldScalarType::Scalar,
                                                                         field_scalar_type[melting_enthalpy_name]);
  fm0.template registerEvaluator<EvalT>(ev);

  // --- LandIce Liquid Water Fraction
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Liquid Water Fraction"));

  //Input
  p->set<std::string>("Enthalpy Hs Variable Name", melting_enthalpy_name);
  p->set<std::string>("Enthalpy Variable Name", dof_names[2]);

  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

  //Output
  p->set<std::string>("Water Content Variable Name", water_content_name);

  ev = createEvaluatorWithOneScalarType<LandIce::LiquidWaterFraction,EvalT>(p,dl,field_scalar_type[melting_enthalpy_name]);
  fm0.template registerEvaluator<EvalT>(ev);

  // --- LandIce pressure-melting enthalpy
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Pressure Melting Enthalpy"));

  //Input
  p->set<std::string>("Hydrostatic Pressure Variable Name", hydrostatic_pressure_name);
  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

  //Output
  p->set<std::string>("Melting Temperature Variable Name", melting_temperature_name);
  p->set<std::string>("Enthalpy Hs Variable Name", melting_enthalpy_name);

  ev = createEvaluatorWithOneScalarType<LandIce::PressureMeltingEnthalpy,EvalT>(p,dl,field_scalar_type[melting_temperature_name]);
  fm0.template registerEvaluator<EvalT>(ev);

  // --- LandIce Surface Air Enthalpy
  {
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce surface Air Enthalpy"));

    //Input
    p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));
    p->set<std::string>("Surface Air Temperature Name", "surface_air_temperature");

    //Output
    p->set<std::string>("Surface Air Enthalpy Name", "surface_enthalpy");
    ev = createEvaluatorWithOneScalarType<LandIce::SurfaceAirEnthalpy,EvalT>(p,dl,field_scalar_type["surface_air_temperature"]);

    fm0.template registerEvaluator<EvalT>(ev);
  }

  // --- LandIce hydrostatic pressure
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Hydrostatic Pressure"));

  //Input
  p->set<std::string>("Surface Height Variable Name", surface_height_name);
  p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name);
  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

  //Output
  p->set<std::string>("Hydrostatic Pressure Variable Name", hydrostatic_pressure_name);

  ev = createEvaluatorWithOneScalarType<LandIce::HydrostaticPressure,EvalT>(p,dl,field_scalar_type[surface_height_name]);
  fm0.template registerEvaluator<EvalT>(ev);

/*
 *   // --- LandIce Geothermal flux heat
 *   p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Geothermal Flux Heat"));
 * 
 *   //Input
 *   p->set<std::string>("BF Side Name", Albany::bf_name + " "+basalSideName);
 *   p->set<std::string>("Gradient BF Side Name", Albany::grad_bf_name + " "+basalSideName);
 *   p->set<std::string>("Weighted Measure Name", Albany::weighted_measure_name + " "+basalSideName);
 *   p->set<std::string>("Velocity Side QP Variable Name", dof_names[0] + "_" + basalSideName);
 *   p->set<std::string>("Vertical Velocity Side QP Variable Name", "W");
 *   p->set<std::string>("Side Set Name", basalSideName);
 *   p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
 *   if(params->isSublist("LandIce Enthalpy Stabilization")) {
 *     p->set<Teuchos::ParameterList*>("LandIce Enthalpy Stabilization", &params->sublist("LandIce Enthalpy Stabilization"));
 *   }
 *   if(!isGeoFluxConst) {
 *     p->set<std::string>("Geothermal Flux Side QP Variable Name", geothermal_flux_name + "_" + basalSideName);
 *   } else {
 *     p->set<double>("Uniform Geothermal Flux Heat Value", params->sublist("LandIce Physical Parameters",false).get<double>("Uniform Geothermal Flux Heat Value"));
 *   }
 *   p->set<bool>("Constant Geothermal Flux", isGeoFluxConst);
 * 
 *   //Output
 *   p->set<std::string>("Geothermal Flux Heat Variable Name", "Geo Flux Heat");
 *   p->set<std::string>("Geothermal Flux Heat SUPG Variable Name", "Geo Flux Heat SUPG");
 * 
 *   ev = Teuchos::rcp(new LandIce::GeoFluxHeat<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
 *   fm0.template registerEvaluator<EvalT>(ev);
 */

  // --- LandIce Temperature: diff enthalpy is h - hs.
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Temperature"));

  //Input
  p->set<std::string>("Melting Temperature Variable Name", melting_temperature_name);
  p->set<std::string>("Enthalpy Hs Variable Name", melting_enthalpy_name);
  p->set<std::string>("Enthalpy Variable Name", dof_names[2]);
  p->set<std::string>("Thickness Variable Name", ice_thickness_name);
  p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));
  p->set<std::string>("Side Set Name", basalSideName);

  //Output
  p->set<std::string>("Temperature Variable Name", temperature_name);
  p->set<std::string>("Corrected Temperature Variable Name", corrected_temperature_name);
  // p->set<std::string>("Basal dTdz Variable Name", "basal_dTdz");
  p->set<std::string>("Diff Enthalpy Variable Name", "Diff Enth");

  ev = createEvaluatorWithOneScalarType<LandIce::Temperature,EvalT>(p,dl,field_scalar_type[melting_temperature_name]);
  fm0.template registerEvaluator<EvalT>(ev);

  // --- LandIce Basal friction heat ---
  if(needsBasFric)
  {
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Basal Friction Heat"));

    //Input
    p->set<std::string>("BF Side Name", Albany::bf_name + " "+basalSideName);
    p->set<std::string>("Gradient BF Side Name", Albany::grad_bf_name + " "+basalSideName);
    p->set<std::string>("Weighted Measure Name", Albany::weighted_measure_name + " "+basalSideName);
    p->set<std::string>("Velocity Side QP Variable Name", dof_names[0] + "_" + basalSideName);
    p->set<std::string>("Basal Friction Coefficient Side QP Variable Name", "beta");
    p->set<std::string>("Side Set Name", basalSideName);

    if(params->isSublist("LandIce Enthalpy") &&  params->sublist("LandIce Enthalpy").isParameter("Stabilization")) {
      p->set<Teuchos::ParameterList*>("LandIce Enthalpy Stabilization", &params->sublist("LandIce Enthalpy").sublist("Stabilization"));
    }

    p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

    p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);

    //Output
    p->set<std::string>("Basal Friction Heat Variable Name", "Basal Heat");
    p->set<std::string>("Basal Friction Heat SUPG Variable Name", "Basal Heat SUPG");

    ev = Teuchos::rcp(new LandIce::BasalFrictionHeat<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // --- Enthalpy Residual ---
  p = Teuchos::rcp(new Teuchos::ParameterList("Enthalpy Resid"));

  //Input
  p->set<std::string>("Weighted BF Variable Name", Albany::weighted_bf_name);
  p->set<std::string>("Weighted Gradient BF Variable Name", Albany::weighted_grad_bf_name);
  p->set<std::string>("Coordinate Vector Name", Albany::coord_vec_name);
  p->set<std::string>("Enthalpy QP Variable Name", dof_names[2]);
  p->set<std::string>("Enthalpy Gradient QP Variable Name", dof_names[2] + " Gradient");
  p->set<std::string>("Enthalpy Hs QP Variable Name", melting_enthalpy_name);
  p->set<std::string>("Diff Enthalpy Variable Name", "Diff Enth");
  p->set<std::string>("Velocity QP Variable Name", dof_names[0]);
  p->set<std::string>("Velocity Gradient QP Variable Name", dof_names[0] + " Gradient");
  p->set<std::string>("Vertical Velocity QP Variable Name", "W");
  p->set<std::string>("Geothermal Flux Heat QP Variable Name","Geo Flux Heat");
  p->set<std::string>("Geothermal Flux Heat QP SUPG Variable Name","Geo Flux Heat SUPG");
  p->set<std::string>("Melting Temperature Gradient QP Variable Name",melting_temperature_name + " Gradient");
  p->set<std::string>("Enthalpy Basal Residual Variable Name", "Enthalpy Basal Residual");
  p->set<std::string>("Enthalpy Basal Residual SUPG Variable Name", "Enthalpy Basal Residual SUPG");

  if(needsDiss) {
    p->set<std::string>("Dissipation QP Variable Name", "LandIce Dissipation");
  }
  if(needsBasFric) {
    p->set<std::string>("Basal Friction Heat QP Variable Name", "Basal Heat");
    p->set<std::string>("Basal Friction Heat QP SUPG Variable Name", "Basal Heat SUPG");
  }
  p->set<std::string>("Water Content QP Variable Name",water_content_name);
  p->set<std::string>("Water Content Gradient QP Variable Name",water_content_name + " Gradient");
  p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");
  p->set<bool>("Needs Dissipation", needsDiss);
  p->set<bool>("Needs Basal Friction", needsBasFric);
  p->set<bool>("Constant Geothermal Flux", isGeoFluxConst);
  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));
  p->set<Teuchos::ParameterList*>("LandIce Enthalpy Regularization", &params->sublist("LandIce Enthalpy", false).sublist("Regularization", false));
  if(params->isSublist("LandIce Enthalpy") &&  params->sublist("LandIce Enthalpy").isParameter("Stabilization")) {
    p->set<Teuchos::ParameterList*>("LandIce Enthalpy Stabilization", &params->sublist("LandIce Enthalpy").sublist("Stabilization"));
  }

  //Output
  p->set<std::string>("Residual Variable Name", resid_names[2]);

  ev = createEvaluatorWithTwoScalarTypes<LandIce::EnthalpyResid,EvalT>(p,dl,FieldScalarType::Scalar,field_scalar_type[melting_temperature_name]);
  fm0.template registerEvaluator<EvalT>(ev);

  // --- Enthalpy Basal Residual ---
  p = Teuchos::rcp(new Teuchos::ParameterList("Enthalpy Basal Resid"));

  //Input
  p->set<std::string>("BF Side Name", Albany::bf_name + " "+basalSideName);
  p->set<std::string>("Weighted Measure Side Name", Albany::weighted_measure_name + " "+basalSideName);
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
  p->set<std::string>("Basal Melt Rate Side QP Variable Name", "basal_melt_rate_" + basalSideName);

  //Output
  p->set<std::string>("Enthalpy Basal Residual Variable Name", "Enthalpy Basal Residual");

  ev = Teuchos::rcp(new LandIce::EnthalpyBasalResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);
}

} // namespace LandIce

#endif // LANDICE_STOKES_FO_THERMO_COUPLED_PROBLEM_HPP
