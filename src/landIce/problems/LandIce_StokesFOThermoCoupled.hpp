//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_PROBLEMS_LANDICE_STOKESFOTHERMOCOUPLED_HPP_
#define LANDICE_PROBLEMS_LANDICE_STOKESFOTHERMOCOUPLED_HPP_

#include "Albany_GeneralPurposeFieldsNames.hpp"
#include "LandIce_StokesFOBase.hpp"
#include "LandIce_BasalMeltRate.hpp"
#include "LandIce_EnthalpyBasalResid.hpp"
#include "LandIce_EnthalpyResid.hpp"
#include "LandIce_LiquidWaterFraction.hpp"
#include "LandIce_PressureMeltingEnthalpy.hpp"
#include "LandIce_Temperature.hpp"
#include "LandIce_w_Resid.hpp"
#include "LandIce_SurfaceAirEnthalpy.hpp"

#include "PHAL_GatherCoordinateVector.hpp"

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

  //! Build unmanaged fields
  virtual void buildFields(PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  //! Each problem must generate it's list of valid parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  //! Main problem setup routine. Not directly called, but indirectly by following functions
  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  template <typename EvalT>
  void constructFields(PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

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

  bool adjustBedTopo;
  bool adjustSurfaceHeight;
  bool fluxDivIsPartOfSolution;
  bool l2ProjectedBoundaryEquation;

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
  bool require_old_coords=false;
  if (Albany::mesh_depends_on_parameters() && (is_dist_param[ice_thickness_name])) {
    require_old_coords=true;
    if(adjustBedTopo && !adjustSurfaceHeight) {
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
  }

  // If the mesh depends on parameters AND the bed topography or the surface height are parameters,
  // after gathering the coordinates, we modify the z coordinate of the mesh.
  else if (Albany::mesh_depends_on_parameters() && (is_dist_param[surface_height_param_name]||is_dist_param[bed_topography_param_name])) {
    //------ Update Z Coordinate
    require_old_coords=true;
    p = Teuchos::rcp(new Teuchos::ParameterList("Update Z Coordinate"));

    p->set<std::string>("Old Coords Name",  "Coord Vec Old");
    p->set<std::string>("New Coords Name",  Albany::coord_vec_name);
    p->set<std::string>("Thickness Name",   ice_thickness_name);
    p->set<std::string>("Top Surface Name", surface_height_name);
    if(is_dist_param[surface_height_param_name])
      p->set<std::string>("Top Surface Parameter Name", surface_height_param_name);
    if(is_dist_param[bed_topography_param_name])
      p->set<std::string>("Bed Topography Parameter Name", bed_topography_param_name);
    p->set<std::string>("Bed Topography Name", bed_topography_name);
    p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

    ev = Teuchos::rcp(new LandIce::UpdateZCoordinateGivenTopAndBedSurfaces<EvalT,PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if(require_old_coords) {
    //----- Gather Coordinate Vector (ad hoc parameters)
    p = Teuchos::rcp(new Teuchos::ParameterList("Gather Coordinate Vector"));

    // Output:: Coordinate Vector at vertices
    p->set<std::string>("Coordinate Vector Name", "Coord Vec Old");

    ev = Teuchos::rcp(new PHAL::GatherCoordinateVector<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  } else {
    //----- Gather Coordinate Vector (general parameters)
    ev = evalUtils.constructGatherCoordinateVectorEvaluator();
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // --- Vertical velocity equation evaluators --- //
  constructVerticalVelocityEvaluators<EvalT> (fm0, fieldManagerChoice, meshSpecs);

  // --- Enthalpy equation evaluators --- //
  constructEnthalpyEvaluators<EvalT> (fm0, fieldManagerChoice);

  // --- ProjectedLaplacian-related evaluators (if needed) --- //
  if(l2ProjectedBoundaryEquation) {
    int eqId = 3;
    constructProjLaplEvaluators<EvalT> (fm0, fieldManagerChoice, eqId);
  }

  // --- FluxDiv-related evaluators (if needed) --- //
  if(fluxDivIsPartOfSolution) {
    int eqId = l2ProjectedBoundaryEquation ? 4 : 3;
    constructFluxDivEvaluators<EvalT> (fm0, fieldManagerChoice, eqId, meshSpecs);
  }

  // --- StokesFOBase evaluators --- //
  constructStokesFOBaseEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice);

  // Finally, construct responses, and return the tags
  return constructStokesFOBaseResponsesEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice, responseList);
}

template <typename EvalT>
void StokesFOThermoCoupled::
constructVerticalVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                     Albany::FieldManagerChoice fieldManagerChoice,
                                     const Albany::MeshSpecsStruct&)
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

  ev = evalUtils.constructDOFGradInterpolationEvaluator (dof_names[1], dof_offsets[1]);
  fm0.template registerEvaluator<EvalT> (ev);

  // --- W Residual ---
  p = Teuchos::rcp(new Teuchos::ParameterList(resid_names[1]));

  //Input
  p->set<std::string>("Velocity QP Variable Name", dof_names[0]);
  p->set<std::string>("Weighted BF Variable Name", Albany::weighted_bf_name);
  p->set<std::string>("BF Side Name", Albany::bf_name + "_" + basalSideName);
  p->set<std::string>("Weighted Gradient BF Variable Name", Albany::weighted_grad_bf_name);
  p->set<std::string>("Weighted Measure Side Name", Albany::weighted_measure_name + "_" + basalSideName);
  p->set<std::string>("Side Normal Name", Albany::normal_name + "_" + basalSideName);
  p->set<std::string>("w Side QP Variable Name", dof_names[1] + "_" + basalSideName);
  p->set<std::string>("w Gradient QP Variable Name", dof_names[1] + " Gradient");
  p->set<std::string>("Basal Vertical Velocity Side QP Variable Name", "basal_vert_velocity_" + basalSideName);
  p->set<std::string>("Velocity Gradient QP Variable Name", dof_names[0] + " Gradient");
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
  p->set<std::string>("Coordinate Vector Name", Albany::coord_vec_name);

  //Output
  p->set<std::string>("Residual Variable Name", resid_names[1]);

  ev = Teuchos::rcp(new w_Resid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

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
  p->set<std::string>("Water Content Side Variable Name", water_content_name + "_" + basalSideName);
  p->set<std::string>("Geothermal Flux Side Variable Name", geothermal_flux_name + "_" + basalSideName);
  p->set<std::string>("Velocity Side Variable Name", dof_names[0] + "_" + basalSideName);
  p->set<std::string>("Basal Friction Coefficient Side Variable Name", "beta_" + basalSideName);
  p->set<std::string>("Enthalpy Hs Side Variable Name", melting_enthalpy_name + "_" + basalSideName);
  p->set<std::string>("Enthalpy Side Variable Name", dof_names[2] + "_" + basalSideName);
  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));
  p->set<Teuchos::ParameterList*>("LandIce Enthalpy", &params->sublist("LandIce Enthalpy", false));

  p->set<std::string>("Side Set Name", basalSideName);

  //Output
  p->set<std::string>("Basal Vertical Velocity Variable Name", "basal_vert_velocity_" + basalSideName);
  p->set<std::string>("Basal Melt Rate Variable Name", "basal_melt_rate_" + basalSideName);
  ev = Teuchos::rcp(new BasalMeltRate<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl->side_layouts[basalSideName]));
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

  ev = Teuchos::rcp(new LiquidWaterFraction<EvalT, PHAL::AlbanyTraits, typename EvalT::ScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- LandIce pressure-melting enthalpy
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Pressure Melting Enthalpy"));

  //Input
  p->set<std::string>("Surface Height Variable Name", "surface_height");
  p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");
  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

  //Output
  field_deps[melting_temperature_name].insert(hydrostatic_pressure_name);
  field_deps[melting_enthalpy_name].insert(hydrostatic_pressure_name);
  p->set<std::string>("Melting Temperature Variable Name", melting_temperature_name);
  p->set<std::string>("Enthalpy Hs Variable Name", melting_enthalpy_name);

  ev = Teuchos::rcp(new PressureMeltingEnthalpy<EvalT, PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- LandIce Surface Air Enthalpy
  {
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce surface Air Enthalpy"));

    //Input
    p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));
    p->set<std::string>("Surface Air Temperature Name", "surface_air_temperature");

    //Output
    p->set<std::string>("Surface Air Enthalpy Name", "surface_enthalpy");
    ev = createEvaluatorWithOneScalarType<SurfaceAirEnthalpy,EvalT>(p,dl,get_scalar_type("surface_air_temperature"));

    fm0.template registerEvaluator<EvalT>(ev);
  }

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

  ev = Teuchos::rcp(new LandIce::Temperature<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

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
  p->set<std::string>("Vertical Velocity QP Variable Name", dof_names[1]);
  p->set<std::string>("Melting Temperature Gradient QP Variable Name",melting_temperature_name + " Gradient");
  p->set<std::string>("Enthalpy Basal Residual Variable Name", "Enthalpy Basal Residual");

  if(needsDiss) {
    p->set<std::string>("Dissipation QP Variable Name", "LandIce Dissipation");
  }

  p->set<std::string>("Water Content QP Variable Name",water_content_name);
  p->set<std::string>("Water Content Gradient QP Variable Name",water_content_name + " Gradient");
  p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");
  p->set<bool>("Needs Dissipation", needsDiss);
  p->set<bool>("Needs Basal Friction", needsBasFric);
  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));
  p->set<Teuchos::ParameterList*>("LandIce Enthalpy Regularization", &params->sublist("LandIce Enthalpy", false).sublist("Regularization", false));
  if(params->isSublist("LandIce Enthalpy") &&  params->sublist("LandIce Enthalpy").isParameter("Stabilization")) {
    p->set<Teuchos::ParameterList*>("LandIce Enthalpy Stabilization", &params->sublist("LandIce Enthalpy").sublist("Stabilization"));
  }

  //Output
  p->set<std::string>("Residual Variable Name", resid_names[2]);

  ev = Teuchos::rcp(new  EnthalpyResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- Enthalpy Basal Residual ---
  p = Teuchos::rcp(new Teuchos::ParameterList("Enthalpy Basal Resid"));

  //Input
  p->set<std::string>("BF Side Name", Albany::bf_name + "_"+basalSideName);
  p->set<std::string>("Weighted Measure Side Name", Albany::weighted_measure_name + "_"+basalSideName);
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
  p->set<std::string>("Basal Melt Rate Side QP Variable Name", "basal_melt_rate_" + basalSideName);

  //Output
  p->set<std::string>("Enthalpy Basal Residual Variable Name", "Enthalpy Basal Residual");

  ev = Teuchos::rcp(new EnthalpyBasalResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);
}

template <typename EvalT>
void
LandIce::StokesFOThermoCoupled::constructFields(PHX::FieldManager<PHAL::AlbanyTraits> &fm0)
{
  constructStokesFOBaseFields<EvalT>(fm0);
}

} // namespace LandIce

#endif /* LANDICE_PROBLEMS_LANDICE_STOKESFOTHERMOCOUPLED_HPP_ */
