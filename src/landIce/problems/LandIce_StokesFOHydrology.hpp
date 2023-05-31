//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_FO_HYDROLOGY_PROBLEM_HPP
#define LANDICE_STOKES_FO_HYDROLOGY_PROBLEM_HPP 1

#include "LandIce_StokesFOBase.hpp"

#include "LandIce_HydraulicPotential.hpp"
#include "LandIce_HydrologyWaterThickness.hpp"
#include "LandIce_HydrologyWaterDischarge.hpp"
#include "LandIce_HydrologyMeltingRate.hpp"
#include "LandIce_HydrologyResidualMassEqn.hpp"
#include "LandIce_HydrologyResidualCavitiesEqn.hpp"
#include "LandIce_HydrologyResidualTillStorageEqn.hpp"
#include "LandIce_HydrologyBasalGravitationalWaterPotential.hpp"
#include "LandIce_SimpleOperationEvaluator.hpp"

namespace LandIce
{

/*
 * \brief The coupled problem StokesFO+Hydrology
 * 
 *   This problem is the union of StokesFO and Hydrology problems.
 *   You can look at StokesFO and StokesFOBase for more details on
 *   the ice stokes problem and Hydrology for details on the
 *   subglacial hydrology problem.
 *   This problem will solve the ice equation in 3d, while the
 *   hydrology problem is solved on the basal sideset of the mesh.
 */

class StokesFOHydrology : public StokesFOBase {
public:

  //! Default constructor
  StokesFOHydrology (const Teuchos::RCP<Teuchos::ParameterList>& params,
                     const Teuchos::RCP<Teuchos::ParameterList>& discParams,
                     const Teuchos::RCP<ParamLib>& paramLib,
                     const int numDim_);

  //! Destructor
  ~StokesFOHydrology() = default;

  // Build evaluators
  virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
  buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                   const Albany::MeshSpecs& meshSpecs,
                   Albany::StateManager& stateMgr,
                   Albany::FieldManagerChoice fmchoice,
                   const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Build unmanaged fields
  virtual void buildFields(PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  //! Each problem must generate it's list of valid parameters
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
                       const Albany::MeshSpecs& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  template <typename EvalT>
  void constructFields(PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  void constructDirichletEvaluators(const Albany::MeshSpecs& meshSpecs);
  void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecs>& meshSpecs);

protected:

  template <typename EvalT>
  void constructHydrologyEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0); 

  void setFieldsProperties ();
  void setupEvaluatorRequests ();

  // Flags to determine the kind of hydro problem
  bool eliminate_h;   // Solve for h in the cavity eqn, and eliminate the eqn (only for steady)
  bool unsteady;      // Whether the hydrology is steady or unsteady.
  bool has_h_till;    // Adds a third eqn for the water thickness in the till storage
  bool has_p_dot;     // Include C*d(p_w)/dt in the cavity equation (englacial porosity term)

  int hydro_neq;        // Number of hydrology equations
  int hydro_ndofs;      // Number of hydrology dofs
  int hydro_ndofs_dot;  // Number of hydrology dofs under time derivative
  int stokes_ndofs;     // Number of stokes dofs (should be 1)
  int stokes_neq;       // Number of stokes equations (should be 2)

  int stokes_dof_offset;      // Offset to first stokes dof
  int hydro_dof_offset;       // Offset to first hydro dof
  int hydro_dof_dot_offset;   // Offset to first hydro dof dot

  // Diagnostic dofs do not appear under time derivative. The most common hydro case
  // only has h (water thickness) under time derivative, and not p (the water pressure).
  // Some pb config, however, may have p_dot.
  Teuchos::ArrayRCP<std::string> hydro_dofs_dot_names;

  Teuchos::ArrayRCP<std::string> hydro_dofs_names;
  Teuchos::ArrayRCP<std::string> hydro_resids_names;
  Teuchos::ArrayRCP<std::string> stokes_dofs_names;
  Teuchos::ArrayRCP<std::string> stokes_resids_names;

  std::string grad_fname(const std::string& name) const {
    return name + "_gradient";
  }

  std::string water_pressure_name;
  std::string water_thickness_name;
  std::string till_water_storage_name;
  std::string water_pressure_dot_name;
  std::string water_thickness_dot_name;
  std::string till_water_storage_dot_name;

  std::string hydropotential_name;
  std::string ice_overburden_name;
  std::string beta_name;
  std::string melting_rate_name;
  std::string surface_water_input_name;
  std::string surface_mass_balance_name;
  std::string geothermal_flux_name;
  std::string water_discharge_name;
  std::string sliding_velocity_name;
  std::string grav_hydropotential_name;
};

// ================================ IMPLEMENTATION ============================ //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
StokesFOHydrology::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                               const Albany::MeshSpecs& meshSpecs,
                                               Albany::StateManager& stateMgr,
                                               Albany::FieldManagerChoice fieldManagerChoice,
                                               const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  const auto eval_name = PHX::print<EvalT>();

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // --- Hydrology evaluators --- //
  constructHydrologyEvaluators<EvalT> (fm0);

  // --- Gather dofs --- //

  // Gather ice velocity field
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(true, stokes_dofs_names, stokes_dof_offset);
  fm0.template registerEvaluator<EvalT> (ev);

  // Gather hydro dofs
  for (int i=0; i<hydro_dofs_names.size(); ++i) {
    ev = evalUtils.constructGatherSolutionSideEvaluator (hydro_dofs_names[i], basalSideName, cellType, hydro_dof_offset + i);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  for (int i=0; i<hydro_dofs_dot_names.size(); ++i) {
    // Gather prognostic hydro dofs
    ev = evalUtils.constructGatherSolutionSideEvaluator (hydro_dofs_dot_names[i],basalSideName, cellType, hydro_dof_dot_offset + i);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // --- Gather Coordinate Vector --- //
  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);

  // Finally, construct responses, and return the tags
  auto tag = constructStokesFOBaseResponsesEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice, responseList);

  // --- StokesFOBase evaluators --- //
  // Note: we do these last, so that if an evaluator for some field was already created by
  //       this class (or FOBase's response evaluator creation), FOBase can skip it,
  //       by calling is_available(...).
  constructStokesFOBaseEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice);

  // --- Scatter residuals ---- //
  ev = evalUtils.constructScatterResidualEvaluator(true, stokes_resids_names, stokes_dof_offset, scatter_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructScatterSideEqnResidualEvaluator(basalSideName,false,hydro_resids_names,hydro_dof_offset,scatter_names[1]);
  fm0.template registerEvaluator<EvalT> (ev);

  return tag;
}

template <typename EvalT>
void StokesFOHydrology::
constructHydrologyEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Teuchos::RCP<Teuchos::ParameterList> p;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;

  auto dl_side = dl->side_layouts.at(basalSideName);
  auto& hy_pl = params->sublist("LandIce Hydrology");
  auto& phys_pl = params->sublist("LandIce Physical Parameters");
  auto& visc_pl = params->sublist("LandIce Viscosity");

  // ================== Residual(s) ===================== //

  // ------- Hydrology Residual Mass Eqn-------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Residual Mass Eqn"));

  //Input
  p->set<std::string> ("BF Name", basal_fname(Albany::bf_name));
  p->set<std::string> ("Gradient BF Name", basal_fname(Albany::grad_bf_name));
  p->set<std::string> ("Weighted Measure Name", basal_fname(Albany::weighted_measure_name));
  p->set<std::string> ("Metric Name", basal_fname(Albany::metric_name));
  p->set<std::string> ("Water Discharge Variable Name", basal_fname(water_discharge_name));
  p->set<std::string> ("Till Water Storage Dot Variable Name", basal_fname(till_water_storage_dot_name));
  p->set<std::string> ("Water Thickness Dot Variable Name", basal_fname(water_thickness_dot_name));
  p->set<std::string> ("Melting Rate Variable Name",basal_fname(melting_rate_name));
  p->set<std::string> ("Surface Water Input Variable Name",basal_fname(surface_water_input_name));
  p->set<bool>("Unsteady",unsteady);
  p->set<bool>("Has Till Storage",has_h_till);
  p->set<std::string> ("Side Set Name",basalSideName);

  p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&phys_pl);
  p->set<Teuchos::ParameterList*> ("LandIce Hydrology Parameters",&hy_pl);

  //Output
  p->set<std::string> ("Mass Eqn Residual Name",hydro_resids_names[0]);

  ev = Teuchos::rcp(new LandIce::HydrologyResidualMassEqn<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  if (eliminate_h) {
    // -------- Hydrology Water Thickness (QPs) ------- //
    p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Water Thickness"));

    //Input
    p->set<std::string> ("Effective Pressure Variable Name",basal_fname(effective_pressure_name));
    p->set<std::string> ("Melting Rate Variable Name",basal_fname(melting_rate_name));
    p->set<std::string> ("Sliding Velocity Variable Name",basal_fname(sliding_velocity_name));
    p->set<std::string> ("Ice Softness Variable Name",basal_fname(flow_factor_name));
    p->set<bool> ("Nodal", false);
    p->set<Teuchos::ParameterList*> ("LandIce Hydrology Parameters",&hy_pl);
    p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&phys_pl);
    p->set<std::string> ("Side Set Name",basalSideName);

    //Output
    p->set<std::string> ("Water Thickness Variable Name", basal_fname(water_thickness_name));

    ev = Teuchos::rcp(new LandIce::HydrologyWaterThickness<EvalT,PHAL::AlbanyTraits,true,false>(*p,dl_side));
    fm0.template registerEvaluator<EvalT>(ev);

    // -------- Hydrology Water Thickness (nodes) ------- //
    p->set<bool> ("Nodal", true);

    ev = Teuchos::rcp(new LandIce::HydrologyWaterThickness<EvalT,PHAL::AlbanyTraits,true,false>(*p,dl_side));
    fm0.template registerEvaluator<EvalT>(ev);
  } else {
    // ------- Hydrology Cavities Equation Residual -------- //
    p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Residual Cavities Eqn"));

    //Input
    p->set<std::string> ("BF Name", Albany::bf_name);
    p->set<std::string> ("Weighted Measure Name", Albany::weights_name);
    p->set<std::string> ("Water Thickness Variable Name",basal_fname(water_thickness_name));
    p->set<std::string> ("Water Thickness Dot Variable Name",basal_fname(water_thickness_dot_name));
    p->set<std::string> ("Water Pressure Dot Variable Name",basal_fname(water_pressure_dot_name));
    p->set<std::string> ("Effective Pressure Variable Name",basal_fname(effective_pressure_name));
    p->set<std::string> ("Melting Rate Variable Name",basal_fname(melting_rate_name));
    p->set<std::string> ("Sliding Velocity Variable Name",basal_fname(sliding_velocity_name));
    p->set<std::string> ("Ice Softness Variable Name",basal_fname(flow_factor_name));
    p->set<bool> ("Unsteady", unsteady);
    p->set<Teuchos::ParameterList*> ("LandIce Hydrology Parameters",&hy_pl);
    p->set<Teuchos::ParameterList*> ("LandIce Viscosity Parameters",&visc_pl);
    p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&phys_pl);
    p->set<std::string> ("Side Set Name",basalSideName);

    //Output
    p->set<std::string> ("Cavities Eqn Residual Name", hydro_resids_names[1]);

    ev = Teuchos::rcp(new LandIce::HydrologyResidualCavitiesEqn<EvalT,PHAL::AlbanyTraits,true,false>(*p,dl_side));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (has_h_till) {
    // ------- Hydrology Till Water Storage Residual -------- //
    p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Till Water Storage Residual"));

    //Input
    p->set<std::string> ("BF Name", Albany::bf_name);
    p->set<std::string> ("Weighted Measure Name", Albany::weights_name);
    p->set<std::string> ("Till Water Storage Dot Variable Name",till_water_storage_dot_name);
    p->set<std::string> ("Melting Rate Variable Name",melting_rate_name);
    p->set<std::string> ("Surface Water Input Variable Name",surface_water_input_name);
    p->set<Teuchos::ParameterList*> ("LandIce Hydrology Parameters",&hy_pl);
    p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&phys_pl);
    p->set<std::string> ("Side Set Name",basalSideName);

    //Output
    p->set<std::string> ("Till Water Storage Eqn Residual Name", hydro_resids_names[2]);

    ev = Teuchos::rcp(new HydrologyResidualTillStorageEqn<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // =============== Hydrology quantities =============== //

  //--- Effective pressure ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Effective Pressure"));
  add_dep(basal_fname(effective_pressure_name),basal_fname(ice_overburden_name));
  add_dep(basal_fname(effective_pressure_name),basal_fname(water_pressure_name));

  // Input
  p->set<bool>("Nodal",false);
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<std::string>("Ice Overburden Variable Name", basal_fname(ice_overburden_name));
  p->set<std::string>("Water Pressure Variable Name", basal_fname(water_pressure_name));

  // Output
  p->set<std::string>("Effective Pressure Variable Name", basal_fname(effective_pressure_name));

  // ... QPs...
  ev = Teuchos::rcp(new LandIce::EffectivePressure<EvalT,PHAL::AlbanyTraits,false>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  // ... and Nodes
  p->set<bool>("Nodal",true);
  ev = Teuchos::rcp(new LandIce::EffectivePressure<EvalT,PHAL::AlbanyTraits,false>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Water Discharge ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology: Water Discharge"));
  add_dep(basal_fname(water_discharge_name),basal_fname(water_thickness_name));
  add_dep(basal_fname(water_discharge_name),basal_fname(hydropotential_name));

  // Input
  p->set<std::string> ("Water Thickness Variable Name",basal_fname(water_thickness_name));
  p->set<std::string> ("Hydraulic Potential Gradient Variable Name", basal_fname(grad_fname(hydropotential_name)));
  p->set<std::string> ("Hydraulic Potential Gradient Norm Variable Name", basal_fname(grad_fname(hydropotential_name)+"_norm"));
  p->set<std::string>("Transmissivity Parameter Name", LandIce::ParamEnumName::Kappa);
  p->set<std::string> ("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*> ("LandIce Hydrology",&hy_pl);
  p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&phys_pl);

  //Output
  p->set<std::string> ("Water Discharge Variable Name",basal_fname(water_discharge_name));

  ev = Teuchos::rcp(new HydrologyWaterDischarge<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Melting Rate -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Melting Rate"));

  //Input
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<std::string>("Geothermal Heat Source Variable Name",basal_fname(geothermal_flux_name));
  p->set<std::string>("Sliding Velocity Variable Name",basal_fname(sliding_velocity_name));
  p->set<std::string>("Basal Friction Coefficient Variable Name",basal_fname(beta_name));
  p->set<Teuchos::ParameterList*>("LandIce Hydrology",&hy_pl);
  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters",&phys_pl);

  //Output
  p->set<std::string> ("Melting Rate Variable Name",basal_fname(melting_rate_name));

  // ... QPs...
  ev = Teuchos::rcp(new HydrologyMeltingRate<EvalT,PHAL::AlbanyTraits,true>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  // ... and Nodes
  p->set<bool>("Nodal", true);    // If we have mass lumping or we are saving melting_rate to mesh
  ev = Teuchos::rcp(new HydrologyMeltingRate<EvalT,PHAL::AlbanyTraits,true>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- Hydraulic Potential --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydraulic Potential"));
  add_dep(basal_fname(hydropotential_name),basal_fname(ice_overburden_name));
  add_dep(basal_fname(hydropotential_name),basal_fname(water_pressure_name));
  add_dep(basal_fname(hydropotential_name),basal_fname(water_thickness_name));

  //Input
  p->set<std::string>("Basal Gravitational Water Potential Variable Name",basal_fname(grav_hydropotential_name));
  p->set<std::string>("Ice Overburden Variable Name",basal_fname(ice_overburden_name));
  p->set<std::string>("Water Pressure Variable Name", basal_fname(water_pressure_name));
  p->set<std::string>("Water Thickness Variable Name", basal_fname(water_thickness_name));
  p->set<std::string>("Side Set Name", basalSideName);

  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters",&phys_pl);
  p->set<Teuchos::ParameterList*>("LandIce Hydrology", &hy_pl);

  //Output
  p->set<std::string> ("Hydraulic Potential Variable Name",basal_fname(hydropotential_name));

  // ... QPs...
  ev = Teuchos::rcp(new HydraulicPotential<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  // ... and Nodes
  p->set<bool>("Nodal", true);
  ev = Teuchos::rcp(new HydraulicPotential<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Basal Gravitational Potential (QPs) -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Basal Gravitational Water Potential"));

  //Input
  p->set<std::string> ("Surface Height Variable Name",basal_fname(surface_height_name));
  p->set<std::string> ("Ice Thickness Variable Name",basal_fname(ice_thickness_name));
  p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&phys_pl);
  p->set<std::string>("Side Set Name", basalSideName);

  //Output
  p->set<std::string> ("Basal Gravitational Water Potential Variable Name",basal_fname(grav_hydropotential_name));

  // ... QPs...
  ev = Teuchos::rcp(new BasalGravitationalWaterPotential<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  // ... and Nodes
  p->set<bool> ("Nodal", true);
  ev = Teuchos::rcp(new BasalGravitationalWaterPotential<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  // =============== Intermediate variables =============== //

  //--- Norm of hydropotential gradient --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Hydropotential Norm"));
  add_dep(basal_fname(sliding_velocity_name),basal_fname(velocity_name));

  // Input
  p->set<std::string>("Field Name",basal_fname(grad_fname(hydropotential_name)));
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Field Norm"));

  // Output
  p->set<std::string>("Field Norm Name",basal_fname(grad_fname(hydropotential_name)+"_norm"));
  p->set<std::string>("Field Layout","Cell Side Node Vector");
  ev = Teuchos::rcp(new PHAL::FieldFrobeniusNorm<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Sliding velocity ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Velocity Norm"));
  add_dep(basal_fname(sliding_velocity_name),basal_fname(velocity_name));

  // Input
  p->set<std::string>("Field Name",basal_fname(velocity_name));
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Field Norm"));

  // Output
  p->set<std::string>("Field Norm Name",basal_fname(sliding_velocity_name));

  // ... QPs...
  p->set<std::string>("Field Layout","Cell Side Node Vector");
  ev = Teuchos::rcp(new PHAL::FieldFrobeniusNorm<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  // ... and Nodes
  p->set<std::string>("Field Layout","Cell Side QuadPoint Vector");
  ev = Teuchos::rcp(new PHAL::FieldFrobeniusNorm<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Ice Overburden --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Velocity Norm"));
  add_dep(basal_fname(ice_overburden_name),basal_fname(ice_thickness_name));

  // Input
  p->set<std::string>("Side Set Name", basalSideName);
  p->set<std::string>("Ice Thickness Variable Name", basal_fname(ice_thickness_name));
  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &phys_pl);

  // Output
  p->set<std::string>("Ice Overburden Variable Name", basal_fname(ice_overburden_name));

  // ... QPs...
  ev = Teuchos::rcp(new LandIce::IceOverburden<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  // ... and Nodes
  p->set<bool>("Nodal",true);
  ev = Teuchos::rcp(new LandIce::IceOverburden<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Shared Parameter for transmissivity coefficient: kappa ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: lambda"));

  auto param_name = ParamEnumName::Kappa;
  p->set<std::string>("Parameter Name", param_name);
  p->set<Teuchos::RCP<Albany::ScalarParameterAccessors<EvalT>>>("Accessors", this->getAccessors()->template at<EvalT>());
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
  p->set<double>("Default Nominal Value", hy_pl.sublist("Darcy Law").get<double>(param_name,-1.0));

  Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>> ptr_kappa;
  ptr_kappa = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ptr_kappa);

  // -------- Regularization from Homotopy Parameter h: reg = 10^(-10*h)
  p = Teuchos::rcp(new Teuchos::ParameterList("Simple Op"));

  //Input
  p->set<std::string> ("Input Field Name",ParamEnumName::GLHomotopyParam);
  p->set<Teuchos::RCP<PHX::DataLayout>> ("Field Layout",dl->shared_param);
  p->set<double>("Tau",-10.0*log(10.0));

  //Output
  p->set<std::string> ("Output Field Name","Regularization");

  ev = Teuchos::rcp(new UnaryExpOp<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);
}

template <typename EvalT>
void
LandIce::StokesFOHydrology::constructFields(PHX::FieldManager<PHAL::AlbanyTraits> &fm0)
{
  constructStokesFOBaseFields<EvalT>(fm0);
}

} // Namespace LandIce

#endif // LANDICE_STOKES_FO_HYDROLOGY_PROBLEM_HPP
