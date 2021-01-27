//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_HYDROLOGY_PROBLEM_HPP
#define LANDICE_HYDROLOGY_PROBLEM_HPP 1

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"
#include "LandIce_ResponseUtilities.hpp"

#include "PHAL_Dimension.hpp"
#include "PHAL_FieldFrobeniusNorm.hpp"
#include "PHAL_LoadStateField.hpp"
#include "PHAL_SaveStateField.hpp"
#include "PHAL_Workset.hpp"

#include "LandIce_BasalFrictionCoefficient.hpp"
#include "LandIce_HydrologyBasalGravitationalWaterPotential.hpp"
#include "LandIce_HydraulicPotential.hpp"
#include "LandIce_EffectivePressure.hpp"
#include "LandIce_FlowRate.hpp"
#include "LandIce_IceOverburden.hpp"
#include "LandIce_HydrologyMeltingRate.hpp"
#include "LandIce_HydrologyResidualCavitiesEqn.hpp"
#include "LandIce_HydrologyResidualMassEqn.hpp"
#include "LandIce_HydrologyResidualTillStorageEqn.hpp"
#include "LandIce_HydrologyWaterDischarge.hpp"
#include "LandIce_HydrologySurfaceWaterInput.hpp"
#include "LandIce_HydrologyWaterThickness.hpp"
#include "LandIce_ParamEnum.hpp"

#include "PHAL_SharedParameter.hpp"
#include "LandIce_SimpleOperationEvaluator.hpp"
#include "LandIce_ProblemUtils.hpp"

namespace LandIce
{

/*!
 * \brief  A 2D problem for the subglacial hydrology
 */
/*
 * +------------------------------------------------------------+
 * |                           Summary                          |
 * +------------------------------------------------------------+
 *
 *   We are solving two equations:
 *
 *      dh/dt + div(q) = m/\rho_w + \omega
 *      dh/dt          = (h_r-h)*|u_b|/l_r + m/\rho_i - c_cr AhN^3
 *
 *   where
 *
 *      q   = -kh^a|grad(\phi)|^b grad(\phi)   (water discharge)
 *      m   = (G-\beta*u_b)/L                  (melting rate)
 *      N   = P_o - P                          (eff. pressure def.)
 *      P_o = \rho_i g H                       (ice overburden)
 *      P   = \phi - \rho_w g z_b - \rho_w g h (transportable water pressure)
 *
 *   The unknowns are h (water thickness) and phi (hydraulic potential).
 *   Note: one could chose P instead of \phi.
 *   The first equaiton is a mass conservation equation, while the second
 *   is an evolution equation for the cavities height. Cavities are
 *   supposed to be filled, which is why the equation is for dh/dt.
 *   The other quantities are:
 *
 *      k    : transmissivity constant
 *      rho_i: ice density
 *      rho_w: water density
 *      L    : ice latent heat
 *      G    : (net) geothermal flux
 *      beta : friction coefficient in the ice sliding law
 *      g    : gravity acceleration
 *      H    : ice thickness
 *      z_b  : bed elevation
 *      omega: water input reacing bed from surface (e.g., through moulins)
 *      h_r  : typical bed bump height
 *      l_r  : typical bed bump length
 *      u_b  : ice basal velocity
 *      A    : ice softness (A in Glen's law). May be temperature dependent
 *      c_cr : creep closure coefficient
 */

class Hydrology : public Albany::AbstractProblem
{
public:

  //! Constructor
  Hydrology (const Teuchos::RCP<Teuchos::ParameterList>& problemPparams,
             const Teuchos::RCP<Teuchos::ParameterList>& discParams,
             const Teuchos::RCP<ParamLib>& paramLib,
             const int numDimensions);

  //! Return number of spatial dimensions
  int spatialDimension () const { return numDim; }

  //! Get boolean telling code if SDBCs are utilized
  virtual bool useSDBCs() const {return use_sdbcs_; }

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

  //! Main problem setup routine. Not directly called, but indirectly by buildEvaluators
  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Boundary conditions evaluators
  void constructDirichletEvaluators (const Albany::MeshSpecsStruct& meshSpecs);
  void constructNeumannEvaluators   (const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

protected:

  bool eliminate_h;
  bool unsteady;
  bool has_h_till;
  bool has_p_dot;

  //! Problem PL 
  const Teuchos::RCP<Teuchos::ParameterList> params;

  unsigned int numDim;
  std::string elementBlockName;

  //! Discretization parameter list
  Teuchos::RCP<Teuchos::ParameterList> discParams;

  Teuchos::ArrayRCP<std::string> dofs_names;
  Teuchos::ArrayRCP<std::string> diagnostic_dofs_names;
  Teuchos::ArrayRCP<std::string> prognostic_dofs_names;
  Teuchos::ArrayRCP<std::string> prognostic_dofs_names_dot;
  Teuchos::ArrayRCP<std::string> resid_names;

  Teuchos::RCP<Albany::Layouts> dl;

  Teuchos::RCP<shards::CellTopology> cellType;

  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>> intrepidBasis;

  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature;

  /// Boolean marking whether SDBCs are used
  bool use_sdbcs_;

  static constexpr char water_pressure_name[]                    = "water_pressure";
  static constexpr char water_thickness_name[]                   = "water_thickness";
  static constexpr char till_water_storage_name[]                = "till_water_storage";
  static constexpr char water_pressure_dot_name[]                = "water_pressure_dot";
  static constexpr char water_thickness_dot_name[]               = "water_thickness_dot";
  static constexpr char till_water_storage_dot_name[]            = "till_water_storage_dot";

  static constexpr char hydraulic_potential_name[]               = "hydraulic_potential";
  static constexpr char hydraulic_potential_gradient_name[]      = "hydraulic_potential Gradient";
  static constexpr char hydraulic_potential_gradient_norm_name[] = "hydraulic_potential Gradient Norm";
  static constexpr char ice_softness_name[]                      = "ice_softness";
  static constexpr char ice_overburden_name[]                    = "ice_overburden";
  static constexpr char effective_pressure_name[]                = "effective_pressure";
  static constexpr char ice_temperature_name[]                   = "ice_temperature";
  static constexpr char ice_thickness_name[]                     = "ice_thickness";
  static constexpr char surface_height_name[]                    = "surface_height";
  static constexpr char beta_name[]                              = "beta";
  static constexpr char melting_rate_name[]                      = "melting_rate";
  static constexpr char surface_water_input_name[]               = "surface_water_input";
  static constexpr char surface_mass_balance_name[]              = "surface_mass_balance";
  static constexpr char geothermal_flux_name[]                   = "geothermal_flux";
  static constexpr char water_discharge_name[]                   = "water_discharge";
  static constexpr char sliding_velocity_name[]                  = "sliding_velocity";
  static constexpr char basal_velocity_name[]                    = "basal_velocity";
  static constexpr char basal_grav_water_potential_name[]        = "basal_gravitational_water_potential";
};

// ===================================== IMPLEMENTATION ======================================= //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Hydrology::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                const Albany::MeshSpecsStruct& meshSpecs,
                                Albany::StateManager& stateMgr,
                                Albany::FieldManagerChoice fieldManagerChoice,
                                const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  using FL  = Albany::FieldLocation;
  using FRT = Albany::FieldRankType;

  // Using the utility for the common evaluators
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Service variables for registering state variables and evaluators
  Albany::StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // +---------------------------------------------------------+
  // |       Gathering info on distributed parameters          |
  // +---------------------------------------------------------+

  std::string param_name, stateName, fieldName;

  // Getting the names and infos of the distributed parameters (they are gathered/scattered, not loaded/saved)
  std::map<std::string,bool> is_dist_param;
  std::map<std::string,bool> is_dirichlet_field;
  std::map<std::string,bool> is_dist;
  std::map<std::string,std::string> dist_params_name_to_mesh_part;
  std::set<std::string> inputs_found;

  if (this->params->isSublist("Parameters")) {
    Teuchos::ParameterList& parameterParams = this->params->sublist("Parameters");
    int total_num_param_vecs, num_param_vecs, num_dist_param_vecs;
    Albany::getParameterSizes(parameterParams, total_num_param_vecs, num_param_vecs, num_dist_param_vecs);

    for (unsigned int p_index=0; p_index< (unsigned int) num_dist_param_vecs; ++p_index)
    {
      std::string parameter_sublist_name = Albany::strint("Parameter", p_index+num_param_vecs);
      Teuchos::ParameterList param_list = parameterParams.sublist(parameter_sublist_name);
      param_name = param_list.get<std::string>("Name");
      dist_params_name_to_mesh_part[param_name] = param_list.get<std::string>("Mesh Part","");
      is_dist_param[param_name] = true;
    }
  }

  // Check if a state is a Dirichlet Field. We do this because dirichlet fields MUST
  // be in the DistributedParameterLibrary, which means that a dirichlet field MUST be registered as
  // a NodalDistParameter field, rather than NodalDataToElemNode field.
  // There are three scenarios:
  //  - the user listed the dirichlet field in both the discretization section and
  //    the distributed parameters section: in this case, we assume that the field changes
  //    during the simulation due to optimization. If in the discretization section
  //    it appears as 'Input' or 'Input-Output', then the field loaded in the mesh is used
  //    as 'initial guess'. In this case, there is nothing to do for the field as part of the BC,
  //    since GenericSTKMeshStruct already takes care of loading the field into the mesh,
  //    Albany::Application already takes care of filling the vector in the DistributedParameterLibrary at
  //    the beginning of the simulation (during finalSetUp), and ROL (or whatever package)
  //    will take care of updating the vector in the DistributedParameterLibrary. However, the field *may*
  //    be used also by other parts of the problem, so we gather it.
  //  - the user listed the dirichlet field in the discretization section but NOT in
  //    the distributed parameters section: as in the previous case, we have that
  //    GenericSTKMeshStruct already takes care of loading the field into the mesh, and
  //    Albany::Application already takes care of filling the vector in the DistributedParameterLibrary at
  //    the beginning of the simulation (during finalSetUp). So the field is in the DistributedParameterLibrary,
  //    and we are all set for applying the BC. However, the field *may* be used also by
  //    other parts of the problem, so we gather it.
  //  - the user did not list the dirichlet field in either the discretization section nor
  //    the distributed parameters section: in this case, we ASSUME the user is computing
  //    the field at every iteration, based on states. Notice that it is WRONG to make the field depend
  //    on the solution, since the Jacobian would NOT be correct.
  if (this->params->isSublist("Dirichlet BCs")) {
    const auto& dbcs = this->params->sublist("Dirichlet BCs");
    for(auto it = dbcs.begin(); it !=dbcs.end(); ++it) {
      std::string pname = dbcs.name(it);
      if (pname.find("prescribe Field")!=std::string::npos) {
        // We are prescribing a dirichlet field. The field name is a distributed parameter
        stateName = dbcs.get<std::string>(pname);
        is_dirichlet_field[stateName] = true;
      }
    }
  }

  // +---------------------------------------------------------+
  // |              States variables registration              |
  // +---------------------------------------------------------+

  Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
  unsigned int num_fields = req_fields_info.get<int>("Number Of Fields",0);

  std::string fieldType, fieldUsage, meshPart;
  bool nodal_state, scalar_state;
  std::map<std::string, bool> is_input_state_scalar;
  // Loop over the number of required fields
  for (unsigned int ifield=0; ifield<num_fields; ++ifield)
  {
    // Get info on this field
    Teuchos::ParameterList& thisFieldList = req_fields_info.sublist(Albany::strint("Field", ifield));

    // Get current state name and usage
    stateName  = fieldName = thisFieldList.get<std::string>("Field Name");
    fieldUsage = thisFieldList.get<std::string>("Field Usage","Input"); // WARNING: assuming Input if not specified

    // Skip everything else if the field is unused. This may seem odd (why is it in the plist to start with?),
    // but it could be the user is switching between different options for the model, which require different
    // inputs. To minimize changes in the input files, the user can simply mark a field 'Unused' when the
    // current configuration does not need/produce it.
    if (fieldUsage == "Unused") {
      continue;
    }

    is_dist_param.insert(std::pair<std::string,bool>(stateName, false));      //gets inserted only if not there.
    is_dirichlet_field.insert(std::pair<std::string,bool>(stateName, false)); //gets inserted only if not there.

    // Determine if we need to load/save (or gather/scatter) the field
    bool inputField  = (fieldUsage == "Input")  || (fieldUsage == "Input-Output") || is_dist_param[stateName] || is_dirichlet_field[stateName];
    bool outputField = (fieldUsage == "Output") || (fieldUsage == "Input-Output");

    // Mark the field as found (useful for more verbose errors later on), and, if a parameter, getting its mesh part
    inputs_found.insert(stateName);
    meshPart = is_dist_param[stateName] ? dist_params_name_to_mesh_part[stateName] : "";

    // Getting the type of field: node/elem scalar/vector
    fieldType  = thisFieldList.get<std::string>("Field Type");

    // Registering the state, according to its type
    if(fieldType == "Elem Scalar") {
      entity = Albany::StateStruct::ElemData;
      p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, elementBlockName, true, &entity, meshPart);
      nodal_state = false;
      scalar_state = true;

      // Sanity check: dist parameters and dirichlet field MUST be node scalars
      TEUCHOS_TEST_FOR_EXCEPTION (is_dist_param[stateName] || is_dirichlet_field[stateName], std::logic_error,
                                  "Error! Distributed parameters and dirichlet fields MUST be node scalars.\n");
    } else if(fieldType == "Node Scalar") {
      // Note: a Dirichlet field must be registered as a NodalDistParameter, since it must end up in the DistributedParameterLibrary
      entity = is_dist_param[stateName] || is_dirichlet_field[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, meshPart);
      nodal_state = true;
      scalar_state = true;
    } else if(fieldType == "Elem Vector") {
      entity = Albany::StateStruct::ElemData;
      p = stateMgr.registerStateVariable(stateName, dl->cell_vector, elementBlockName, true, &entity, meshPart);
      nodal_state = false;
      scalar_state = false;

      // Sanity check: dist parameters and dirichlet field MUST be node scalars
      TEUCHOS_TEST_FOR_EXCEPTION (is_dist_param[stateName] || is_dirichlet_field[stateName], std::logic_error,
                                  "Error! Distributed parameters and dirichlet fields MUST be node scalars.\n");
    } else if(fieldType == "Node Vector") {
      entity = is_dist_param[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, true, &entity, meshPart);
      nodal_state = true;
      scalar_state = false;

      // Sanity check: dist parameters and dirichlet field MUST be node scalars
      TEUCHOS_TEST_FOR_EXCEPTION (is_dist_param[stateName] || is_dirichlet_field[stateName], std::logic_error,
                                  "Error! Distributed parameters and dirichlet fields MUST be node scalars.\n");
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue, "Error! Invalid value '" << fieldType << "' for parameter 'Field Type'.\n");
    }

    // If an output field and not a parameter, save it.
    if (outputField)
    {
      // A distributed parameter should not be updated by the problem, in general, so we should not scatter it.
      // We do not need to save it either, since it is already taken care of by ObserverImpl, which is hidden
      // inside a PiroObserver inside the Piro solver.

      // A dirichlet field MUST be registered as a NodalDistParameter, so that it ends up is in the DistributedParameterLibrary.
      // The ObserverImpl (hidden in the PiroObserver, hidden itself in the Piro solver) already takes care
      // of updating the mesh with the current vectors in the DistributedParameterLibrary.
      // Hence, a dirichlet field is also already taken care of.
      if (!is_dist_param[stateName] || !is_dirichlet_field[stateName]) {
        // A 'regular' field output: save it.
        p->set<bool>("Nodal State", nodal_state);
        ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }

      // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
      if ( fieldManagerChoice==Albany::BUILD_RESID_FM && ev->evaluatedFields().size()>0) {
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
      }
    }

    // If an input field. If a parameter, gather it; if a dirichlet field, still gather it, since we *may*
    // need it somewhere else; if none of the above, simply load it.
    if (inputField) {
      if (is_dist_param[stateName] || is_dirichlet_field[stateName])
      {
        // A parameter: gather it
        ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
        fm0.template registerEvaluator<EvalT>(ev);
      } else {
        // A 'regular' field input: load it.
        p->set<std::string>("Field Name", fieldName);
        ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
      is_input_state_scalar[stateName] = scalar_state;
    }
  }

  // +---------------------------------------------------------+
  // |      Handling dist params not already processed         |
  // +---------------------------------------------------------+

  // Trying to be nice to the user, and setting up things automatically if we can
  for (auto it : dist_params_name_to_mesh_part) {
    if (inputs_found.find(it.first)==inputs_found.end()) {
      // The user has not specified this distributed parameter in the discretization list.
      // That's ok, since we have all we need to register the state ourselves, and create
      // and register its gather evaluator

      // Get info
      entity = Albany::StateStruct::NodalDistParameter;
      meshPart = dist_params_name_to_mesh_part[it.first];

      // Register the state
      p = stateMgr.registerStateVariable(it.first, dl->node_scalar, elementBlockName, true, &entity, meshPart);

      // Gather evaluator
      ev = evalUtils.constructGatherScalarNodalParameter(it.first,it.first);
      fm0.template registerEvaluator<EvalT>(ev);
      is_input_state_scalar[it.first] = true;

      // Mark this state as 'found'
      inputs_found.insert(it.first);
    }
  }

  // +---------------------------------------------------------+
  // |     Handling dirichlet fields not already processed     |
  // +---------------------------------------------------------+

  // If a dirichlet field was not declared in the mesh, and was not found in the 'Distributed Parameters'
  // section, then we ASSUME that the user computes it as a field during the field manager evaluation.
  // In this case, we need to scatter it, so that its values end up in the DistributedParameterLibrary.
  // Notice that, although it is WRONG to make the field depend on the solution (the Jacobian would be wrong),
  // it is still possible that the field depends on time-dependent states, so we need to scatter it at every
  // iteration. However, here we ASSUME this is not the case, and scatter it only once.
  for (auto it : is_dirichlet_field) {
    if (inputs_found.find(it.first)==inputs_found.end()) {
      // Get info
      entity = Albany::StateStruct::NodalDistParameter;
      meshPart = dist_params_name_to_mesh_part[it.first];

      // Register the state
      p = stateMgr.registerStateVariable(it.first, dl->node_scalar, elementBlockName, true, &entity, meshPart);

      // Create the scatter evaluator
      ev = evalUtils.constructScatterScalarNodalParameter(it.first,it.first);
      fm0.template registerEvaluator<EvalT>(ev);

      // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
      if ( fieldManagerChoice==Albany::BUILD_RESID_FM && ev->evaluatedFields().size()>0) {
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
      }

      // Mark the state as 'found'
      inputs_found.insert(it.first);
    }
  }

  // ------------------------------- END OF STATES REGISTRATION/LOAD/SAVE -------------------------------- //

  // +---------------------------------------------------------+
  // |     Creating interpolation and utilities evaluators     |
  // +---------------------------------------------------------+

  // Gather solution field (possibly with time derivatives)
  if (unsteady) {
    ev = evalUtils.constructGatherSolutionEvaluator_noTransient (false, diagnostic_dofs_names);
    fm0.template registerEvaluator<EvalT> (ev);

    int offset = diagnostic_dofs_names.size();
    ev = evalUtils.constructGatherSolutionEvaluator (false, prognostic_dofs_names, prognostic_dofs_names_dot, offset);
    fm0.template registerEvaluator<EvalT> (ev);
  } else {
    ev = evalUtils.constructGatherSolutionEvaluator_noTransient (false, dofs_names);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // Compute basis functions
  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Gather coordinates
  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter residual
  int offset = 0;
  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Hydrology");
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolating all input fields
  for (auto it : is_input_state_scalar) {
    if (it.second) {
      ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator(it.first);
    } else {
      ev = evalUtils.getPSTUtils().constructDOFVecInterpolationEvaluator(it.first);
    }
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // Interpolate Water Pressure
  ev = evalUtils.constructDOFInterpolationEvaluator(water_pressure_name);
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate Water Discharge from QPs to cell (in case we want to save it)
  ev = evalUtils.constructCellAverageEvaluator(water_discharge_name,FL::QuadPoint,FRT::Vector);
  fm0.template registerEvaluator<EvalT>(ev);

  // Interpolate Water Thickness
  if (!eliminate_h) {
    ev = evalUtils.constructDOFInterpolationEvaluator(water_thickness_name);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // Interpolate time derivatives
  for (auto dof_dot_name : prognostic_dofs_names_dot) {
    ev = evalUtils.constructDOFInterpolationEvaluator(dof_dot_name);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // Hydraulic Potential Gradient
  ev = evalUtils.constructDOFGradInterpolationEvaluator(hydraulic_potential_name);
  fm0.template registerEvaluator<EvalT> (ev);

  // +---------------------------------------------------------+
  // |           Creating LandIce specific evaluators            |
  // +---------------------------------------------------------+

  auto hy_pl = params->sublist("LandIce Hydrology");

  // ---- Compute Water Input ----- //
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Hydrology Water Input"));

  // Input
  p->set<std::string>("Surface Mass Balance Variable Name",surface_mass_balance_name);
  p->set<std::string>("Surface Height Variable Name",surface_height_name);
  p->set<Teuchos::ParameterList*> ("Surface Water Input Params",&hy_pl.sublist("Surface Water Input"));

  // Output
  p->set<std::string>("Surface Water Input Variable Name",surface_water_input_name);

  ev = Teuchos::rcp(new HydrologySurfaceWaterInput<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Basal Gravitational Potential (QPs) -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Basal Gravitational Water Potential"));

  //Input
  p->set<std::string> ("Surface Height Variable Name",surface_height_name);
  p->set<std::string> ("Ice Thickness Variable Name",ice_thickness_name);
  p->set<bool> ("Nodal", false);

  p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&params->sublist("LandIce Physical Parameters"));

  //Output
  p->set<std::string> ("Basal Gravitational Water Potential Variable Name",basal_grav_water_potential_name);

  ev = Teuchos::rcp(new BasalGravitationalWaterPotential<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Basal Gravitational Potential (Nodes) -------- //
  p->set<bool> ("Nodal", true);
  ev = Teuchos::rcp(new BasalGravitationalWaterPotential<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------------ Ice Overburden (QPs) ------------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Ice Overburden"));

  //Input
  p->set<std::string>("Ice Thickness Variable Name", ice_thickness_name);
  p->set<bool>("Nodal", false);

  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters",&params->sublist("LandIce Physical Parameters"));

  //Output
  p->set<std::string>("Ice Overburden Variable Name",ice_overburden_name);

  ev = Teuchos::rcp(new IceOverburden<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------------ Ice Overburden (Nodes) ------------- //

  p->set<bool>("Nodal", true);
  ev = Teuchos::rcp(new IceOverburden<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydraulic Potential (QPs) -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydraulic Potential"));

  //Input
  p->set<std::string>("Basal Gravitational Water Potential Variable Name",basal_grav_water_potential_name);
  p->set<std::string>("Ice Overburden Variable Name",ice_overburden_name);
  p->set<std::string>("Water Pressure Variable Name", water_pressure_name);
  p->set<std::string>("Water Thickness Variable Name", water_thickness_name);
  p->set<bool>("Nodal", false);

  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters",&params->sublist("LandIce Physical Parameters"));
  p->set<Teuchos::ParameterList*>("LandIce Hydrology", &hy_pl);

  //Output
  p->set<std::string> ("Hydraulic Potential Variable Name",hydraulic_potential_name);

  ev = Teuchos::rcp(new HydraulicPotential<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydraulic Potential (Nodes) -------- //

  p->set<bool>("Nodal", true);
  ev = Teuchos::rcp(new HydraulicPotential<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Water Discharge -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Water Discharge"));

  p->set<std::string>("Water Thickness Variable Name",water_thickness_name);
  p->set<std::string>("Hydraulic Potential Gradient Variable Name",hydraulic_potential_gradient_name);
  p->set<std::string>("Hydraulic Potential Gradient Norm Variable Name",hydraulic_potential_gradient_norm_name);
  p->set<std::string>("Regularization Parameter Name","Regularization");

  p->set<Teuchos::ParameterList*>("LandIce Hydrology",&hy_pl);

  //Output
  p->set<std::string>("Water Discharge Variable Name",water_discharge_name);

  ev = Teuchos::rcp(new HydrologyWaterDischarge<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Melting Rate (QPs) -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Melting Rate"));

  //Input
  p->set<std::string>("Geothermal Heat Source Variable Name",geothermal_flux_name);
  p->set<std::string>("Sliding Velocity Variable Name",sliding_velocity_name);
  p->set<std::string>("Basal Friction Coefficient Variable Name",beta_name);
  p->set<Teuchos::ParameterList*>("LandIce Hydrology",&hy_pl);
  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters",&params->sublist("LandIce Physical Parameters"));

  //Output
  p->set<std::string> ("Melting Rate Variable Name",melting_rate_name);

  ev = Teuchos::rcp(new HydrologyMeltingRate<EvalT,PHAL::AlbanyTraits,false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Melting Rate (Nodes) -------- //
  p->set<bool>("Nodal", true);    // If we have mass lumping or we are saving melting_rate to mesh
  ev = Teuchos::rcp(new HydrologyMeltingRate<EvalT,PHAL::AlbanyTraits,false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --------- Ice Softness --------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Flow Rate"));

  // Input
  p->set<std::string>("Temperature Variable Name",ice_temperature_name);
  p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&params->sublist("LandIce Physical Parameters"));
  p->set<Teuchos::ParameterList*> ("Parameter List",&params->sublist("LandIce Viscosity"));

  // Output
  p->set<std::string>("Flow Rate Variable Name",ice_softness_name);

  ev = Teuchos::rcp(new FlowRate<EvalT,PHAL::AlbanyTraits, RealType>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Sliding Velocity -------- //
  if (!is_input_state_scalar[sliding_velocity_name]) {
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Velocity Norm"));

    // Input
    p->set<std::string>("Field Name",basal_velocity_name);
    p->set<std::string>("Field Layout","Cell QuadPoint Vector");
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Field Norm"));

    // Output
    p->set<std::string>("Field Norm Name",sliding_velocity_name);

    ev = Teuchos::rcp(new PHAL::FieldFrobeniusNormParam<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    // ------- Sliding Velocity at nodes (for output in the mesh, if needed) -------- //
    p->set<std::string>("Field Layout","Cell Node Vector");
    ev = Teuchos::rcp(new PHAL::FieldFrobeniusNormParam<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // ------- Hydraulic Potential Gradient Norm -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Velocity Norm"));

  // Input
  p->set<std::string>("Field Name",hydraulic_potential_gradient_name);
  p->set<std::string>("Field Layout","Cell QuadPoint Gradient");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Field Norm"));

  // Output
  p->set<std::string>("Field Norm Name",hydraulic_potential_gradient_norm_name);

  ev = Teuchos::rcp(new PHAL::FieldFrobeniusNorm<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Effective pressure (QPs) ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Effective Pressure"));

  // Input
  p->set<std::string>("Ice Overburden Variable Name", ice_overburden_name);
  p->set<std::string>("Water Pressure Variable Name", water_pressure_name);
  p->set<bool>("Nodal",false);

  // Output
  p->set<std::string>("Effective Pressure Variable Name",effective_pressure_name);

  ev = Teuchos::rcp(new EffectivePressure<EvalT,PHAL::AlbanyTraits, false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Effective pressure (Nodes) ---//
  p->set<bool>("Nodal",true);
  ev = Teuchos::rcp(new EffectivePressure<EvalT,PHAL::AlbanyTraits, false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if (params->sublist("LandIce Basal Friction Coefficient").isParameter("Type")) {
    //--- LandIce basal friction coefficient ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Basal Friction Coefficient"));

    //Input
    p->set<std::string>("Sliding Velocity Variable Name", sliding_velocity_name);
    p->set<std::string>("BF Variable Name", Albany::bf_name);
    p->set<std::string>("Effective Pressure Variable Name", effective_pressure_name);
    p->set<std::string>("Ice Softness Variable Name", ice_softness_name);
    p->set<Teuchos::ParameterList*>("Viscosity Parameter List", &params->sublist("LandIce Viscosity"));
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Basal Friction Coefficient"));
    p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));

    //Output
    p->set<std::string>("Basal Friction Coefficient Variable Name", beta_name);

    ev = createEvaluatorWithThreeScalarTypes<BasalFrictionCoefficient, EvalT>(p,dl, FieldScalarType::Scalar, FieldScalarType::ParamScalar,FieldScalarType::ParamScalar);
    fm0.template registerEvaluator<EvalT>(ev);

    //--- LandIce basal friction coefficient nodal (for output in the mesh, if needed) ---//
    p->set<bool>("Nodal",true);
    ev = createEvaluatorWithThreeScalarTypes<BasalFrictionCoefficient, EvalT>(p,dl, FieldScalarType::Scalar, FieldScalarType::ParamScalar,FieldScalarType::ParamScalar);
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // ------- Hydrology Residual Mass Eqn-------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Residual Mass Eqn"));

  //Input
  p->set<std::string> ("BF Name", Albany::bf_name);
  p->set<std::string> ("Gradient BF Name", Albany::grad_bf_name);
  p->set<std::string> ("Weighted Measure Name", Albany::weights_name);
  p->set<std::string> ("Water Discharge Variable Name", water_discharge_name);
  p->set<std::string> ("Till Water Storage Dot Variable Name", till_water_storage_dot_name);
  p->set<std::string> ("Water Thickness Dot Variable Name", water_thickness_dot_name);
  p->set<std::string> ("Melting Rate Variable Name",melting_rate_name);
  p->set<std::string> ("Surface Water Input Variable Name",surface_water_input_name);
  p->set<bool>("Unsteady",unsteady);
  p->set<bool>("Has Till Storage",has_h_till);

  p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&params->sublist("LandIce Physical Parameters"));
  p->set<Teuchos::ParameterList*> ("LandIce Hydrology Parameters",&hy_pl);

  //Output
  p->set<std::string> ("Mass Eqn Residual Name",resid_names[0]);

  ev = Teuchos::rcp(new HydrologyResidualMassEqn<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if (eliminate_h) {
    // -------- Hydrology Water Thickness (QPs) ------- //
    p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Water Thickness"));

    //Input
    p->set<std::string> ("Water Thickness Variable Name",water_thickness_name);
    p->set<std::string> ("Effective Pressure Variable Name",effective_pressure_name);
    p->set<std::string> ("Melting Rate Variable Name",melting_rate_name);
    p->set<std::string> ("Sliding Velocity Variable Name",sliding_velocity_name);
    p->set<std::string> ("Ice Softness Variable Name",ice_softness_name);
    p->set<bool> ("Nodal", false);
    p->set<Teuchos::ParameterList*> ("LandIce Hydrology Parameters",&hy_pl);
    p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&params->sublist("LandIce Physical Parameters"));

    //Output
    p->set<std::string> ("Water Thickness Variable Name", water_thickness_name);

    ev = Teuchos::rcp(new HydrologyWaterThickness<EvalT,PHAL::AlbanyTraits,false,false>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    // -------- Hydrology Water Thickness (nodes) ------- //
    p->set<bool> ("Nodal", true);
    ev = Teuchos::rcp(new HydrologyWaterThickness<EvalT,PHAL::AlbanyTraits,false,false>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

  } else {
    // ------- Hydrology Cavities Equation Residual -------- //
    p = Teuchos::rcp(new Teuchos::ParameterList("Hydrology Residual Cavities Eqn"));

    //Input
    p->set<std::string> ("BF Name", Albany::bf_name);
    p->set<std::string> ("Weighted Measure Name", Albany::weights_name);
    p->set<std::string> ("Water Thickness Variable Name",water_thickness_name);
    p->set<std::string> ("Water Thickness Dot Variable Name",water_thickness_dot_name);
    p->set<std::string> ("Water Pressure Dot Variable Name",water_pressure_dot_name);
    p->set<std::string> ("Effective Pressure Variable Name",effective_pressure_name);
    p->set<std::string> ("Melting Rate Variable Name",melting_rate_name);
    p->set<std::string> ("Sliding Velocity Variable Name",sliding_velocity_name);
    p->set<std::string> ("Ice Softness Variable Name",ice_softness_name);
    p->set<bool> ("Unsteady", unsteady);
    p->set<Teuchos::ParameterList*> ("LandIce Hydrology Parameters",&hy_pl);
    p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&params->sublist("LandIce Physical Parameters"));

    //Output
    p->set<std::string> ("Cavities Eqn Residual Name", resid_names[1]);

    ev = Teuchos::rcp(new HydrologyResidualCavitiesEqn<EvalT,PHAL::AlbanyTraits,false,false>(*p,dl));
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
    p->set<Teuchos::ParameterList*> ("LandIce Physical Parameters",&params->sublist("LandIce Physical Parameters"));

    //Output
    p->set<std::string> ("Till Water Storage Eqn Residual Name", resid_names[2]);

    ev = Teuchos::rcp(new HydrologyResidualTillStorageEqn<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // -------- Regularization from Homotopy Parameter h: reg = 10^(-10*h)
  p = Teuchos::rcp(new Teuchos::ParameterList("Simple Op"));

  //Input
  p->set<std::string> ("Input Field Name",ParamEnumName::HomotopyParam);
  p->set<Teuchos::RCP<PHX::DataLayout>> ("Field Layout",dl->shared_param);
  p->set<double>("Tau",-10.0*log(10.0));

  //Output
  p->set<std::string> ("Output Field Name","Regularization");

  ev = Teuchos::rcp(new UnaryExpOp<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Shared Parameter for basal friction coefficient: lambda ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: lambda"));

  param_name = ParamEnumName::Lambda;
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Lambda>> ptr_lambda;
  ptr_lambda = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Lambda>(*p,dl));
  ptr_lambda->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Basal Friction Coefficient").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_lambda);

  //--- Shared Parameter for basal friction coefficient: mu ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: mu"));

  param_name = ParamEnumName::MuCoulomb;
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::MuCoulomb>> ptr_mu;
  ptr_mu = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::MuCoulomb>(*p,dl));
  ptr_mu->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Basal Friction Coefficient").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_mu);

  //--- Shared Parameter for basal friction coefficient: power ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: power"));

  param_name = ParamEnumName::Power;
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Power>> ptr_power;
  ptr_power = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Power>(*p,dl));
  ptr_power->setNominalValue(params->sublist("Parameters"),params->sublist("LandIce Basal Friction Coefficient").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_power);

  //--- Shared Parameter for Continuation:  ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Homotopy Parameter"));

  param_name = ParamEnumName::HomotopyParam;
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  // Note: homotopy param (h) is used to regularize. Hence, set default to 1.0, in case there's no continuation,
  //       so we regularize very little. Recall that if no nominal values are set in input files, setNominalValue picks
  //       the value passed as second input.
  Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>> ptr_homotopy;
  ptr_homotopy = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Homotopy>(*p,dl));
  ptr_homotopy->setNominalValue(params->sublist("Parameters"),1.0);
  fm0.template registerEvaluator<EvalT>(ptr_homotopy);

  // ----------------------------------------------------- //

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Hydrology", dl->dummy);
    fm0.template requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));

    Teuchos::RCP<const Albany::MeshSpecsStruct> meshSpecsPtr = Teuchos::rcpFromRef(meshSpecs);
    paramList->set<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", meshSpecsPtr);
    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

} // Namespace LandIce

#endif // LANDICE_HYDROLOGY_PROBLEM_HPP
