//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ProblemUtils.hpp"
#include "Albany_Utils.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "ProjectionProblem.hpp"
#include "Shards_CellTopology.hpp"

Albany::ProjectionProblem::ProjectionProblem(
    RCP<ParameterList> const & parameter_list,
    RCP<ParamLib> const & parameter_library,
    int const number_dimensions) :
    Albany::AbstractProblem(
        parameter_list,
        parameter_library,
        number_dimensions + 9), // additional DOF for pore pressure
    have_boundary_source_(false),
    number_dimensions_(number_dimensions),
    projection_(
        params->sublist("Projection").get("Projection Variable", ""),
        params->sublist("Projection").get("Projection Rank", 0),
        params->sublist("Projection").get("Projection Comp", 0),
        number_dimensions_)
{
  std::string &
  method = params->get("Name", "Total Lagrangian Plasticity with Projection ");

  have_boundary_source_ = params->isSublist("Source Functions");

  material_model_name_ =
      params->sublist("Material Model").get("Model Name", "Neohookean");

  projected_field_name_ =
      params->sublist("Projection").get("Projection Variable", "");

  projection_rank_ = params->sublist("Projection").get("Projection Rank", 0);

  *out << "Problem Name = " << method << std::endl;
  *out << "Projection Variable: " << projected_field_name_ << std::endl;
  *out << "Projection Variable Rank: " << projection_rank_ << std::endl;

  insertion_criterion_ =
      params->sublist("Insertion Criteria").get("Insertion Criteria", "");

  // Only run if there is a projection variable defined
  if (projection_.isProjected()) {

    // For debug purposes
    *out << "Will variable be projected? " << projection_.isProjected();
    *out << std::endl;
    *out << "Number of components: " << projection_.getProjectedComponents();
    *out << std::endl;
    *out << "Rank of variable: " << projection_.getProjectedRank();
    *out << std::endl;

    //
    // the evaluator constructor requires information on the size of the
    // projected variable as boolean flags in the argument list. Allowed
    // variable types are vector, (rank 2) tensor, or scalar (default).
    // Currently doesn't really do anything. Have to change when
    // I decide how to store the variable
    //
    switch (projection_.getProjectedRank()) {

    case 1:
      is_field_vector_ = true;
      is_field_tensor_ = false;
      break;

    case 2:
      is_field_vector_ = true;
      is_field_tensor_ = false;
      break;

    default:
      is_field_vector_ = false;
      is_field_tensor_ = false;
      break;
    }
  }

// Changing this ifdef changes ordering from  (X,Y,T) to (T,X,Y)
//#define NUMBER_T_FIRST
#ifdef NUMBER_T_FIRST
  target_offset_= 0;
  source_offset_= projection_.getProjectedComponents();
#else
  source_offset_ = 0;
  target_offset_ = number_dimensions_;
#endif

// returns the problem information required for setting the rigid body modes
// (RBMs) for elasticity problems (in src/Albany_SolverFactory.cpp)
// IK, 2012-02

  int number_PDEs = number_dimensions_ + projection_.getProjectedComponents();
  int number_elasticity_dimensions = number_dimensions_;
  int number_scalar_dimensions = projection_.getProjectedComponents();
  int null_space_dimensions = 0;

  switch (number_dimensions_) {
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Invalid number of dimensions");
    break;
  case 1:
    null_space_dimensions = 0;
    break;
  case 2:
    null_space_dimensions = 3;
    break;
  case 3:
    null_space_dimensions = 6;
    break;
  }

  rigidBodyModes->setParameters(number_PDEs, number_elasticity_dimensions, 
       number_scalar_dimensions, null_space_dimensions);

}

//
// Simple destructor
//
Albany::ProjectionProblem::~ProjectionProblem()
{
}

void
Albany::ProjectionProblem::buildProblem(
    ArrayRCP<RCP<Albany::MeshSpecsStruct>> mesh_specs,
    Albany::StateManager & state_manager)
{
  // Construct all Phalanx evaluators
  TEUCHOS_TEST_FOR_EXCEPTION(
      mesh_specs.size() != 1,
      std::logic_error,
      "Problem supports one Material Block");

  fm.resize(1);
  fm[0] = rcp(new PHX::FieldManager<AlbanyTraits>);

  buildEvaluators(
      *fm[0],
      *mesh_specs[0],
      state_manager,
      BUILD_RESID_FM,
      Teuchos::null);

  constructDirichletEvaluators(*mesh_specs[0]);
}

Teuchos::Array<RCP<const PHX::FieldTag>>
Albany::ProjectionProblem::buildEvaluators(
    PHX::FieldManager<AlbanyTraits> & field_manager,
    Albany::MeshSpecsStruct const & mesh_specs,
    Albany::StateManager & state_manager,
    Albany::FieldManagerChoice field_manager_choice,
    RCP<ParameterList> const & response_list)
{
  // Call
  // constructeEvaluators<Evaluator>(*rfm[0], *mesh_specs[0], state_manager);
  // for each Evaluator in AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<ProjectionProblem>
  construct_evaluator(
      *this,
      field_manager,
      mesh_specs,
      state_manager,
      field_manager_choice,
      response_list);

  Sacado::mpl::for_each<AlbanyTraits::BEvalTypes>(construct_evaluator);

  return *construct_evaluator.tags;
}

void
Albany::ProjectionProblem::constructDirichletEvaluators(
    Albany::MeshSpecsStruct const & mesh_specs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string>
  dirichlet_names(neq);

  dirichlet_names[source_offset_] = "X";
  if (number_dimensions_ > 1) dirichlet_names[source_offset_ + 1] = "Y";
  if (number_dimensions_ > 2) dirichlet_names[source_offset_ + 2] = "Z";

  dirichlet_names[target_offset_] = "T";

  Albany::BCUtils<Albany::DirichletTraits>
  dirichlet_utils;

  dfm = dirichlet_utils.constructBCEvaluators(
      mesh_specs.nsNames,
      dirichlet_names,
      this->params,
      this->paramLib);
}

RCP<const ParameterList>
Albany::ProjectionProblem::getValidProblemParameters() const
{
  RCP<ParameterList>
  parameters = this->getGenericProblemParams("ValidProjectionProblemParams");

  parameters->sublist("Material Model", false, "");

  parameters->set<bool>("avgJ", false,
      "Flag to indicate the J should be averaged");

  parameters->set<bool>("volavgJ", false,
      "Flag to indicate the J should be volume averaged");

  parameters->set<bool>("weighted_Volume_Averaged_J", false,
      "Flag to indicate the J should be volume averaged with stabilization");

  parameters->sublist("Elastic Modulus", false, "");
  parameters->sublist("Shear Modulus", false, "");
  parameters->sublist("Poissons Ratio", false, "");
  parameters->sublist("Projection", false, "");
  parameters->sublist("Insertion Criteria", false, "");

  if (material_model_name_ == "J2" || material_model_name_ == "J2Fiber") {

    parameters->set<bool>("Compute Dislocation Density Tensor", false,
        "Flag to compute the dislocaiton density tensor (only for 3D)");

    parameters->sublist("Hardening Modulus", false, "");
    parameters->sublist("Saturation Modulus", false, "");
    parameters->sublist("Saturation Exponent", false, "");
    parameters->sublist("Yield Strength", false, "");

    if (material_model_name_ == "J2Fiber") {
      parameters->set<RealType>("xiinf_J2", false, "");
      parameters->set<RealType>("tau_J2", false, "");
      parameters->set<RealType>("k_f1", false, "");
      parameters->set<RealType>("q_f1", false, "");
      parameters->set<RealType>("vol_f1", false, "");
      parameters->set<RealType>("xiinf_f1", false, "");
      parameters->set<RealType>("tau_f1", false, "");
      parameters->set<RealType>("Mx_f1", false, "");
      parameters->set<RealType>("My_f1", false, "");
      parameters->set<RealType>("Mz_f1", false, "");
      parameters->set<RealType>("k_f2", false, "");
      parameters->set<RealType>("q_f2", false, "");
      parameters->set<RealType>("vol_f2", false, "");
      parameters->set<RealType>("xiinf_f2", false, "");
      parameters->set<RealType>("tau_f2", false, "");
      parameters->set<RealType>("Mx_f2", false, "");
      parameters->set<RealType>("My_f2", false, "");
      parameters->set<RealType>("Mz_f2", false, "");
    }
  }

  return parameters;
}

void
Albany::ProjectionProblem::getAllocatedStates(
    ArrayRCP<ArrayRCP<RCP<FieldContainer<RealType>>>> old_state,
    ArrayRCP<ArrayRCP<RCP<FieldContainer<RealType>>>> new_state) const
{
  old_state = old_state_;
  new_state = new_state_;
}
