//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if 0

#include "SolidMechanics.h"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_Utils.hpp"
#include "ConstitutiveModelInterface.hpp"
#include "ConstitutiveModelParameters.hpp"
#include "Kinematics.hpp"
#include "Albany_MaterialDatabase.hpp"
#include "MechanicsResidual.hpp"
#include "PHAL_NSMaterialProperty.hpp"
#include "PHAL_SaveStateField.hpp"
#include "PHAL_Source.hpp"
#include "SolutionSniffer.hpp"
#include "Time.hpp"

// Helper functions
namespace {

  int
  computeNumRigidBodyModes(int const space_dimension)
  {
    int
    num_rigid_modes {0};

    switch (space_dimension) {

      default:
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          '\n' << "ERROR: " << __PRETTY_FUNCTION__ <<
          ": wrong number of dimensions." << '\n');
      break;

      case 1:
      num_rigid_modes = 1;
      break;

      case 2:
      num_rigid_modes = 3;
      break;

      case 3:
      num_rigid_modes = 6;
      break;
    }

    return num_rigid_modes;
  }

} // anonymous namespace

//
//
//
Albany::SolidMechanics::
SolidMechanics(
    Teuchos::RCP<Teuchos::ParameterList> const & params,
    Teuchos::RCP<ParamLib> const & param_lib,
    int const num_dims,
    Teuchos::RCP<Teuchos::Comm<int> const> & comm) :
Albany::AbstractProblem(params, param_lib),
num_dims_(num_dims), neq(num_dims)
{

  std::string const &
  method = params->get("Name", "Mechanics ");

  *out << "Problem Name = " << method << '\n';

  this->setNumEquations(num_dims_);

  // Print out a summary of the problem
  *out << "Mechanics problem:" << '\n'
  << "\tSpatial dimension             : " << num_dims_ << '\n'
  << "\tMechanics variables           : "
  << '\n';

  // Need numPDEs should be num_dims_ + nDOF for other governing equations  -SS

  int const
  null_space_dim = computeNumRigidBodyModes(num_dims_);

  rigidBodyModes->setParameters(
      num_dims_,
      num_dims_,
      0,
      null_space_dim);

  material_db_ = Albany::createMaterialDatabase(params, comm);
}

//
//
//
Albany::SolidMechanics::
~SolidMechanics()
{
  return;
}

//
// TODO: This problem should not know about multiple physics.
// It should only deal with a single physics set. Eventually there should
// be a ProblemManager that deals with the different physics sets and
// dispatches each to the appropriate problem.
//
void
Albany::SolidMechanics::
buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> mesh_specs,
    Albany::StateManager & state_mgr)
{
  // Construct All Phalanx Evaluators
  int const
  num_physics_sets = mesh_specs.size();

  assert(num_physics_sets == 1);

  *out << "Num MeshSpecs: " << num_physics_sets << '\n';

  fm.resize(num_physics_sets);

  *out << "Calling SolidMechanics::buildEvaluators" << '\n';

  fm[0] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);

  buildEvaluators(
      *fm[0],
      *mesh_specs[0],
      state_mgr,
      BUILD_RESID_FM,
      Teuchos::null);

  *out << "Calling SolidMechanics::constructDirichletEvaluators" << '\n';
  constructDirichletEvaluators(*mesh_specs[0]);

  bool const
  have_sidesets = mesh_specs[0]->ssNames.size() > 0;

  if (have_sidesets) {
    *out << "Calling SolidMechanics::constructNeumannEvaluators" << '\n';
    constructNeumannEvaluators(*mesh_specs[0]);
  }

  return;
}

//
//
//
Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
Albany::SolidMechanics::
buildEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits> & field_mgr,
    Albany::MeshSpecsStruct const & mesh_specs,
    Albany::StateManager & state_mgr,
    Albany::FieldManagerChoice fm_choice,
    Teuchos::RCP<Teuchos::ParameterList> const & response_list)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *mesh_specs[0], state_mgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<SolidMechanics>
  op(*this,
      field_mgr,
      mesh_specs,
      state_mgr,
      fm_choice,
      response_list);

  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>
  fe(op);

  return *op.tags;
}

//
//
//
void
Albany::SolidMechanics::
constructDirichletEvaluators(const Albany::MeshSpecsStruct & mesh_specs)
{

  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string>
  dirichlet_names(neq);

  dirichlet_names[0] = "X";

  if (num_dims_ > 1) dirichlet_names[1] = "Y";
  if (num_dims_ > 2) dirichlet_names[2] = "Z";

  // Pass on the Application as well. This is needed for
  // the coupled Schwarz BC. It is just ignored otherwise.
  Teuchos::RCP<Albany::Application> const &
  application = getApplication();

  this->params->set<Teuchos::RCP<Albany::Application>>(
      "Application", application);

  Albany::BCUtils<Albany::DirichletTraits>
  dirichlet_utils;

  dfm = dirichlet_utils.constructBCEvaluators(
      mesh_specs.nsNames,
      dirichlet_names,
      this->params,
      this->paramLib);

  offsets_ = dirichlet_utils.getOffsets();
  nodeSetIDs_ = dirichlet_utils.getNodeSetIDs();

  return;
}

//
// Traction BCs
//
void
Albany::SolidMechanics::
constructNeumannEvaluators(
    Albany::MeshSpecsStruct const & mesh_specs)
{
  // Note: we only enter this function if sidesets are defined in the mesh file
  // i.e. mesh_specs.ssNames.size() > 0
  Albany::BCUtils<Albany::NeumannTraits>
  neumann_utils;

  // Check to make sure that Neumann BCs are given in the input file
  if (neumann_utils.haveBCSpecified(this->params) == false) return;

  // Construct BC evaluators for all side sets and names
  // Note that the string index sets up the equation offset,
  // so ordering is important
  std::vector<std::string>
  neumann_names(neq + 1);

  Teuchos::Array<Teuchos::Array<int>>
  offsets(neq + 1);

  neumann_names[0] = "sig_x";
  offsets[0].resize(1);
  offsets[0][0] = 0;

  // The Neumann BC code uses offsets[neq].size() as num dim, so use num_dims_
  // here rather than neq.
  offsets[neq].resize(num_dims_);
  offsets[neq][0] = 0;

  if (num_dims_ > 1) {
    neumann_names[1] = "sig_y";
    offsets[1].resize(1);
    offsets[1][0] = 1;
    offsets[neq][1] = 1;
  }

  if (num_dims_ > 2) {
    neumann_names[2] = "sig_z";
    offsets[2].resize(1);
    offsets[2][0] = 2;
    offsets[neq][2] = 2;
  }

  neumann_names[neq] = "all";

  // Construct BC evaluators for all possible names of conditions
  // Should only specify flux vector components (dudx, dudy, dudz),
  // or dudn, not both
  std::vector<std::string>
  cond_names(3);//dudx, dudy, dudz, dudn, P

  Teuchos::ArrayRCP<std::string>
  dof_names(1);

  dof_names[0] = "Displacement";

  // Note that sidesets are only supported for two and 3D currently
  switch (num_dims_) {
    default:
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
        '\n' << "Error: Sidesets only supported in 2 and 3D." << '\n');
    break;

    case 2:
    cond_names[0] = "(t_x, t_y)";
    break;

    case 3:
    cond_names[0] = "(t_x, t_y, t_z)";
    break;

  }

  cond_names[1] = "dudn";
  cond_names[2] = "P";

  nfm.resize(1); // Elasticity problem only has one element block

  Teuchos::RCP<Albany::MeshSpecsStruct> const &
  pmesh_specs = Teuchos::rcp(&mesh_specs, false);

  nfm[0] = neumann_utils.constructBCEvaluators(
      pmesh_specs,
      neumann_names,
      dof_names,
      true,
      0,
      cond_names,
      offsets,
      dl_,
      this->params,
      this->paramLib);
}

//------------------------------------------------------------------------------
Teuchos::RCP<const Teuchos::ParameterList>
Albany::SolidMechanics::
getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
  this->getGenericProblemParams("ValidSolidMechanicsParams");

  validPL->set<std::string>("MaterialDB Filename",
      "materials.xml",
      "Filename of material database xml file");
  validPL->sublist("Displacement", false, "");

  return validPL;
}

//------------------------------------------------------------------------------
void
Albany::SolidMechanics::
getAllocatedStates(
    Teuchos::ArrayRCP<
    Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>>
    old_state,
    Teuchos::ArrayRCP<
    Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>>
    new_state
) const
{
  old_state = old_state_;
  new_state = new_state_;
}

//------------------------------------------------------------------------------
void
Albany::SolidMechanics::
applyProblemSpecificSolverSettings(
    Teuchos::RCP<Teuchos::ParameterList> params)
{
  // Acquire the NOX "Solver Options" and "Status Tests" parameter lists
  Teuchos::RCP<Teuchos::ParameterList> solverOptionsParameterList;
  Teuchos::RCP<Teuchos::ParameterList> statusTestsParameterList;
  if(params->isSublist("Piro")) {
    if(params->sublist("Piro").isSublist("NOX")) {
      if(params->sublist("Piro").sublist("NOX").isSublist("Solver Options")) {
        solverOptionsParameterList = Teuchos::rcpFromRef( params->sublist("Piro").sublist("NOX").sublist("Solver Options") );
      }
      if(params->sublist("Piro").sublist("NOX").isSublist("Status Tests")) {
        statusTestsParameterList = Teuchos::rcpFromRef( params->sublist("Piro").sublist("NOX").sublist("Status Tests") );
      }
    }
  }

  if(!solverOptionsParameterList.is_null() && !statusTestsParameterList.is_null()) {

    // Add the model evaulator flag as a status test.
    Teuchos::ParameterList originalStatusTestParameterList = *statusTestsParameterList;
    Teuchos::ParameterList newStatusTestParameterList;
    newStatusTestParameterList.set<std::string>("Test Type", "Combo");
    newStatusTestParameterList.set<std::string>("Combo Type", "OR");
    newStatusTestParameterList.set<int>("Number of Tests", 2);
    newStatusTestParameterList.sublist("Test 0");
    newStatusTestParameterList.sublist("Test 0").set("Test Type", "User Defined");
    newStatusTestParameterList.sublist("Test 0").set("User Status Test", nox_status_test_);
    newStatusTestParameterList.sublist("Test 1") = originalStatusTestParameterList;
    *statusTestsParameterList = newStatusTestParameterList;

    // Create a NOX observer that will reset the status flag at the beginning of a nonlinear solve
    Teuchos::RCP<NOX::Abstract::PrePostOperator> pre_post_operator = Teuchos::rcp(new LCM::SolutionSniffer);
    Teuchos::RCP<LCM::SolutionSniffer> nox_solver_pre_post_operator =
    Teuchos::rcp_dynamic_cast<LCM::SolutionSniffer>(pre_post_operator);
    Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> statusTest =
    Teuchos::rcp_dynamic_cast<NOX::StatusTest::ModelEvaluatorFlag>(nox_status_test_);
    nox_solver_pre_post_operator->setStatusTest(statusTest);
    solverOptionsParameterList->set("User Defined Pre/Post Operator", pre_post_operator);
  }
}

#endif
