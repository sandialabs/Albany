//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if 0
#include "Albany_ProblemUtils.hpp"
#include "Albany_Utils.hpp"
#include "LCM_Utils.h"
#include "SolidMechanics.h"
#include "PHAL_AlbanyTraits.hpp"
#include "NOXSolverPrePostOperator.h"

//------------------------------------------------------------------------------
Albany::SolidMechanics::
SolidMechanics(
    Teuchos::RCP<Teuchos::ParameterList> const & params,
    Teuchos::RCP<ParamLib> const & param_lib,
    int const num_dims,
    Teuchos::RCP<Teuchos::Comm<int> const> & comm) :
    Albany::AbstractProblem(params, param_lib),
    num_dims_(num_dims),
    have_mech_eq_(false)
{

  std::string& method = params->get("Name", "Mechanics ");
  *out << "Problem Name = " << method << '\n';

  // Is adaptation specified?
  bool adapt_sublist_exists = params->isSublist("Adaptation");

  if(adapt_sublist_exists){

    Teuchos::ParameterList const &
    adapt_params = params->sublist("Adaptation");

    std::string const &
    adaptation_method_name = adapt_params.get<std::string>("Method");

    have_sizefield_adaptation_ = (adaptation_method_name == "RPI Albany Size");

  }

  getVariableType(params->sublist("Displacement"),
      "DOF",
      mech_type_,
      have_mech_,
      have_mech_eq_);

  // Compute number of equations
  int num_eq = 0;
  if (have_mech_eq_) num_eq += num_dims_;
  this->setNumEquations(num_eq);

  // Print out a summary of the problem
  *out << "Mechanics problem:" << '\n'
      << "\tSpatial dimension             : " << num_dims_ << '\n'
      << "\tMechanics variables           : "
      << variableTypeToString(mech_type_)
      << '\n';

  material_db_ = LCM::createMaterialDatabase(params, comm);

  //the following function returns the problem information required for
  //setting the rigid body modes (RBMs) for elasticity problems (in
  //src/Albany_SolverFactory.cpp) written by IK, Feb. 2012

  // Need numPDEs should be num_dims_ + nDOF for other governing equations  -SS

  int num_PDEs = neq;
  int num_elasticity_dim = 0;
  if (have_mech_eq_) num_elasticity_dim = num_dims_;
  int num_scalar = neq - num_elasticity_dim;
  int null_space_dim(0);
  if (have_mech_eq_) {
    if (num_dims_ == 1) {
      null_space_dim = 1;
    }
    else if (num_dims_ == 2) {
      null_space_dim = 3;
    }
    else if (num_dims_ == 3) {
      null_space_dim = 6;
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          '\n' << "Error: " << __FILE__ << " line " << __LINE__ <<
          ": num_dims_ set incorrectly." << '\n');
    }
  }

  rigidBodyModes->setParameters(
      num_PDEs,
      num_elasticity_dim,
      num_scalar,
      null_space_dim);

  // Check whether we are doing adaptive insertion with topology modification.
  bool const
  have_adaptation = params->isSublist("Adaptation");

  if (have_adaptation == true) {
    Teuchos::ParameterList const &
    adapt_params = params->sublist("Adaptation");

    std::string const &
    adaptation_method_name = adapt_params.get<std::string>("Method");

    have_topmod_adaptation_ = adaptation_method_name == "Topmod";
  }

  // Create a user-defined NOX status test that can be passed to the ModelEvaluators
  userDefinedNOXStatusTest = Teuchos::rcp(new NOX::StatusTest::ModelEvaluatorFlag); 
}
//------------------------------------------------------------------------------
Albany::SolidMechanics::
~SolidMechanics()
{
}
//------------------------------------------------------------------------------
void
Albany::SolidMechanics::
buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> mesh_specs,
    Albany::StateManager& state_mgr)
{
  // Construct All Phalanx Evaluators
  int physSets = mesh_specs.size();
  *out << "Num MeshSpecs: " << physSets << '\n';
  fm.resize(physSets);
  bool haveSidesets = false;

  *out << "Calling SolidMechanics::buildEvaluators" << '\n';
  for (int ps = 0; ps < physSets; ++ps) {
    fm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *mesh_specs[ps], state_mgr, BUILD_RESID_FM,
        Teuchos::null);
    if (mesh_specs[ps]->ssNames.size() > 0) haveSidesets = true;
  }
  *out << "Calling SolidMechanics::constructDirichletEvaluators" << '\n';
  constructDirichletEvaluators(*mesh_specs[0]);

  if (haveSidesets) {
    *out << "Calling SolidMechanics::constructNeumannEvaluators" << '\n';
    constructNeumannEvaluators(mesh_specs[0]);
  }

}
//------------------------------------------------------------------------------
Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
Albany::SolidMechanics::
buildEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& field_mgr,
    const Albany::MeshSpecsStruct& mesh_specs,
    Albany::StateManager& state_mgr,
    Albany::FieldManagerChoice fm_choice,
    const Teuchos::RCP<Teuchos::ParameterList>& response_list)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *mesh_specs[0], state_mgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<SolidMechanics> op(*this,
      field_mgr,
      mesh_specs,
      state_mgr,
      fm_choice,
      response_list);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}
//------------------------------------------------------------------------------
void
Albany::SolidMechanics::
constructDirichletEvaluators(const Albany::MeshSpecsStruct& mesh_specs)
{

  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  int index = 0;
  if (have_mech_eq_) {
    dirichletNames[index++] = "X";
    if (num_dims_ > 1) dirichletNames[index++] = "Y";
    if (num_dims_ > 2) dirichletNames[index++] = "Z";
  }

  // Pass on the Application as well that is needed for
  // the coupled Schwarz BC. It is just ignored otherwise.
  Teuchos::RCP<Albany::Application> const &
  application = getApplication();

  this->params->set<Teuchos::RCP<Albany::Application>>(
      "Application", application);

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(mesh_specs.nsNames, dirichletNames,
      this->params, this->paramLib);
  offsets_ = dirUtils.getOffsets(); 

}
//------------------------------------------------------------------------------
// Traction BCs
void
Albany::SolidMechanics::
constructNeumannEvaluators(
    Albany::MeshSpecsStruct const & mesh_specs)
{
  // Note: we only enter this function if sidesets are defined in the mesh file
  // i.e. mesh_specs.ssNames.size() > 0

  Albany::BCUtils<Albany::NeumannTraits> neuUtils;

  // Check to make sure that Neumann BCs are given in the input file

  if (!neuUtils.haveBCSpecified(this->params))

  return;

  // Construct BC evaluators for all side sets and names
  // Note that the string index sets up the equation offset,
  // so ordering is important
  
  std::vector<std::string> neumannNames(neq + 1);
  Teuchos::Array<Teuchos::Array<int>> offsets;
  offsets.resize(neq + 1);

  neumannNames[0] = "sig_x";
  offsets[0].resize(1);
  offsets[0][0] = 0;
  // The Neumann BC code uses offsets[neq].size() as num dim, so use num_dims_
  // here rather than neq.
  offsets[neq].resize(num_dims_);
  offsets[neq][0] = 0;

  if (num_dims_ > 1) {
    neumannNames[1] = "sig_y";
    offsets[1].resize(1);
    offsets[1][0] = 1;
    offsets[neq][1] = 1;
  }

  if (num_dims_ > 2) {
    neumannNames[2] = "sig_z";
    offsets[2].resize(1);
    offsets[2][0] = 2;
    offsets[neq][2] = 2;
  }

  neumannNames[neq] = "all";

  // Construct BC evaluators for all possible names of conditions
  // Should only specify flux vector components (dudx, dudy, dudz),
  // or dudn, not both
  std::vector<std::string> condNames(3); //dudx, dudy, dudz, dudn, P
  Teuchos::ArrayRCP<std::string> dof_names(1);
  dof_names[0] = "Displacement";

  // Note that sidesets are only supported for two and 3D currently
  if (num_dims_ == 2)
    condNames[0] = "(t_x, t_y)";
  else if (num_dims_ == 3)
    condNames[0] = "(t_x, t_y, t_z)";
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        '\n' << "Error: Sidesets only supported in 2 and 3D." << '\n');

  condNames[1] = "dudn";
  condNames[2] = "P";

  nfm.resize(1); // Elasticity problem only has one element block

  Teuchos::RCP<Albany::MeshSpecsStruct> const &
  pmesh_specs = Teuchos::rcp(&mesh_specs, false);

  nfm[0] = neuUtils.constructBCEvaluators(
      pmesh_specs,
      neumannNames,
      dof_names,
      true,
      0,
      condNames,
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
        Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>>>
    old_state,
    Teuchos::ArrayRCP<
        Teuchos::ArrayRCP<Teuchos::RCP<Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>>>
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
  if(params->isSublist("Piro")){
    if(params->sublist("Piro").isSublist("NOX")){
      if(params->sublist("Piro").sublist("NOX").isSublist("Solver Options")){
	solverOptionsParameterList = Teuchos::rcpFromRef( params->sublist("Piro").sublist("NOX").sublist("Solver Options") );
      }
      if(params->sublist("Piro").sublist("NOX").isSublist("Status Tests")){
	statusTestsParameterList = Teuchos::rcpFromRef( params->sublist("Piro").sublist("NOX").sublist("Status Tests") );
      }
    }
  }

  if(!solverOptionsParameterList.is_null() && !statusTestsParameterList.is_null()){

    // Add the model evaulator flag as a status test.
    Teuchos::ParameterList originalStatusTestParameterList = *statusTestsParameterList;
    Teuchos::ParameterList newStatusTestParameterList;
    newStatusTestParameterList.set<std::string>("Test Type", "Combo");
    newStatusTestParameterList.set<std::string>("Combo Type", "OR");
    newStatusTestParameterList.set<int>("Number of Tests", 2);
    newStatusTestParameterList.sublist("Test 0");
    newStatusTestParameterList.sublist("Test 0").set("Test Type", "User Defined");
    newStatusTestParameterList.sublist("Test 0").set("User Status Test", userDefinedNOXStatusTest);
    newStatusTestParameterList.sublist("Test 1") = originalStatusTestParameterList;
    *statusTestsParameterList = newStatusTestParameterList;

    // Create a NOX observer that will reset the status flag at the beginning of a nonlinear solve
    Teuchos::RCP<NOX::Abstract::PrePostOperator> pre_post_operator = Teuchos::rcp(new NOXSolverPrePostOperator);
    Teuchos::RCP<NOXSolverPrePostOperator> nox_solver_pre_post_operator =
      Teuchos::rcp_dynamic_cast<NOXSolverPrePostOperator>(pre_post_operator);
    Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> statusTest =
      Teuchos::rcp_dynamic_cast<NOX::StatusTest::ModelEvaluatorFlag>(userDefinedNOXStatusTest);
    nox_solver_pre_post_operator->setStatusTest(statusTest);
    solverOptionsParameterList->set("User Defined Pre/Post Operator", pre_post_operator);
  }
}
#endif
