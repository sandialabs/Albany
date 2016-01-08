//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ProblemUtils.hpp"
#include "Albany_Utils.hpp"
#include "LCM_Utils.h"
#include "MechanicsProblem.hpp"
#include "PHAL_AlbanyTraits.hpp"

void
Albany::MechanicsProblem::
getVariableType(Teuchos::ParameterList& param_list,
    const std::string& default_type,
    Albany::MechanicsProblem::MECH_VAR_TYPE& variable_type,
    bool& have_variable,
    bool& have_equation)
{
  std::string type = param_list.get("Variable Type", default_type);
  if (type == "None")
    variable_type = MECH_VAR_TYPE_NONE;
  else if (type == "Constant")
    variable_type = MECH_VAR_TYPE_CONSTANT;
  else if (type == "DOF")
    variable_type = MECH_VAR_TYPE_DOF;
  else if (type == "Time Dependent")
    variable_type = MECH_VAR_TYPE_TIMEDEP;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Unknown variable type " << type << '\n');
  have_variable = (variable_type != MECH_VAR_TYPE_NONE);
  have_equation = (variable_type == MECH_VAR_TYPE_DOF);
}
//------------------------------------------------------------------------------
std::string
Albany::MechanicsProblem::
variableTypeToString(Albany::MechanicsProblem::MECH_VAR_TYPE variable_type)
{
  if (variable_type == MECH_VAR_TYPE_NONE)
    return "None";
  else if (variable_type == MECH_VAR_TYPE_CONSTANT)
    return "Constant";
  else if (variable_type == MECH_VAR_TYPE_TIMEDEP)
    return "Time Dependent";
  return "DOF";
}

//------------------------------------------------------------------------------
Albany::MechanicsProblem::
MechanicsProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
    const Teuchos::RCP<ParamLib>& param_lib,
    const int num_dims,
    const Teuchos::RCP<AAdapt::rc::Manager>& rc_mgr,
    Teuchos::RCP<const Teuchos::Comm<int>>& commT) :
    Albany::AbstractProblem(params, param_lib),
    have_source_(false),
    thermal_source_(SOURCE_TYPE_NONE),
    thermal_source_evaluated_(false),
    have_contact_(false),
    num_dims_(num_dims),
    have_mech_eq_(false),
    have_temperature_eq_(false),
    have_pore_pressure_eq_(false),
    have_transport_eq_(false),
    have_hydrostress_eq_(false),
    have_damage_eq_(false),
    have_stab_pressure_eq_(false),
    have_peridynamics_(false),
    have_topmod_adaptation_(false),
    have_sizefield_adaptation_(false),
    rc_mgr_(rc_mgr)
{

  std::string& method = params->get("Name", "Mechanics ");
  *out << "Problem Name = " << method << '\n';

  // Are any source functions specified?
  have_source_ = params->isSublist("Source Functions");

  // Is contact specified?
  have_contact_ = params->isSublist("Contact");

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
  getVariableType(params->sublist("Temperature"),
      "None",
      temperature_type_,
      have_temperature_,
      have_temperature_eq_);
  getVariableType(params->sublist("Pore Pressure"),
      "None",
      pore_pressure_type_,
      have_pore_pressure_,
      have_pore_pressure_eq_);
  getVariableType(params->sublist("Transport"),
      "None",
      transport_type_,
      have_transport_,
      have_transport_eq_);
  getVariableType(params->sublist("HydroStress"),
      "None",
      hydrostress_type_,
      have_hydrostress_,
      have_hydrostress_eq_);
  getVariableType(params->sublist("Damage"),
      "None",
      damage_type_,
      have_damage_,
      have_damage_eq_);
  getVariableType(params->sublist("Stabilized Pressure"),
      "None",
      stab_pressure_type_,
      have_stab_pressure_,
      have_stab_pressure_eq_);

  // Compute number of equations
  int num_eq = 0;
  if (have_mech_eq_) num_eq += num_dims_;
  if (have_temperature_eq_) num_eq += 1;
  if (have_pore_pressure_eq_) num_eq += 1;
  if (have_transport_eq_) num_eq += 1;
  if (have_hydrostress_eq_) num_eq += 1;
  if (have_damage_eq_) num_eq += 1;
  if (have_stab_pressure_eq_) num_eq += 1;
  this->setNumEquations(num_eq);

  // Print out a summary of the problem
  *out << "Mechanics problem:" << '\n'
      << "\tSpatial dimension             : " << num_dims_ << '\n'
      << "\tMechanics variables           : "
      << variableTypeToString(mech_type_)
      << '\n'
      << "\tTemperature variables         : "
      << variableTypeToString(temperature_type_)
      << '\n'
      << "\tPore Pressure variables       : "
      << variableTypeToString(pore_pressure_type_)
      << '\n'
      << "\tTransport variables           : "
      << variableTypeToString(transport_type_)
      << '\n'
      << "\tHydroStress variables         : "
      << variableTypeToString(hydrostress_type_)
      << '\n'
      << "\tDamage variables              : "
      << variableTypeToString(damage_type_)
      << '\n'
      << "\tStabilized Pressure variables : "
      << variableTypeToString(stab_pressure_type_)
      << '\n';

  material_db_ = LCM::createMaterialDatabase(params, commT);

  // Determine the Thermal source 
  //   - the "Source Functions" list must be present in the input file,
  //   - we must have temperature and have included a temperature equation

  if (have_source_ && have_temperature_ && have_temperature_eq_) {
    // If a thermal source is specified
    if (params->sublist("Source Functions").isSublist("Thermal Source")) {

      Teuchos::ParameterList& thSrcPL = params->sublist("Source Functions")
          .sublist("Thermal Source");

      if (thSrcPL.get<std::string>("Thermal Source Type", "None")
          == "Block Dependent") {

        if (Teuchos::nonnull(material_db_)) {
          thermal_source_ = SOURCE_TYPE_MATERIAL;
        }
      }
      else {

        thermal_source_ = SOURCE_TYPE_INPUT;

      }
    }
  }

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
      null_space_dim = 0;
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

}
//------------------------------------------------------------------------------
Albany::MechanicsProblem::
~MechanicsProblem()
{
}
//------------------------------------------------------------------------------
void
Albany::MechanicsProblem::
buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> meshSpecs,
    Albany::StateManager& stateMgr)
{
  // Construct All Phalanx Evaluators
  int physSets = meshSpecs.size();
  *out << "Num MeshSpecs: " << physSets << '\n';
  fm.resize(physSets);
  bool haveSidesets = false;

  *out << "Calling MechanicsProblem::buildEvaluators" << '\n';
  for (int ps = 0; ps < physSets; ++ps) {
    fm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM,
        Teuchos::null);
    if (meshSpecs[ps]->ssNames.size() > 0) haveSidesets = true;
  }
  *out << "Calling MechanicsProblem::constructDirichletEvaluators" << '\n';
  constructDirichletEvaluators(*meshSpecs[0]);

  if (haveSidesets) {
    *out << "Calling MechanicsProblem::constructDirichletEvaluators" << '\n';
    constructNeumannEvaluators(meshSpecs[0]);
  }

}
//------------------------------------------------------------------------------
Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
Albany::MechanicsProblem::
buildEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    Albany::FieldManagerChoice fmchoice,
    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<MechanicsProblem> op(*this,
      fm0,
      meshSpecs,
      stateMgr,
      fmchoice,
      responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}
//------------------------------------------------------------------------------
void
Albany::MechanicsProblem::
constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs)
{

  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  int index = 0;
  if (have_mech_eq_) {
    dirichletNames[index++] = "X";
    if (num_dims_ > 1) dirichletNames[index++] = "Y";
    if (num_dims_ > 2) dirichletNames[index++] = "Z";
  }

  if (have_temperature_eq_) dirichletNames[index++] = "T";
  if (have_pore_pressure_eq_) dirichletNames[index++] = "P";
  if (have_transport_eq_) dirichletNames[index++] = "C";
  if (have_hydrostress_eq_) dirichletNames[index++] = "TAU";
  if (have_damage_eq_) dirichletNames[index++] = "D";
  if (have_stab_pressure_eq_) dirichletNames[index++] = "SP";

  // Pass on the Application as well that is needed for
  // the coupled Schwarz BC. It is just ignored otherwise.
  Teuchos::RCP<Albany::Application> const &
  application = getApplication();

  this->params->set<Teuchos::RCP<Albany::Application>>(
      "Application", application);

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
      this->params, this->paramLib);

}
//------------------------------------------------------------------------------
// Traction BCs
void
Albany::MechanicsProblem::
constructNeumannEvaluators(
    const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
  // Note: we only enter this function if sidesets are defined in the mesh file
  // i.e. meshSpecs.ssNames.size() > 0

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

  nfm[0] = neuUtils.constructBCEvaluators(
      meshSpecs,
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
Albany::MechanicsProblem::
getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      this->getGenericProblemParams("ValidMechanicsProblemParams");

  validPL->set<std::string>("MaterialDB Filename",
      "materials.xml",
      "Filename of material database xml file");
  validPL->sublist("Displacement", false, "");
  validPL->sublist("Temperature", false, "");
  validPL->sublist("Pore Pressure", false, "");
  validPL->sublist("Transport", false, "");
  validPL->sublist("HydroStress", false, "");
  validPL->sublist("Damage", false, "");
  validPL->sublist("Stabilized Pressure", false, "");

  return validPL;
}

//------------------------------------------------------------------------------
void
Albany::MechanicsProblem::
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
