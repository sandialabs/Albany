//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "MechanicsProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
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
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "Unknown variable type " << type << std::endl);
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
  return "DOF";
}

//------------------------------------------------------------------------------
Albany::MechanicsProblem::
MechanicsProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
                 const Teuchos::RCP<ParamLib>& param_lib,
                 const int num_dims,
                 const Teuchos::RCP<const Epetra_Comm>& comm) :
  Albany::AbstractProblem(params, param_lib),
  have_source_(false),
  num_dims_(num_dims),
  have_mech_eq_(false),
  have_temperature_eq_(false),
  have_pressure_eq_(false),
  have_transport_eq_(false),
  have_hydrostress_eq_(false),
  have_peridynamics_(false)
{

  std::string& method = params->get("Name", "Mechanics ");
  *out << "Problem Name = " << method << std::endl;

  have_source_ =  params->isSublist("Source Functions");

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
                  pressure_type_,
                  have_pressure_,
                  have_pressure_eq_);
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

  if (have_temperature_eq_)
    have_source_ =  params->isSublist("Source Functions");

  // Compute number of equations
  int num_eq = 0;
  if (have_mech_eq_) num_eq += num_dims_;
  if (have_temperature_eq_) num_eq += 1;
  if (have_pressure_eq_) num_eq += 1;
  if (have_transport_eq_) num_eq += 1;
  if (have_hydrostress_eq_) num_eq +=1;
  this->setNumEquations(num_eq);

  // Print out a summary of the problem
  *out << "Mechanics problem:" << std::endl
       << "\tSpatial dimension:       " << num_dims_ << std::endl
       << "\tMechanics variables:     " << variableTypeToString(mech_type_)
       << std::endl
       << "\tTemperature variables:   " << variableTypeToString(temperature_type_)
       << std::endl
       << "\tPore Pressure variables: " << variableTypeToString(pressure_type_)
       << std::endl
       << "\tTransport variables:     " << variableTypeToString(transport_type_)
       << std::endl
       << "\tHydroStress variables:   " << variableTypeToString(hydrostress_type_)
       << std::endl;

  bool I_Do_Not_Have_A_Valid_Material_DB(true);
  if(params->isType<std::string>("MaterialDB Filename")){
    I_Do_Not_Have_A_Valid_Material_DB = false;
    std::string filename = params->get<std::string>("MaterialDB Filename");
    material_db_ = Teuchos::rcp(new QCAD::MaterialDatabase(filename, comm));
  }
  TEUCHOS_TEST_FOR_EXCEPTION(I_Do_Not_Have_A_Valid_Material_DB, 
                             std::logic_error,
                             "Mechanics Problem Requires a Material Database");

//the following function returns the problem information required for
//setting the rigid body modes (RBMs) for elasticity problems (in
//src/Albany_SolverFactory.cpp) written by IK, Feb. 2012

  // Need numPDEs should be num_dims_ + nDOF for other governing equations  -SS

  int num_PDEs = neq;
  int num_elasticity_dim = 0;
  if (have_mech_eq_) num_elasticity_dim = num_dims_;
  int num_scalar = neq - num_elasticity_dim;
  int null_space_dim;
  if (have_mech_eq_) {
    if (num_dims_ == 1) {null_space_dim = 0; }
    if (num_dims_ == 2) {null_space_dim = 3; }
    if (num_dims_ == 3) {null_space_dim = 6; }
  }

  rigidBodyModes->setParameters(num_PDEs, num_elasticity_dim, num_scalar, null_space_dim);
  
}
//------------------------------------------------------------------------------
Albany::MechanicsProblem::
~MechanicsProblem()
{
}
//------------------------------------------------------------------------------
void
Albany::MechanicsProblem::
buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
             Albany::StateManager& stateMgr)
{
  // Construct All Phalanx Evaluators
  int physSets = meshSpecs.size();
  std::cout << "Num MeshSpecs: " << physSets << std::endl;
  fm.resize(physSets);

  std::cout << "Calling MechanicsProblem::buildEvaluators" << std::endl;
  for (int ps=0; ps < physSets; ++ps) {
    fm[ps]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM,
                    Teuchos::null);
  }
  constructDirichletEvaluators(*meshSpecs[0]);
}
//------------------------------------------------------------------------------
Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
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
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
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
    if (neq>1) dirichletNames[index++] = "Y";
    if (neq>2) dirichletNames[index++] = "Z";
  }

  if (have_temperature_eq_) dirichletNames[index++] = "T";
  if (have_pressure_eq_) dirichletNames[index++] = "P";
  if (have_transport_eq_) dirichletNames[index++] = "C";
  if (have_hydrostress_eq_) dirichletNames[index++] = "TAU";

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
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

  return validPL;
}

void
Albany::MechanicsProblem::
getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > old_state,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > new_state
                   ) const
{
  old_state = old_state_;
  new_state = new_state_;
}
