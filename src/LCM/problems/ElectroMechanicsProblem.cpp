//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ProblemUtils.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_MaterialDatabase.hpp"
#include "ElectroMechanicsProblem.hpp"
#include "PHAL_AlbanyTraits.hpp"


//------------------------------------------------------------------------------
Albany::ElectroMechanicsProblem::
ElectroMechanicsProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
    const Teuchos::RCP<ParamLib>& param_lib,
    const int num_dims,
    Teuchos::RCP<const Teuchos::Comm<int>>& commT) :
    Albany::AbstractProblem(params, param_lib),
    num_dims_(num_dims)
{

  std::string& method = params->get("Name", "ElectroMechanics ");
  *out << "Problem Name = " << method << '\n';

  // Compute number of equations
  int num_eq = num_dims_ + 1;
  this->setNumEquations(num_eq);

  material_db_ = Albany::createMaterialDatabase(params, commT);

  //the following function returns the problem information required for
  //setting the rigid body modes (RBMs) for elasticity problems (in
  //src/Albany_SolverFactory.cpp) written by IK, Feb. 2012

  // Need numPDEs should be num_dims_ + nDOF for other governing equations  -SS

  int num_PDEs = neq;
  int num_scalar = neq - num_dims_;
  int null_space_dim(0);
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

  rigidBodyModes->setParameters(
      num_PDEs,
      num_dims_,
      num_scalar,
      null_space_dim);
}
//------------------------------------------------------------------------------
Albany::ElectroMechanicsProblem::
~ElectroMechanicsProblem()
{
}
//------------------------------------------------------------------------------
void
Albany::ElectroMechanicsProblem::
buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> meshSpecs,
    Albany::StateManager& stateMgr)
{
  // Construct All Phalanx Evaluators
  int physSets = meshSpecs.size();
  *out << "Num MeshSpecs: " << physSets << '\n';
  fm.resize(physSets);
  bool haveSidesets = false;

  *out << "Calling ElectroMechanicsProblem::buildEvaluators" << '\n';
  for (int ps = 0; ps < physSets; ++ps) {
    fm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM,
        Teuchos::null);
    if (meshSpecs[ps]->ssNames.size() > 0) haveSidesets = true;
  }
  constructDirichletEvaluators(*meshSpecs[0]);

  if (haveSidesets)

  constructNeumannEvaluators(meshSpecs[0]);

}
//------------------------------------------------------------------------------
Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
Albany::ElectroMechanicsProblem::
buildEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    Albany::FieldManagerChoice fmchoice,
    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<ElectroMechanicsProblem> op(*this,
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
Albany::ElectroMechanicsProblem::
constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs)
{

  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  int index = 0;
  dirichletNames[index++] = "X";
  if (num_dims_ > 1) dirichletNames[index++] = "Y";
  if (num_dims_ > 2) dirichletNames[index++] = "Z";

  dirichletNames[index++] = "Potential";

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
      this->params, this->paramLib);
  offsets_ = dirUtils.getOffsets();
  nodeSetIDs_ = dirUtils.getNodeSetIDs();
}
//------------------------------------------------------------------------------
// Traction BCs
void
Albany::ElectroMechanicsProblem::
constructNeumannEvaluators(
    const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
  // Note: we only enter this function if sidesets are defined in the mesh file
  // i.e. meshSpecs.ssNames.size() > 0

  Albany::BCUtils<Albany::NeumannTraits> neuUtils;

  // Check to make sure that Neumann BCs are given in the input file

  if (!neuUtils.haveBCSpecified(this->params)) return;

  // Construct BC evaluators for all side sets and names
  // Note that the string index sets up the equation offset,
  // so ordering is important
  std::vector<std::string> neumannNames(neq + 1);
  Teuchos::Array<Teuchos::Array<int>> offsets;
  offsets.resize(neq + 1);

  neumannNames[0] = "sig_x";
  offsets[0].resize(1);
  offsets[0][0] = 0;
  offsets[neq].resize(neq);
  offsets[neq][0] = 0;

  if (neq > 1) {
    neumannNames[1] = "sig_y";
    offsets[1].resize(1);
    offsets[1][0] = 1;
    offsets[neq][1] = 1;
  }

  if (neq > 2) {
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
      meshSpecs, neumannNames, dof_names, true, 0, condNames,
      offsets, dl_, this->params, this->paramLib);

}
//------------------------------------------------------------------------------
Teuchos::RCP<const Teuchos::ParameterList>
Albany::ElectroMechanicsProblem::
getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      this->getGenericProblemParams("ValidElectroMechanicsProblemParams");

  validPL->set<std::string>("MaterialDB Filename",
      "materials.xml",
      "Filename of material database xml file");

  return validPL;
}

void
Albany::ElectroMechanicsProblem::
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
