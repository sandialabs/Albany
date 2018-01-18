//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_ProblemUtils.hpp"
#include "Albany_Utils.hpp"
#include "Albany_MaterialDatabase.hpp"
#include "ConstitutiveDriverProblem.hpp"
#include "PHAL_AlbanyTraits.hpp"

//------------------------------------------------------------------------------
Albany::ConstitutiveDriverProblem::
ConstitutiveDriverProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
                          const Teuchos::RCP<ParamLib>& param_lib,
                          const int num_dims,
                          Teuchos::RCP<const Teuchos::Comm<int>>& commT) :
  Albany::AbstractProblem(params, param_lib),
  have_temperature_(false),
  use_sdbcs_(false),
  num_dims_(num_dims)
{

  std::string& method = params->get("Name", "ConstitutiveDriver");
  *out << "Problem Name = " << method << '\n';

   // Compute number of equations
  int num_eq = num_dims_ * num_dims_;
  this->setNumEquations(num_eq);

  material_db_ = Albany::createMaterialDatabase(params, commT);

  int num_PDEs = neq;
  int num_elasticity_dim = 0;
  int num_scalar = neq - num_elasticity_dim;
  int null_space_dim(0);

  rigidBodyModes->setParameters(
      num_PDEs,
      num_elasticity_dim,
      num_scalar,
      null_space_dim);

}
//------------------------------------------------------------------------------
Albany::ConstitutiveDriverProblem::
~ConstitutiveDriverProblem()
{
}
//------------------------------------------------------------------------------
void
Albany::ConstitutiveDriverProblem::
buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> meshSpecs,
    Albany::StateManager& stateMgr)
{
  // Construct All Phalanx Evaluators
  int physSets = meshSpecs.size();
  *out << "Num MeshSpecs: " << physSets << '\n';
  fm.resize(physSets);
  bool haveSidesets = false;

  *out << "Calling ConstitutiveDriverProblem::buildEvaluators" << '\n';
  for (int ps = 0; ps < physSets; ++ps) {
    fm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM,
        Teuchos::null);
    if (meshSpecs[ps]->ssNames.size() > 0) haveSidesets = true;
  }
}
//------------------------------------------------------------------------------
Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
Albany::ConstitutiveDriverProblem::
buildEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    Albany::FieldManagerChoice fmchoice,
    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<ConstitutiveDriverProblem>
    op(*this,
       fm0,
       meshSpecs,
       stateMgr,
       fmchoice,
       responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}
//------------------------------------------------------------------------------
Teuchos::RCP<const Teuchos::ParameterList>
Albany::ConstitutiveDriverProblem::
getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      this->getGenericProblemParams("ValidConstitutiveDriverProblemParams");

  validPL->set<std::string>("MaterialDB Filename",
      "materials.xml",
      "Filename of material database xml file");
  validPL->sublist("Temperature", false, "");
  validPL->sublist("Constitutive Model Driver Parameters", false, "");

  return validPL;
}
//------------------------------------------------------------------------------
void
Albany::ConstitutiveDriverProblem::
getAllocatedStates(
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> old_state,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> new_state) const
{
  old_state = old_state_;
  new_state = new_state_;
}
//------------------------------------------------------------------------------
