//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "ConcurrentMultiscaleProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"

//------------------------------------------------------------------------------
Albany::ConcurrentMultiscaleProblem::
ConcurrentMultiscaleProblem(
    Teuchos::RCP<Teuchos::ParameterList> const & params,
    Teuchos::RCP<ParamLib> const & param_lib,
    int const num_dims,
    Teuchos::RCP<const Epetra_Comm> const & comm) :
  Albany::AbstractProblem(params, param_lib),
  have_source_(false),
  num_dims_(num_dims)
{

  std::string &
  method = params->get("Name", "Mechanics ");

  *out << "Problem Name = " << method << '\n';

  bool
  invalid_material_DB(true);
  if (params->isType<std::string>("MaterialDB Filename")) {
    invalid_material_DB = false;
    std::string
    filename = params->get<std::string>("MaterialDB Filename");
    material_db_ = Teuchos::rcp(new QCAD::MaterialDatabase(filename, comm));
  }

  TEUCHOS_TEST_FOR_EXCEPTION(
      invalid_material_DB,
      std::logic_error,
      "ConcurrentMultiscale Problem Requires a Material Database"
  );


  // Compute number of equations
  int
  num_eq = num_dims_;

  this->setNumEquations(num_eq);

  //the following function returns the problem information required for
  //setting the rigid body modes (RBMs)
  int number_PDEs = neq;
  int number_elasticity_dimensions = spatialDimension();
  int number_scalar_dimensions = neq - spatialDimension();
  int null_space_dimensions = 0;

  switch (number_elasticity_dimensions) {
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::logic_error,
        "Invalid number of dimensions"
    );
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

  rigidBodyModes->setParameters(
      number_PDEs,
      number_elasticity_dimensions,
      number_scalar_dimensions,
      null_space_dimensions
  );

}

//
//
//
Albany::ConcurrentMultiscaleProblem::
~ConcurrentMultiscaleProblem()
{
}

//
//
//
void
Albany::ConcurrentMultiscaleProblem::
buildProblem(
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > mesh_specs,
    Albany::StateManager & state_mgr)
{
  // Construct All Phalanx Evaluators
  int
  physSets = mesh_specs.size();
  std::cout << "Num MeshSpecs: " << physSets << '\n';
  fm.resize(physSets);

  std::cout << "Calling ConcurrentMultiscaleProblem::buildEvaluators" << '\n';
  for (int ps = 0; ps < physSets; ++ps) {

    std::string const
    eb_name = mesh_specs[ps]->ebName;

    std::string const
    lmb_str = "Lagrange Multiplier Block";

    bool const
    is_lmb = matDB().isElementBlockParam(eb_name, lmb_str);

    if (is_lmb == true) {
      bool const
      ebp_lmb = matDB().getElementBlockParam<bool>(eb_name, lmb_str);
      lm_overlap_map_.insert(std::make_pair(eb_name, ebp_lmb));
    }

    std::string const
    cob_str = "Coarse Overlap Block";

    bool const
    is_cob = matDB().isElementBlockParam(eb_name, cob_str);

    if (is_cob == true) {
      bool const
      ebp_cob = matDB().getElementBlockParam<bool>(eb_name, cob_str);
      coarse_overlap_map_.insert(std::make_pair(eb_name, ebp_cob));
    }

    std::string const
    fob_str = "Fine Overlap Block";

    bool const
    is_fob = matDB().isElementBlockParam(eb_name, fob_str);

    if (is_fob == true) {
      bool const
      ebp_fob = matDB().getElementBlockParam<bool>(eb_name, fob_str);
      fine_overlap_map_.insert(std::make_pair(eb_name, ebp_fob));
    }
    
    fm[ps]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(
        *fm[ps],
        *mesh_specs[ps],
        state_mgr,
        BUILD_RESID_FM,
        Teuchos::null
    );
  }
  constructDirichletEvaluators(*mesh_specs[0]);
}

//
//
//
Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::ConcurrentMultiscaleProblem::
buildEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits> & fm0,
    Albany::MeshSpecsStruct const & mesh_specs,
    Albany::StateManager & state_mgr,
    Albany::FieldManagerChoice fm_choice,
    Teuchos::RCP<Teuchos::ParameterList> const & response_list)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *mesh_specs[0], state_mgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<ConcurrentMultiscaleProblem> 
    op(
        *this,
        fm0,
        mesh_specs,
        state_mgr,
        fm_choice,
        response_list
    );
  boost::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

//
//
//
void
Albany::ConcurrentMultiscaleProblem::
constructDirichletEvaluators(Albany::MeshSpecsStruct const & mesh_specs)
{

  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string>
  dirichletNames(neq);

  int
  index = 0;

  dirichletNames[index++] = "X";
  if (neq>1) dirichletNames[index++] = "Y";
  if (neq>2) dirichletNames[index++] = "Z";

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(
      mesh_specs.nsNames,
      dirichletNames,
      this->params,
      this->paramLib
  );
}
//------------------------------------------------------------------------------
Teuchos::RCP<const Teuchos::ParameterList>
Albany::ConcurrentMultiscaleProblem::
getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> valid_pl =
    this->getGenericProblemParams("ValidConcurrentMultiscaleProblemParams");

  valid_pl->set<std::string>(
      "MaterialDB Filename",
      "materials.xml",
      "Filename of material database xml file"
  );

  return valid_pl;
}

//------------------------------------------------------------------------------
void
Albany::ConcurrentMultiscaleProblem::
getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC> > > old_state,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC> > > new_state
                   ) const
{
  old_state = old_state_;
  new_state = new_state_;
}
