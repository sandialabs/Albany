//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_Macros.hpp"
#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "Albany_STKFieldContainerHelper.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

// Start of STK stuff
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>
#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

namespace Albany {

static const char* sol_tag_name[3] = {"Exodus Solution Name",
                                      "Exodus SolutionDot Name",
                                      "Exodus SolutionDotDot Name"};

static const char* sol_id_name[3] = {"solution",
                                     "solution_dot",
                                     "solution_dotdot"};

#ifdef ALBANY_DTK
static const char* sol_dtk_tag_name[3] = {"Exodus Solution DTK Name",
                                          "Exodus SolutionDot DTK Name",
                                          "Exodus SolutionDotDot DTK Name"};

static const char* sol_dtk_id_name[3] = {"solution dtk",
                                         "solution_dot dtk",
                                         "solution_dotdot dtk"};
#endif

OrdinarySTKFieldContainer::OrdinarySTKFieldContainer(
    const Teuchos::RCP<Teuchos::ParameterList>&               params_,
    const Teuchos::RCP<stk::mesh::MetaData>&                  metaData_,
    const Teuchos::RCP<stk::mesh::BulkData>&                  bulkData_,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    const DiscType                                            interleaved_,
    const int                                                 numDim_,
    const Teuchos::RCP<StateInfoStruct>&                      sis,
    const int                                                 num_params_)
    : GenericSTKFieldContainer(
          params_,
          metaData_,
          bulkData_,
          interleaved_,
          numDim_,
          num_params_)
{
  typedef typename AbstractSTKFieldContainer::VectorFieldType       VFT;
  typedef typename AbstractSTKFieldContainer::ScalarFieldType       SFT;

#ifdef ALBANY_DTK
  bool output_dtk_field =
      params_->get<bool>("Output DTK Field to Exodus", false);
#endif

  // Start STK stuff
  this->coordinates_field =
      &metaData_->declare_field<VFT>(stk::topology::NODE_RANK, "coordinates");
  stk::mesh::put_field_on_mesh(
      *this->coordinates_field, metaData_->universal_part(), numDim_, nullptr);
#ifdef ALBANY_SEACAS
  stk::io::set_field_role(*this->coordinates_field, Ioss::Field::MESH);
#endif
  if (numDim_ == 3) {
    this->coordinates_field3d = this->coordinates_field;
  } else {
    this->coordinates_field3d = &metaData_->declare_field<VFT>(
        stk::topology::NODE_RANK, "coordinates3d");
    stk::mesh::put_field_on_mesh(
        *this->coordinates_field3d, metaData_->universal_part(), 3, nullptr);
#ifdef ALBANY_SEACAS
    if (params_->get<bool>("Export 3d coordinates field", false)) {
      stk::io::set_field_role(
          *this->coordinates_field3d, Ioss::Field::TRANSIENT);
    }
#endif
  }

  this->addStateStructs(sis);

  initializeProcRankField();

}

OrdinarySTKFieldContainer::OrdinarySTKFieldContainer(
    const Teuchos::RCP<Teuchos::ParameterList>&               params_,
    const Teuchos::RCP<stk::mesh::MetaData>&                  metaData_,
    const Teuchos::RCP<stk::mesh::BulkData>&                  bulkData_,
    const DiscType                                            interleaved_,
    const int                                                 neq_,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    const int                                                 numDim_,
    const Teuchos::RCP<StateInfoStruct>&                      sis,
    const int                                                 num_params_)
    : GenericSTKFieldContainer(
          params_,
          metaData_,
          bulkData_,
          interleaved_,
          neq_,
          numDim_,
          num_params_)
{
  typedef typename AbstractSTKFieldContainer::VectorFieldType       VFT;
  typedef typename AbstractSTKFieldContainer::ScalarFieldType       SFT;

  int num_time_deriv = params_->get<int>("Number Of Time Derivatives");
#ifdef ALBANY_DTK
  bool output_dtk_field =
      params_->get<bool>("Output DTK Field to Exodus", false);
#endif
  //IKT FIXME? - currently won't write dxdp to output file if problem is steady,
  //as this output doesn't work in same way.  May want to change in the future.
  bool output_sens_field = false;
  const std::string sens_method = params_->get<std::string>("Sensitivity Method"); 
  const int resp_fn_index = params_->get<int>("Response Function Index"); 
  const int param_sens_index = params_->get<int>("Sensitivity Parameter Index"); 

  if (this->num_params > 0 && num_time_deriv > 0 && sens_method != "None") output_sens_field = true;

  //Create tag and id arrays for sensitivity field (dxdp or dgdp)
  std::vector<std::string> sol_sens_tag_name_vec;
  std::vector<std::string> sol_sens_id_name_vec;
  if (sens_method == "Forward") {
    sol_sens_tag_name_vec.resize(this->num_params);
    sol_sens_id_name_vec.resize(this->num_params);
    for (int np=0; np<this->num_params; np++) {
      sol_sens_tag_name_vec[np] = "Exodus Solution Sensitivity Name" + std::to_string(np);
      sol_sens_id_name_vec[np] = "sensitivity dxdp" + std::to_string(np); 
    }
  }
  else if (sens_method == "Adjoint") {
    //Adjoint sensitivities can only be computed for 1 response/parameter at a time.
    sol_sens_tag_name_vec.resize(1);
    sol_sens_id_name_vec.resize(1);
    sol_sens_tag_name_vec[0] = "Exodus Solution Sensitivity Name";
    //WARNING IKT 8/24/2021: I am not sure that the following will do the right thing in the case the parameter
    //p is not defined on the entire mesh.  A different way of observing dgdp may need to be implemented
    //for that case.  Also note that dgdp will not be written correctly to the mesh for the case of a scalar (vs. distributed) parameter.
    sol_sens_id_name_vec[0] = "sensitivity dg" + std::to_string(resp_fn_index) + "dp" + std::to_string(param_sens_index); 
  }

  solution_field.resize(num_time_deriv + 1);
  solution_field_dtk.resize(num_time_deriv + 1);
  solution_field_dxdp.resize(this->num_params);

  for (int num_vecs = 0; num_vecs <= num_time_deriv; num_vecs++) {
    solution_field[num_vecs] = &metaData_->declare_field<VFT>(
        stk::topology::NODE_RANK,
        params_->get<std::string>(
            sol_tag_name[num_vecs], sol_id_name[num_vecs]));
    stk::mesh::put_field_on_mesh(
        *solution_field[num_vecs], metaData_->universal_part(), neq_, nullptr); // KL: this

#if defined(ALBANY_DTK)
    if (output_dtk_field == true) {
      solution_field_dtk[num_vecs] = &metaData_->declare_field<VFT>(
          stk::topology::NODE_RANK,
          params_->get<std::string>(
              sol_dtk_tag_name[num_vecs], sol_dtk_id_name[num_vecs]));
      stk::mesh::put_field_on_mesh(
          *solution_field_dtk[num_vecs],
          metaData_->universal_part(),
          neq_,
          nullptr);
    }
#endif

#ifdef ALBANY_SEACAS
    stk::io::set_field_role(*solution_field[num_vecs], Ioss::Field::TRANSIENT);
#if defined(ALBANY_DTK)
    if (output_dtk_field == true)
      stk::io::set_field_role(
          *solution_field_dtk[num_vecs], Ioss::Field::TRANSIENT);
#endif
#endif
  }

  //Transient sensitivities output to Exodus
  const int num_sens = (sens_method == "Forward") ? this->num_params : 1; 
  for (int np = 0; np < num_sens; np++) {
    if (output_sens_field == true) {
      solution_field_dxdp[np] = &metaData_->declare_field<VFT>(
          stk::topology::NODE_RANK,
          params_->get<std::string>(
              sol_sens_tag_name_vec[np], sol_sens_id_name_vec[np]));
      stk::mesh::put_field_on_mesh(
          *solution_field_dxdp[np],
          metaData_->universal_part(),
          neq_,
          nullptr);
    }
#ifdef ALBANY_SEACAS
    if (output_sens_field == true)
      stk::io::set_field_role(
          *solution_field_dxdp[np], Ioss::Field::TRANSIENT);
#endif
  }

  this->addStateStructs(sis);

  //initializeProcRankField();

}

void
OrdinarySTKFieldContainer::initializeProcRankField()
{
  using ISFT = AbstractSTKFieldContainer::IntScalarFieldType;
  using SFT  = AbstractSTKFieldContainer::ScalarFieldType;

  this->proc_rank_field = &this->metaData->template declare_field<ISFT>(
      stk::topology::ELEMENT_RANK, "proc_rank");

  // Processor rank field, a scalar
  stk::mesh::put_field_on_mesh(
      *this->proc_rank_field, this->metaData->universal_part(), nullptr);

#ifdef ALBANY_SEACAS
  stk::io::set_field_role(*this->proc_rank_field, Ioss::Field::MESH);
#endif
}

void
OrdinarySTKFieldContainer::fillVector(
    Thyra_Vector&                                field_vector,
    const std::string&                           field_name,
    stk::mesh::Selector&                         field_selection,
    const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
    const NodalDOFManager&                       nodalDofManager)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    !(this->solutionFieldContainer),
    std::logic_error,
    "Error OrdinarySTKFieldContainer::fillVector not called from a solution field container."
    << std::endl);

  fillVectorImpl(
      field_vector,
      field_name,
      field_selection,
      field_node_vs,
      nodalDofManager);
}

void
OrdinarySTKFieldContainer::fillSolnVector(
    Thyra_Vector&                                solution,
    stk::mesh::Selector&                         sel,
    const Teuchos::RCP<const Thyra_VectorSpace>& node_vs)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    !(this->solutionFieldContainer),
    std::logic_error,
    "Error OrdinarySTKFieldContainer::fillSolnVector not called from a solution field container."
    << std::endl);

  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, interleaved);

  fillVectorImpl(
      solution, solution_field[0]->name(), sel, node_vs, nodalDofManager);
}

void
OrdinarySTKFieldContainer::fillSolnMultiVector(
    Thyra_MultiVector&                           solution,
    stk::mesh::Selector&                         sel,
    const Teuchos::RCP<const Thyra_VectorSpace>& node_vs)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    !(this->solutionFieldContainer),
    std::logic_error,
    "Error OrdinarySTKFieldContainer::fillSolnMultiVector not called from a solution field container."
    << std::endl);

  typedef typename AbstractSTKFieldContainer::VectorFieldType VFT;

  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, interleaved);

  for (int icomp = 0; icomp < solution.domain()->dim(); ++icomp) {
    fillVectorImpl(
        *solution.col(icomp),
        solution_field[icomp]->name(),
        sel,
        node_vs,
        nodalDofManager);
  }
}

void
OrdinarySTKFieldContainer::saveVector(
    const Thyra_Vector&                          field_vector,
    const std::string&                           field_name,
    stk::mesh::Selector&                         field_selection,
    const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
    const NodalDOFManager&                       nodalDofManager)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    !(this->solutionFieldContainer),
    std::logic_error,
    "Error OrdinarySTKFieldContainer::saveVector not called from a solution field container."
    << std::endl);

  saveVectorImpl(
      field_vector,
      field_name,
      field_selection,
      field_node_vs,
      nodalDofManager);
}

void
OrdinarySTKFieldContainer::saveSolnVector(
    const Thyra_Vector&                          solution,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    stk::mesh::Selector&                         sel,
    const Teuchos::RCP<const Thyra_VectorSpace>& node_vs)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    !(this->solutionFieldContainer),
    std::logic_error,
    "Error OrdinarySTKFieldContainer::saveSolnVector not called from a solution field container."
    << std::endl);

  // IKT, FIXME? throw exception if num_time_deriv == 0 and we are calling this
  // function?

  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, interleaved);

  saveVectorImpl(
      solution, solution_field[0]->name(), sel, node_vs, nodalDofManager);

  if (soln_dxdp != Teuchos::null) {
    if (soln_dxdp->domain()->dim() != this->num_params) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::runtime_error,
          "Error in saveSolnVector! Number of vectors in soln_dxdp (" << soln_dxdp->domain()->dim()
	  << ") != num_params (" << this->num_params << ").\n");
    }
    for (int np = 0; np < this->num_params; np++) {
      saveVectorImpl(
        *soln_dxdp->col(np), solution_field_dxdp[np]->name(), sel, node_vs, nodalDofManager);
    }
  }

}

void
OrdinarySTKFieldContainer::saveSolnVector(
    const Thyra_Vector&                          solution,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector&                          solution_dot,
    stk::mesh::Selector&                         sel,
    const Teuchos::RCP<const Thyra_VectorSpace>& node_vs)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    !(this->solutionFieldContainer),
    std::logic_error,
    "Error OrdinarySTKFieldContainer::saveSolnVector not called from a solution field container."
    << std::endl);

  // IKT, FIXME? throw exception if num_time_deriv == 0 and we are calling this
  // function?

  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, interleaved);

  saveVectorImpl(
      solution, solution_field[0]->name(), sel, node_vs, nodalDofManager);
  saveVectorImpl(
      solution_dot, solution_field[1]->name(), sel, node_vs, nodalDofManager);

  if (soln_dxdp != Teuchos::null) {
    if (soln_dxdp->domain()->dim() != this->num_params) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::runtime_error,
          "Error in saveSolnVector! Number of vectors in soln_dxdp (" << soln_dxdp->domain()->dim()
	  << ") != num_params (" << this->num_params << ").\n");
    }
    for (int np = 0; np < this->num_params; np++) {
      saveVectorImpl(
        *soln_dxdp->col(np), solution_field_dxdp[np]->name(), sel, node_vs, nodalDofManager);
    }
  }
}

void
OrdinarySTKFieldContainer::saveSolnVector(
    const Thyra_Vector&                          solution,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector&                          solution_dot,
    const Thyra_Vector&                          solution_dotdot,
    stk::mesh::Selector&                         sel,
    const Teuchos::RCP<const Thyra_VectorSpace>& node_vs)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    !(this->solutionFieldContainer),
    std::logic_error,
    "Error OrdinarySTKFieldContainer::saveSolnVector not called from a solution field container."
    << std::endl);

  // IKT, FIXME? throw exception if num_time_deriv < 2 and we are calling this
  // function?

  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, interleaved);

  saveVectorImpl(
      solution, solution_field[0]->name(), sel, node_vs, nodalDofManager);
  saveVectorImpl(
      solution_dot, solution_field[1]->name(), sel, node_vs, nodalDofManager);
  saveVectorImpl(
      solution_dotdot,
      solution_field[2]->name(),
      sel,
      node_vs,
      nodalDofManager);

  if (soln_dxdp != Teuchos::null) {
    if (soln_dxdp->domain()->dim() != this->num_params) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::runtime_error,
          "Error in saveSolnVector! Number of vectors in soln_dxdp (" << soln_dxdp->domain()->dim()
	  << ") != num_params (" << this->num_params << ").\n");
    }
    for (int np = 0; np < this->num_params; np++) {
      saveVectorImpl(
        *soln_dxdp->col(np), solution_field_dxdp[np]->name(), sel, node_vs, nodalDofManager);
    }
  }
}

void
OrdinarySTKFieldContainer::saveSolnMultiVector(
    const Thyra_MultiVector&                     solution,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    stk::mesh::Selector&                         sel,
    const Teuchos::RCP<const Thyra_VectorSpace>& node_vs)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    !(this->solutionFieldContainer),
    std::logic_error,
    "Error OrdinarySTKFieldContainer::saveSolnMultiVector not called from a solution field container."
    << std::endl);

  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, interleaved);

  for (int icomp = 0; icomp < solution.domain()->dim(); ++icomp) {
    saveVectorImpl(
        *solution.col(icomp),
        solution_field[icomp]->name(),
        sel,
        node_vs,
        nodalDofManager);
  }

  if (soln_dxdp != Teuchos::null) {
    if (soln_dxdp->domain()->dim() != this->num_params) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::runtime_error,
          "Error in saveSolnVector! Number of vectors in soln_dxdp (" << soln_dxdp->domain()->dim()
	  << ") != num_params (" << this->num_params << ").\n");
    }
    for (int np = 0; np < this->num_params; np++) {
      saveVectorImpl(
        *soln_dxdp->col(np), solution_field_dxdp[np]->name(), sel, node_vs, nodalDofManager);
    }
  }
}

void
OrdinarySTKFieldContainer::saveResVector(
    const Thyra_Vector&                          res,
    stk::mesh::Selector&                         sel,
    const Teuchos::RCP<const Thyra_VectorSpace>& node_vs)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    !(this->solutionFieldContainer),
    std::logic_error,
    "Error OrdinarySTKFieldContainer::saveResVector not called from a solution field container."
    << std::endl);

  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, interleaved);

  saveVectorImpl(res, residual_field->name(), sel, node_vs, nodalDofManager);
}

void
OrdinarySTKFieldContainer::transferSolutionToCoords()
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    !(this->solutionFieldContainer),
    std::logic_error,
    "Error OrdinarySTKFieldContainer::transferSolutionToCoords not called from a solution field container."
    << std::endl);

  using VFT    = typename AbstractSTKFieldContainer::VectorFieldType;
  using Helper = STKFieldContainerHelper<VFT>;
  Helper::copySTKField(*solution_field[0], *this->coordinates_field);
}

void
OrdinarySTKFieldContainer::fillVectorImpl(
    Thyra_Vector&                                field_vector,
    const std::string&                           field_name,
    stk::mesh::Selector&                         field_selection,
    const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
    const NodalDOFManager&                       nodalDofManager)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    !(this->solutionFieldContainer),
    std::logic_error,
    "Error OrdinarySTKFieldContainer::fillVectorImpl not called from a solution field container."
    << std::endl);

  using VFT = typename AbstractSTKFieldContainer::VectorFieldType;
  using SFT = typename AbstractSTKFieldContainer::ScalarFieldType;

  auto* raw_field =
      this->metaData->get_field(stk::topology::NODE_RANK, field_name);
  ALBANY_EXPECT(
      raw_field != nullptr,
      "Error! Something went wrong while retrieving a field.\n");
  const int rank = raw_field->field_array_rank();

  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  stk::mesh::BucketVector const& all_elements =
      this->bulkData->get_buckets(stk::topology::NODE_RANK, field_selection);

  auto field_node_vs_indexer = createGlobalLocalIndexer(field_node_vs);
  if (rank == 0) {
    const SFT* field = this->metaData->template get_field<SFT>(
        stk::topology::NODE_RANK, field_name);
    using Helper = STKFieldContainerHelper<SFT>;
    for (stk::mesh::BucketVector::const_iterator it = all_elements.begin();
         it != all_elements.end();
         ++it) {
      const stk::mesh::Bucket& bucket = **it;
      Helper::fillVector(
          field_vector, *field, field_node_vs_indexer, bucket, nodalDofManager, 0);
    }
  } else if (rank == 1) {
    const VFT* field = this->metaData->template get_field<VFT>(
        stk::topology::NODE_RANK, field_name);
    using Helper = STKFieldContainerHelper<VFT>;
    for (stk::mesh::BucketVector::const_iterator it = all_elements.begin();
         it != all_elements.end();
         ++it) {
      const stk::mesh::Bucket& bucket = **it;
      Helper::fillVector(
          field_vector, *field, field_node_vs_indexer, bucket, nodalDofManager, 0);
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::runtime_error,
        "Error! Only scalar and vector fields supported so far.\n");
  }
}

void
OrdinarySTKFieldContainer::saveVectorImpl(
    const Thyra_Vector&                          field_vector,
    const std::string&                           field_name,
    stk::mesh::Selector&                         field_selection,
    const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
    const NodalDOFManager&                       nodalDofManager)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    !(this->solutionFieldContainer),
    std::logic_error,
    "Error OrdinarySTKFieldContainer::saveVectorImpl not called from a solution field container."
    << std::endl);

  using VFT = typename AbstractSTKFieldContainer::VectorFieldType;
  using SFT = typename AbstractSTKFieldContainer::ScalarFieldType;

  auto* raw_field =
      this->metaData->get_field(stk::topology::NODE_RANK, field_name);
  ALBANY_EXPECT(
      raw_field != nullptr,
      "Error! Something went wrong while retrieving a field.\n");
  const int rank = raw_field->field_array_rank();

  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  stk::mesh::BucketVector const& all_elements =
      this->bulkData->get_buckets(stk::topology::NODE_RANK, field_selection);

  auto field_node_vs_indexer = createGlobalLocalIndexer(field_node_vs);
  if (rank == 0) {
    SFT* field = this->metaData->template get_field<SFT>(
        stk::topology::NODE_RANK, field_name);
    using Helper = STKFieldContainerHelper<SFT>;
    for (auto it : all_elements) {
      const stk::mesh::Bucket& bucket = *it;
      Helper::saveVector(
          field_vector, *field, field_node_vs_indexer, bucket, nodalDofManager, 0);
    }
  } else if (rank == 1) {
    VFT* field = this->metaData->template get_field<VFT>(
        stk::topology::NODE_RANK, field_name);
    using Helper = STKFieldContainerHelper<VFT>;
    for (auto it : all_elements) {
      const stk::mesh::Bucket& bucket = *it;
      Helper::saveVector(
          field_vector, *field, field_node_vs_indexer, bucket, nodalDofManager, 0);
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::runtime_error,
        "Error! Only scalar and vector fields supported so far.\n");
  }
}

}  // namespace Albany
