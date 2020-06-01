//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>
#include <string>

#include "Albany_Macros.hpp"
#include "Albany_MultiSTKFieldContainer.hpp"
#include "Albany_STKFieldContainerHelper.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

// Start of STK stuff
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>
#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

#include "Teuchos_VerboseObject.hpp"

namespace Albany {

static const char* sol_tag_name[3] = {"Exodus Solution Name",
                                      "Exodus SolutionDot Name",
                                      "Exodus SolutionDotDot Name"};

static const char* sol_id_name[3] = {"solution",
                                     "solution_dot",
                                     "solution_dotdot"};

static const char* res_tag_name = {
    "Exodus Residual Name",
};

static const char* res_id_name = {
    "residual",
};

template <bool Interleaved>
MultiSTKFieldContainer<Interleaved>::MultiSTKFieldContainer(
    const Teuchos::RCP<Teuchos::ParameterList>& params_,
    const Teuchos::RCP<stk::mesh::MetaData>&    metaData_,
    const Teuchos::RCP<stk::mesh::BulkData>&    bulkData_,
    const int                                   neq_,
    const int                                          numDim_,
    const Teuchos::RCP<StateInfoStruct>&               sis,
    const Teuchos::Array<Teuchos::Array<std::string>>& solution_vector, 
    const int                                          num_params)
    : GenericSTKFieldContainer<Interleaved>(
          params_,
          metaData_,
          bulkData_,
          neq_,
          numDim_)
{
  typedef typename AbstractSTKFieldContainer::VectorFieldType       VFT;
  typedef typename AbstractSTKFieldContainer::ScalarFieldType       SFT;

  sol_vector_name.resize(solution_vector.size());
  sol_index.resize(solution_vector.size());

  // Check the input

  auto const num_derivs = solution_vector[0].size();
  for (auto i = 1; i < solution_vector.size(); ++i) {
    ALBANY_ASSERT(
        solution_vector[i].size() == num_derivs,
        "\n*** ERROR ***\n"
        "Number of derivatives for each variable is different.\n"
        "Check definition of solution vector and its derivatives.\n");
  }

  for (int vec_num = 0; vec_num < solution_vector.size(); vec_num++) {
    if (solution_vector[vec_num].size() ==
        0) {  // Do the default solution vector

      std::string name = params_->get<std::string>(
          sol_tag_name[vec_num], sol_id_name[vec_num]);
      VFT* solution =
          &metaData_->declare_field<VFT>(stk::topology::NODE_RANK, name);
      stk::mesh::put_field_on_mesh(
          *solution, metaData_->universal_part(), neq_, nullptr);
#ifdef ALBANY_SEACAS
      stk::io::set_field_role(*solution, Ioss::Field::TRANSIENT);
#endif

      sol_vector_name[vec_num].push_back(name);
      sol_index[vec_num].push_back(this->neq);
    } else if (solution_vector[vec_num].size() == 1) {  // User is just renaming
                                                        // the entire solution
                                                        // vector

      VFT* solution = &metaData_->declare_field<VFT>(
          stk::topology::NODE_RANK, solution_vector[vec_num][0]);
      stk::mesh::put_field_on_mesh(
          *solution, metaData_->universal_part(), neq_, nullptr);
#ifdef ALBANY_SEACAS
      stk::io::set_field_role(*solution, Ioss::Field::TRANSIENT);
#endif

      sol_vector_name[vec_num].push_back(solution_vector[vec_num][0]);
      sol_index[vec_num].push_back(neq_);

    } else {  // user is breaking up the solution into multiple fields

      // make sure the number of entries is even

      TEUCHOS_TEST_FOR_EXCEPTION(
          (solution_vector[vec_num].size() % 2),
          std::logic_error,
          "Error in input file: specification of solution vector layout is "
          "incorrect."
              << std::endl);

      int len, accum = 0;

      for (int i = 0; i < solution_vector[vec_num].size(); i += 2) {
        if (solution_vector[vec_num][i + 1] == "V") {
          len = numDim_;  // vector
          accum += len;
          VFT* solution = &metaData_->declare_field<VFT>(
              stk::topology::NODE_RANK, solution_vector[vec_num][i]);
          stk::mesh::put_field_on_mesh(
              *solution, metaData_->universal_part(), len, nullptr);
#ifdef ALBANY_SEACAS
          stk::io::set_field_role(*solution, Ioss::Field::TRANSIENT);
#endif
          sol_vector_name[vec_num].push_back(solution_vector[vec_num][i]);
          sol_index[vec_num].push_back(len);

        } else if (solution_vector[vec_num][i + 1] == "S") {
          len = 1;  // scalar
          accum += len;
          SFT* solution = &metaData_->declare_field<SFT>(
              stk::topology::NODE_RANK, solution_vector[vec_num][i]);
          stk::mesh::put_field_on_mesh(
              *solution, metaData_->universal_part(), nullptr);
#ifdef ALBANY_SEACAS
          stk::io::set_field_role(*solution, Ioss::Field::TRANSIENT);
#endif
          sol_vector_name[vec_num].push_back(solution_vector[vec_num][i]);
          sol_index[vec_num].push_back(len);

        } else {
          TEUCHOS_TEST_FOR_EXCEPTION(
              true,
              std::logic_error,
              "Error in input file: specification of solution vector layout is "
              "incorrect."
                  << std::endl);
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(
          accum != neq_,
          std::logic_error,
          "Error in input file: specification of solution vector layout is "
          "incorrect."
              << std::endl);
    }
  }

  // Do the coordinates
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

  initializeSTKAdaptation();
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::initializeSTKAdaptation()
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

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::fillVector(
    Thyra_Vector&                                field_vector,
    const std::string&                           field_name,
    stk::mesh::Selector&                         field_selection,
    const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
    const NodalDOFManager&                       nodalDofManager)
{
  fillVectorImpl(
      field_vector,
      field_name,
      field_selection,
      field_node_vs,
      nodalDofManager,
      0);
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::fillSolnVector(
    Thyra_Vector&                                solution,
    stk::mesh::Selector&                         sel,
    const Teuchos::RCP<const Thyra_VectorSpace>& node_vs)
{
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, Interleaved);

  int offset = 0;
  for (int k = 0; k < sol_index[0].size(); k++) {
    fillVectorImpl(
        solution, sol_vector_name[0][k], sel, node_vs, nodalDofManager, offset);
    offset += sol_index[0][k];
  }
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::fillSolnMultiVector(
    Thyra_MultiVector&                           solution,
    stk::mesh::Selector&                         sel,
    const Teuchos::RCP<const Thyra_VectorSpace>& node_vs)
{
  using VFT = typename AbstractSTKFieldContainer::VectorFieldType;
  using SFT = typename AbstractSTKFieldContainer::ScalarFieldType;

  // Build a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, Interleaved);

  for (int icomp = 0; icomp < solution.domain()->dim(); ++icomp) {
    int offset = 0;

    for (int k = 0; k < sol_index[icomp].size(); k++) {
      fillVectorImpl(
          *solution.col(icomp),
          sol_vector_name[icomp][k],
          sel,
          node_vs,
          nodalDofManager,
          offset);
      offset += sol_index[icomp][k];
    }
  }
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::saveVector(
    const Thyra_Vector&                          field_vector,
    const std::string&                           field_name,
    stk::mesh::Selector&                         field_selection,
    const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
    const NodalDOFManager&                       nodalDofManager)
{
  saveVectorImpl(
      field_vector,
      field_name,
      field_selection,
      field_node_vs,
      nodalDofManager,
      0);
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::saveSolnVector(
    const Thyra_Vector&                          solution,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    stk::mesh::Selector&                         sel,
    const Teuchos::RCP<const Thyra_VectorSpace>& node_vs)
{
  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, Interleaved);

  int offset = 0;
  for (int k = 0; k < sol_index[0].size(); ++k) {
    // Recycle saveVectorImpl method
    saveVectorImpl(
        solution, sol_vector_name[0][k], sel, node_vs, nodalDofManager, offset);
    offset += sol_index[0][k];
  }
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::saveSolnVector(
    const Thyra_Vector& solution,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& /* solution_dot */,
    stk::mesh::Selector&                         sel,
    const Teuchos::RCP<const Thyra_VectorSpace>& node_vs)
{
  // TODO: why can't we save also solution_dot?
  Teuchos::RCP<Teuchos::FancyOStream> out =
      Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "IKT WARNING: calling MultiSTKFieldContainer::saveSolnVectorT with "
          "soln_dotT, but "
       << "this function has not been extended to write soln_dotT properly to "
          "the Exodus file.  Exodus "
       << "file will contain only soln, not soln_dot.\n";

  saveSolnVector(solution, soln_dxdp, sel, node_vs);
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::saveSolnVector(
    const Thyra_Vector& solution,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& /* solution_dot */,
    const Thyra_Vector& /* solution_dotdot */,
    stk::mesh::Selector&                         sel,
    const Teuchos::RCP<const Thyra_VectorSpace>& node_vs)
{
  // TODO: why can't we save also solution_dot?
  Teuchos::RCP<Teuchos::FancyOStream> out =
      Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "IKT WARNING: calling MultiSTKFieldContainer::saveSolnVectorT with "
          "soln_dotT and "
       << "soln_dotdotT, but this function has not been extended to write "
          "soln_dotT "
       << "and soln_dotdotT properly to the Exodus file.  Exodus "
       << "file will contain only soln, not soln_dot and soln_dotdot.\n";

  saveSolnVector(solution, soln_dxdp, sel, node_vs);
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::saveSolnMultiVector(
    const Thyra_MultiVector&                     solution,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    stk::mesh::Selector&                         sel,
    const Teuchos::RCP<const Thyra_VectorSpace>& node_vs)
{
  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, Interleaved);

  for (int icomp = 0; icomp < solution.domain()->dim(); ++icomp) {
    int offset = 0;
    for (int k = 0; k < sol_index[icomp].size(); k++) {
      saveVectorImpl(
          *solution.col(icomp),
          sol_vector_name[icomp][k],
          sel,
          node_vs,
          nodalDofManager,
          offset);
      offset += sol_index[icomp][k];
    }
  }
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::saveResVector(
    const Thyra_Vector&                          res,
    stk::mesh::Selector&                         sel,
    const Teuchos::RCP<const Thyra_VectorSpace>& node_vs)
{
  // Setup a dof manger on the fly (it's cheap anyways).
  // We don't care about global dofs (hence, the -1), since it's used only
  // to retrieve the local entry id in the thyra vector.
  // The number of equations is given by sol_index
  const LO        numLocalNodes = getSpmdVectorSpace(node_vs)->localSubDim();
  NodalDOFManager nodalDofManager;
  nodalDofManager.setup(this->neq, numLocalNodes, -1, Interleaved);

  int offset = 0;
  for (int k = 0; k < res_index.size(); k++) {
    saveVectorImpl(
        res, res_vector_name[k], sel, node_vs, nodalDofManager, offset);
    offset += res_index[k];
  }
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::transferSolutionToCoords()
{
  const bool MultiSTKFieldContainer_transferSolutionToCoords_not_implemented =
      true;
  TEUCHOS_TEST_FOR_EXCEPT(
      MultiSTKFieldContainer_transferSolutionToCoords_not_implemented);
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::fillVectorImpl(
    Thyra_Vector&                                field_vector,
    const std::string&                           field_name,
    stk::mesh::Selector&                         field_selection,
    const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
    const NodalDOFManager&                       nodalDofManager,
    const int                                    offset)
{
  using VFT = typename AbstractSTKFieldContainer::VectorFieldType;
  using SFT = typename AbstractSTKFieldContainer::ScalarFieldType;

  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  const stk::mesh::BucketVector& all_elements =
      this->bulkData->get_buckets(stk::topology::NODE_RANK, field_selection);

  auto* raw_field =
      this->metaData->get_field(stk::topology::NODE_RANK, field_name);
  ALBANY_EXPECT(
      raw_field != nullptr,
      "Error! Something went wrong while retrieving a field.\n");
  const int rank = raw_field->field_array_rank();

  auto field_node_vs_indexer = createGlobalLocalIndexer(field_node_vs);
  if (rank == 0) {
    using Helper     = STKFieldContainerHelper<SFT>;
    const SFT* field = this->metaData->template get_field<SFT>(
        stk::topology::NODE_RANK, field_name);
    for (auto it = all_elements.begin(); it != all_elements.end(); ++it) {
      const stk::mesh::Bucket& bucket = **it;
      Helper::fillVector(
          field_vector, *field, field_node_vs_indexer, bucket, nodalDofManager, offset);
    }
  } else if (rank == 1) {
    using Helper     = STKFieldContainerHelper<VFT>;
    const VFT* field = this->metaData->template get_field<VFT>(
        stk::topology::NODE_RANK, field_name);
    for (auto it = all_elements.begin(); it != all_elements.end(); ++it) {
      const stk::mesh::Bucket& bucket = **it;
      Helper::fillVector(
          field_vector, *field, field_node_vs_indexer, bucket, nodalDofManager, offset);
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::runtime_error,
        "Error! Only scalar and vector fields supported so far.\n");
  }
}

template <bool Interleaved>
void
MultiSTKFieldContainer<Interleaved>::saveVectorImpl(
    const Thyra_Vector&                          field_vector,
    const std::string&                           field_name,
    stk::mesh::Selector&                         field_selection,
    const Teuchos::RCP<const Thyra_VectorSpace>& field_node_vs,
    const NodalDOFManager&                       nodalDofManager,
    const int                                    offset)
{
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
    using Helper = STKFieldContainerHelper<SFT>;
    SFT* field   = this->metaData->template get_field<SFT>(
        stk::topology::NODE_RANK, field_name);
    for (auto it = all_elements.begin(); it != all_elements.end(); ++it) {
      const stk::mesh::Bucket& bucket = **it;
      Helper::saveVector(
          field_vector, *field, field_node_vs_indexer, bucket, nodalDofManager, offset);
    }
  } else if (rank == 1) {
    using Helper = STKFieldContainerHelper<VFT>;
    VFT* field   = this->metaData->template get_field<VFT>(
        stk::topology::NODE_RANK, field_name);
    for (auto it = all_elements.begin(); it != all_elements.end(); ++it) {
      const stk::mesh::Bucket& bucket = **it;
      Helper::saveVector(
          field_vector, *field, field_node_vs_indexer, bucket, nodalDofManager, offset);
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::runtime_error,
        "Error! Only scalar and vector fields supported so far.\n");
  }
}

}  // namespace Albany
