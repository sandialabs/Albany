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
#include "Albany_StringUtils.hpp"

// Start of STK stuff
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>
#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

#include "Teuchos_VerboseObject.hpp"

#include <numeric>

namespace Albany {

static const char* sol_tag_name[3] = {"Exodus Solution Name",
                                      "Exodus SolutionDot Name",
                                      "Exodus SolutionDotDot Name"};

static const char* sol_id_name[3] = {"solution",
                                     "solution_dot",
                                     "solution_dotdot"};

MultiSTKFieldContainer::MultiSTKFieldContainer(
    const Teuchos::RCP<Teuchos::ParameterList>&   params_,
    const Teuchos::RCP<stk::mesh::MetaData>&      metaData_,
    const Teuchos::RCP<stk::mesh::BulkData>&      bulkData_,
    const int                                     numDim_,
    const Teuchos::RCP<Albany::StateInfoStruct>&  sis,
    const int                                     num_params_)
    : GenericSTKFieldContainer(
          params_,
          metaData_,
          bulkData_,
          numDim_,
          num_params_)
{
  // Do the coordinates
  this->coordinates_field =
      metaData_->get_field<double>(stk::topology::NODE_RANK, "coordinates");

  //STK throws when declaring a field that has been already declared
  if(this->coordinates_field == nullptr) {
    this->coordinates_field = 
        &metaData_->declare_field<double>(stk::topology::NODE_RANK, "coordinates");
  }

  stk::mesh::put_field_on_mesh(
      *this->coordinates_field, metaData_->universal_part(), numDim_, nullptr);
#ifdef ALBANY_SEACAS
  stk::io::set_field_role(*this->coordinates_field, Ioss::Field::MESH);
#endif

  if (numDim_ == 3) {
    this->coordinates_field3d = this->coordinates_field;
  } else {
    this->coordinates_field3d = &metaData_->declare_field<double>(
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

  initializeProcRankField();

  this->addStateStructs(sis);
}

MultiSTKFieldContainer::MultiSTKFieldContainer(
    const Teuchos::RCP<Teuchos::ParameterList>&        params_,
    const Teuchos::RCP<stk::mesh::MetaData>&           metaData_,
    const Teuchos::RCP<stk::mesh::BulkData>&           bulkData_,
    const int                                          neq_,
    const int                                          numDim_,
    const Teuchos::RCP<StateInfoStruct>&               sis,
    const Teuchos::Array<Teuchos::Array<std::string>>& solution_vector,
    const int                                          num_params_)
    : GenericSTKFieldContainer(
          params_,
          metaData_,
          bulkData_,
          neq_,
          numDim_,
          num_params_)
{
  typedef typename AbstractSTKFieldContainer::STKFieldType       SFT;

  if (save_solution_field) {
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
        SFT* solution =
            &metaData_->declare_field<double>(stk::topology::NODE_RANK, name);
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

        SFT* solution = &metaData_->declare_field<double>(
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
            SFT* solution = &metaData_->declare_field<double>(
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
            SFT* solution = &metaData_->declare_field<double>(
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
  }
}

void MultiSTKFieldContainer::
initializeProcRankField()
{
  this->proc_rank_field = &this->metaData->template declare_field<int>(
      stk::topology::ELEMENT_RANK, "proc_rank");

  // Processor rank field, a scalar
  stk::mesh::put_field_on_mesh(
      *this->proc_rank_field, this->metaData->universal_part(), nullptr);

#ifdef ALBANY_SEACAS
  stk::io::set_field_role(*this->proc_rank_field, Ioss::Field::MESH);
#endif
}

void MultiSTKFieldContainer::
fillVector(Thyra_Vector&        field_vector,
           const std::string&   field_name,
           const dof_mgr_ptr_t& field_dof_mgr,
           const bool           overlapped)
{
  fillVectorImpl(field_vector, field_name, field_dof_mgr, overlapped);
}

void MultiSTKFieldContainer::
fillSolnVector(Thyra_Vector&        solution,
               const dof_mgr_ptr_t& solution_dof_mgr,
               const bool           overlapped)
{
  // Helper vector. Same as sol_index, but with extra entry=neq at the end,
  // to make computing components idx easier
  std::vector<int> offsets (sol_index[0].size()+1,0);
  for (int k=0; k<sol_index[0].size(); ++k) {
    offsets[k] = sol_index[0][k];
  }
  offsets.back() = this->neq;

  for (int k=0; k<sol_index[0].size(); ++k) {
    // Compute components indices
    std::vector<int> components(offsets[k+1]-offsets[k]);
    std::iota(components.begin(),components.end(),sol_index[0][k]);

    fillVectorImpl(solution, sol_vector_name[0][k], solution_dof_mgr, overlapped, components);
  }
}

void MultiSTKFieldContainer::
fillSolnMultiVector (      Thyra_MultiVector& solution,
                     const dof_mgr_ptr_t&     solution_dof_mgr,
                     const bool               overlapped)
{
  // Loop over time derivatives
  for (int ideriv=0; ideriv<solution.domain()->dim(); ++ideriv) {
    // Helper vector. Same as sol_index, but with extra entry=neq at the end,
    // to make computing components idx easier
    std::vector<int> offsets (sol_index[0].size()+1,0);
    for (int k=0; k<sol_index[ideriv].size(); ++k) {
      offsets[k] = sol_index[ideriv][k];
    }
    offsets.back() = this->neq;

    for (int k=0; k<sol_index[ideriv].size(); k++) {
      // Compute components indices
      std::vector<int> components(offsets[k+1]-offsets[k]);
      std::iota(components.begin(),components.end(),sol_index[ideriv][k]);

      fillVectorImpl(*solution.col(ideriv), sol_vector_name[ideriv][k], solution_dof_mgr, overlapped, components);
    }
  }
}

void MultiSTKFieldContainer::
saveVector (const Thyra_Vector&  field_vector,
            const std::string&   field_name,
            const dof_mgr_ptr_t& field_dof_mgr,
            const bool           overlapped)
{
  saveVectorImpl (field_vector, field_name, field_dof_mgr, overlapped);
}

void MultiSTKFieldContainer::
saveSolnVector (const Thyra_Vector& soln,
                const mv_ptr_t&     /* soln_dxdp */,
                const dof_mgr_ptr_t& sol_dof_mgr,
                const bool           overlapped)
{
  // Helper vector. Same as sol_index, but with extra entry=neq at the end,
  // to make computing components idx easier
  std::vector<int> offsets (sol_index[0].size()+1,0);
  for (int k=0; k<sol_index[0].size(); ++k) {
    offsets[k] = sol_index[0][k];
  }
  offsets.back() = this->neq;

  for (int k=0; k<sol_index[0].size(); ++k) {
    // Compute components indices
    std::vector<int> components(offsets[k+1]-offsets[k]);
    std::iota(components.begin(),components.end(),sol_index[0][k]);

    saveVectorImpl (soln, sol_vector_name[0][k], sol_dof_mgr, overlapped, components);
  }
}

void MultiSTKFieldContainer::
saveSolnVector (const Thyra_Vector&  soln,
                const mv_ptr_t&      soln_dxdp,
                const Thyra_Vector&  /* soln_dot */,
                const dof_mgr_ptr_t& sol_dof_mgr,
                const bool           overlapped)
{
  // TODO: why can't we save also solution_dot?
  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "IKT WARNING: calling MultiSTKFieldContainer::saveSolnVectorT with soln_dot,\n"
       << "but this function has not been extended to write soln_dot properly to the Exodus file.\n"
       << "Exodus file will contain only soln, not soln_dot.\n";

  saveSolnVector(soln, soln_dxdp, sol_dof_mgr, overlapped);
}

void MultiSTKFieldContainer::
saveSolnVector (const Thyra_Vector&  soln,
                const mv_ptr_t&      soln_dxdp,
                const Thyra_Vector&  /* soln_dot */,
                const Thyra_Vector&  /* soln_dotdot */,
                const dof_mgr_ptr_t& sol_dof_mgr,
                const bool           overlapped)
{
  // TODO: why can't we save also solution_dot?
  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  *out << "IKT WARNING: calling MultiSTKFieldContainer::saveSolnVectorT with soln_dot and soln_dotdot,\n"
       << "but this function has not been extended to write soln_dot[dot] properly to the Exodus file.\n"
       << "Exodus file will contain only soln, not soln_dot nor soln_dotdot.\n";

  saveSolnVector(soln, soln_dxdp, sol_dof_mgr, overlapped);
}

void MultiSTKFieldContainer::
saveSolnMultiVector (const Thyra_MultiVector& soln,
                     const mv_ptr_t&          /* soln_dxdp */,
                     const dof_mgr_ptr_t&     sol_dof_mgr,
                     const bool               overlapped)
{
  const auto ncomp = soln.domain()->dim();
  switch (ncomp) {
    case 1:
      saveSolnVector (*soln.col(0), Teuchos::null, sol_dof_mgr, overlapped); break;
    case 2:
      saveSolnVector (*soln.col(0), Teuchos::null, *soln.col(1), sol_dof_mgr, overlapped); break;
    case 3:
      saveSolnVector (*soln.col(0), Teuchos::null, *soln.col(1), *soln.col(2), sol_dof_mgr, overlapped); break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
          "Error! Unexpected number of vectors in solution multivector: " << ncomp << "\n");
  }
}

void MultiSTKFieldContainer::
saveResVector (const Thyra_Vector&  res,
               const dof_mgr_ptr_t& dof_mgr,
               const bool           overlapped)
{
  // Helper vector. Same as res_index, but with extra entry=neq at the end,
  // to make computing components idx easier
  std::vector<int> offsets (res_index.size()+1,0);
  for (int k=0; k<res_index.size(); ++k) {
    offsets[k] = res_index[k];
  }
  offsets.back() = this->neq;

  for (int k=0; k<res_index.size(); ++k) {
    // Compute components indices
    std::vector<int> components(offsets[k+1]-offsets[k]);
    std::iota(components.begin(),components.end(),sol_index[0][k]);

    saveVectorImpl(res, res_vector_name[k], dof_mgr, overlapped, components);
  }
}

void MultiSTKFieldContainer::
transferSolutionToCoords()
{
  TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
    "Error MultiSTKFieldContainer::transferSolutionToCoords not yet implemented.\n");
}

void MultiSTKFieldContainer::
fillVectorImpl (Thyra_Vector&           field_vector,
                const std::string&      field_name,
                const dof_mgr_ptr_t&    field_dof_mgr,
                const bool              overlapped,
                const std::vector<int>& components)
{
  TEUCHOS_TEST_FOR_EXCEPTION (!this->solutionFieldContainer, std::logic_error,
    "Error MultiSTKFieldContainer::fillVectorImpl not called from a solution field container.\n");

  // Figure out if it's a nodal or elem field
  const auto& fp = field_dof_mgr->getGeometricFieldPattern();
  const auto& ftopo = field_dof_mgr->get_topology();
  std::vector<int> entity_dims_with_dofs;
  for (unsigned dim=0; dim<=ftopo.getDimension(); ++dim) {
    if (fp->getSubcellIndices(dim,0).size()>0) {
      entity_dims_with_dofs.push_back(dim);
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION (entity_dims_with_dofs.size()>1, std::runtime_error,
      "Error! We cannot save fields that are defined one more than one type of entity.\n"
      "  - field name: " << field_name << "\n"
      "  - entities with dofs: " << util::join(entity_dims_with_dofs,",") << "\n");

  auto field_entity_rank = static_cast<stk::topology::rank_t>(entity_dims_with_dofs[0]);

  auto* raw_field = metaData->get_field(field_entity_rank, field_name);
  ALBANY_EXPECT (raw_field != nullptr,
      "Error! Something went wrong while retrieving a field.\n");

  const auto* field = this->metaData->template get_field<double>(field_entity_rank, field_name);
  STKFieldContainerHelper::fillVector(field_vector, *field, *this->bulkData, field_dof_mgr, overlapped, components);
}

void MultiSTKFieldContainer::
saveVectorImpl (const Thyra_Vector&     field_vector,
                const std::string&      field_name,
                const dof_mgr_ptr_t&    field_dof_mgr,
                const bool              overlapped,
                const std::vector<int>& components)
{
  TEUCHOS_TEST_FOR_EXCEPTION (!this->solutionFieldContainer, std::logic_error,
    "Error MultiSTKFieldContainer::saveVectorImpl not called from a solution field container.\n");

  // Figure out if it's a nodal or elem field
  const auto& fp = field_dof_mgr->getGeometricFieldPattern();
  const auto& ftopo = field_dof_mgr->get_topology();
  std::vector<int> entity_dims_with_dofs;
  for (unsigned dim=0; dim<=ftopo.getDimension(); ++dim) {
    if (fp->getSubcellIndices(dim,0).size()>0) {
      entity_dims_with_dofs.push_back(dim);
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION (entity_dims_with_dofs.size()!=1, std::runtime_error,
      "Error! We cannot save fields that are defined on n!=1 type of entity.\n"
      "  - field name: " << field_name << "\n"
      "  - entities with dofs: " << util::join(entity_dims_with_dofs,",") << "\n");

  auto field_entity_rank = static_cast<stk::topology::rank_t>(entity_dims_with_dofs[0]);

  auto* raw_field = metaData->get_field(field_entity_rank, field_name);
  ALBANY_EXPECT (raw_field != nullptr,
      "Error! Something went wrong while retrieving a field.\n");

  auto* field = this->metaData->template get_field<double>(field_entity_rank, field_name);
  STKFieldContainerHelper::saveVector(field_vector, *field, *this->bulkData, field_dof_mgr, overlapped, components);
}

}  // namespace Albany
