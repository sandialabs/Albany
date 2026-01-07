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
#include "Albany_DiscretizationUtils.hpp"
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

MultiSTKFieldContainer::
MultiSTKFieldContainer (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                        const Teuchos::RCP<stk::mesh::MetaData>&    metaData_,
                        const Teuchos::RCP<stk::mesh::BulkData>&    bulkData_,
                        const int                                   /* num_params_ */,
                        const bool                                  set_geo_fields_meta_data)
 : GenericSTKFieldContainer(params_,metaData_,bulkData_)
{
  if (set_geo_fields_meta_data) {
    setGeometryFieldsMetadata ();
  }
}

void MultiSTKFieldContainer::
setSolutionFieldsMetadata(const int neq_)
{
  this->solutionFieldContainer = true;
  neq = neq_;
  save_solution_field = params->get("Save Solution Field", true);
  if (not save_solution_field) {
    return;
  }

  using strarr_t = Teuchos::Array<std::string>;
  strarr_t sol_tag_name = {"Exodus Solution Name",
                           "Exodus SolutionDot Name",
                           "Exodus SolutionDotDot Name"};

  strarr_t sol_id_name = {"solution",
                          "solution_dot",
                          "solution_dotdot"};

  int numDim = metaData->spatial_dimension();
  int num_time_deriv = params->get<int>("Number Of Time Derivatives");

  sol_vector_name.resize(num_time_deriv+1);
  sol_index.resize(num_time_deriv+1);
  
  strarr_t vec_names = {"Solution", "SolutionDot", "SolutionDotDot"};
  for (int ider = 0; ider < num_time_deriv; ++ider) {
    auto comp_specs = params->get<strarr_t>(vec_names[ider] + " Vector Components", {});
    if (comp_specs.size()==0) {
      // Do the default solution vector
      std::string name = params->get(sol_tag_name[ider], sol_id_name[ider]);
      add_field_to_mesh<double>(name,true,true,neq);

      sol_vector_name[ider].push_back(name);
      sol_index[ider].push_back(neq);
    } else if (comp_specs.size()==1) {
      // User is just renaming the entire solution vector
      add_field_to_mesh<double>(comp_specs[0],true,true,neq);

      sol_vector_name[ider].push_back(comp_specs[0]);
      sol_index[ider].push_back(neq);

    } else {
      // user is breaking up the solution into multiple fields
      // make sure the number of entries is even, as we want [cmp_name1, type1, cmp_name2, type2,..]
      // where typeN is either 'V' or 'S'
      TEUCHOS_TEST_FOR_EXCEPTION(comp_specs.size() % 2 == 1, std::logic_error,
          "Error in input file: specification of solution vector layout is incorrect.\n");

      int accum = 0;

      for (int i = 0; i < comp_specs.size(); i += 2) {
        if (comp_specs[i+1] == "V") {
          // A vector component
          add_field_to_mesh<double>(comp_specs[i],true,true,numDim);
          accum += numDim;
          sol_vector_name[ider].push_back(comp_specs[i]);
          sol_index[ider].push_back(numDim);
        } else if (comp_specs[i+1] == "S") {
          // A scalar component
          accum += 1;
          add_field_to_mesh<double>(comp_specs[i],true,true,0);
          sol_vector_name[ider].push_back(comp_specs[i]);
          sol_index[ider].push_back(1);
        } else {
          throw std::logic_error("Error in input file: specification of solution vector layout is incorrect.\n");
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(accum != neq, std::logic_error,
          "Error in input file: specification of solution vector layout is incorrect.\n");
    }
  }
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
fillSolnSensitivity(Thyra_MultiVector&                    /* dxdp */,
                    const Teuchos::RCP<const DOFManager>& /* solution_dof_mgr */,
                    const bool                            /* overlapped */)
{
  throw NotYetImplemented ("MultiSTKFieldContainer::fillSolnSensitivity");
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
  *out << "IKT WARNING: calling MultiSTKFieldContainer::saveSolnVector with soln_dot,\n"
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
  *out << "IKT WARNING: calling MultiSTKFieldContainer::saveSolnVector with soln_dot and soln_dotdot,\n"
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

  ALBANY_EXPECT (metaData->get_field(field_entity_rank, field_name) != nullptr,
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

  ALBANY_EXPECT (metaData->get_field(field_entity_rank, field_name) != nullptr,
      "Error! Something went wrong while retrieving a field.\n");

  auto* field = this->metaData->template get_field<double>(field_entity_rank, field_name);
  STKFieldContainerHelper::saveVector(field_vector, *field, *this->bulkData, field_dof_mgr, overlapped, components);
}

}  // namespace Albany
