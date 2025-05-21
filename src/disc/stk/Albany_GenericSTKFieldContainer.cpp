//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_GenericSTKFieldContainer.hpp"

#include "Albany_Utils.hpp"
#include "Albany_StateInfoStruct.hpp"

// Start of STK stuff
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>
#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

namespace Albany {

GenericSTKFieldContainer::GenericSTKFieldContainer(
  const Teuchos::RCP<Teuchos::ParameterList>& params_,
  const Teuchos::RCP<stk::mesh::MetaData>& metaData_,
  const Teuchos::RCP<stk::mesh::BulkData>& bulkData_,
  const int numDim_,
  const int num_params_)
  : AbstractSTKFieldContainer(false),
    metaData(metaData_),
    bulkData(bulkData_),
    params(params_),
    numDim(numDim_),
    num_params(num_params_) {
}

GenericSTKFieldContainer::GenericSTKFieldContainer(
  const Teuchos::RCP<Teuchos::ParameterList>& params_,
  const Teuchos::RCP<stk::mesh::MetaData>& metaData_,
  const Teuchos::RCP<stk::mesh::BulkData>& bulkData_,
  const int neq_,
  const int numDim_,
  const int num_params_)
  : AbstractSTKFieldContainer(true),
    metaData(metaData_),
    bulkData(bulkData_),
    params(params_),
    neq(neq_),
    numDim(numDim_),
    num_params(num_params_)
{
  save_solution_field = params_->get("Save Solution Field", true);
}

namespace {
#ifdef ALBANY_SEACAS
//amb 13 Nov 2014. After new STK was integrated, fields with output set to false
// were nonetheless being written to Exodus output files. As a possibly
// temporary but also possibly permanent fix, set the role of such fields to
// INFORMATION rather than TRANSIENT. The enum RoleType is defined in
// seacas/libraries/ioss/src/Ioss_Field.h. Grepping around there suggests that
// fields having the role INFORMATION are not written to file: first,
// INFORMATION is never actually used; second, I/O behavior is based on chained
// 'else if's with no trailing 'else'; hence, any role type not explicitly
// handled is not acted on.
// It appears that the output boolean is used only in this file in the context
// of role type, so for now I'm applying this fix only to this file.
//
// IKT, 5/9/2020: INFORMATION has gone away in Ioss_Field.h; replaced it with MESH_REDUCTION,
// per Greg Sjaardema's suggestion.  INFORMATION was an alias for MESH_REDUCTION.

inline Ioss::Field::RoleType role_type(const bool output) {
  return output ? Ioss::Field::TRANSIENT : Ioss::Field::MESH_REDUCTION;
}
#endif

void set_output_role (AbstractSTKFieldContainer::STKFieldType& f, bool output) {
#ifdef ALBANY_SEACAS
  stk::io::set_field_role(f, role_type(output));
#else
  (void) f;
  (void) output;
#endif
}
}

void GenericSTKFieldContainer::
addStateStructs(const StateInfoStruct& sis)
{
  // Code to parse the vector of StateStructs and create STK fields
  for (const auto& st : sis) {
    // Add to the stored SIS
    all_sis.push_back(st);

    StateStruct::FieldDims& dim = st->dim;
    if (st->entity == StateStruct::ElemData) {
      if (dim.size()==1) {
        // Scalar on cell
        cell_scalar_states.push_back(& metaData->declare_field< double >(stk::topology::ELEMENT_RANK, st->name));
        stk::mesh::put_field_on_mesh(*cell_scalar_states.back(), metaData->universal_part(), 1, nullptr);
        set_output_role(*cell_scalar_states.back(),st->output);
      } else if (dim.size()==2) {
        // Vector on cell
        cell_vector_states.push_back(& metaData->declare_field< double >(stk::topology::ELEMENT_RANK, st->name));
        stk::mesh::put_field_on_mesh(*cell_vector_states.back(), metaData->universal_part(), dim[1], nullptr);
        set_output_role(*cell_vector_states.back(),st->output);
      } else if (dim.size()==3) {
        // 2nd order tensor on cell
        cell_tensor_states.push_back(& metaData->declare_field< double >(stk::topology::ELEMENT_RANK, st->name));
        stk::mesh::put_field_on_mesh(*cell_tensor_states.back(), metaData->universal_part(), dim[2], dim[1], nullptr);
        set_output_role(*cell_tensor_states.back(),st->output);
      } else {
        throw std::logic_error("Error! Unexpected state rank.\n");
      }
      elem_sis.push_back(st);
    } else if(st->entity == StateStruct::QuadPoint || st->entity == StateStruct::ElemNode){
      if(dim.size() == 2){ // Scalar at QPs
        qpscalar_states.push_back(& metaData->declare_field< double >(stk::topology::ELEMENT_RANK, st->name));
        stk::mesh::put_field_on_mesh(*qpscalar_states.back(), metaData->universal_part(), dim[1], nullptr);
        set_output_role(*qpscalar_states.back(),st->output);
      } else if(dim.size() == 3){ // Vector at QPs
        qpvector_states.push_back(& metaData->declare_field< double >(stk::topology::ELEMENT_RANK, st->name));
        // Multi-dim order is Fortran Ordering, so reversed here
        stk::mesh::put_field_on_mesh(*qpvector_states.back(), metaData->universal_part(), dim[2], dim[1], nullptr);
        set_output_role(*qpvector_states.back(),st->output);
      } else if(dim.size() == 4){ // Tensor at QPs
        qptensor_states.push_back(& metaData->declare_field< double >(stk::topology::ELEMENT_RANK, st->name));
        // Multi-dim order is Fortran Ordering, so reversed here
        auto num_tens_entries = dim[3]*dim[2];
        stk::mesh::put_field_on_mesh(*qptensor_states.back() ,
                       metaData->universal_part(), num_tens_entries, dim[1], nullptr);
        set_output_role(*qptensor_states.back(),st->output);
      } else {
        throw std::logic_error("Error: GenericSTKFieldContainer - cannot match QPData");
      }
      elem_sis.push_back(st);
    } else if(dim.size() == 1 && st->entity == StateStruct::WorksetValue) {
      // A single value that applies over the entire workset (time)
      scalarValue_states.push_back(st->name); // Just save a pointer to the name allocated in st
    } else if ((st->entity == StateStruct::NodalData) ||(st->entity == StateStruct::NodalDataToElemNode) || (st->entity == StateStruct::NodalDistParameter)) {
      // Definitely a nodal state
      nodal_sis.push_back(st);
      auto nodalFieldDim = dim;
      if(st->entity == StateStruct::NodalDataToElemNode) {
        nodalFieldDim.erase(nodalFieldDim.begin()); // Remove the <Cell> extent
      } else if(st->entity == StateStruct::NodalDistParameter) {
        // Also a dist parameter
        nodal_parameter_sis.push_back(st);
        nodalFieldDim.erase(nodalFieldDim.begin()); // Remove the <Cell> extent
      }

      auto& fld = metaData->declare_field<double>(stk::topology::NODE_RANK, st->name);
      switch(nodalFieldDim.size()) {
        case 1: // scalar
          stk::mesh::put_field_on_mesh(fld , metaData->universal_part(), 1, nullptr);
          break;
        case 2: // vector
          stk::mesh::put_field_on_mesh(fld , metaData->universal_part(), nodalFieldDim[1], nullptr);
          break;
        case 3: // tensor
          stk::mesh::put_field_on_mesh(fld , metaData->universal_part(), nodalFieldDim[2], nodalFieldDim[1], nullptr);
      }
      set_output_role(fld,st->output);
    } else {
      throw std::logic_error("Error: GenericSTKFieldContainer - cannot match unknown entity : " + std::to_string(static_cast<int>(st->entity)) + "\n");
    }

    // Checking if the field is layered, in which case the normalized layer coordinates need to be stored in the meta data
    if (st->layered) {
      std::string tmp_str = st->name + "_NLC";

      TEUCHOS_TEST_FOR_EXCEPTION (mesh_vector_states.find(tmp_str)!=mesh_vector_states.end(), std::logic_error,
                                  "Error! Another layered state with the same name already exists.\n");
      TEUCHOS_TEST_FOR_EXCEPTION (dim.back()<=0, std::logic_error,
                                  "Error! Invalid layer dimension for state " + st->name + ".\n");
      mesh_vector_states[tmp_str] = std::vector<double>(dim.back());
    }
  }
}

} // namespace Albany
