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

GenericSTKFieldContainer::
GenericSTKFieldContainer (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                          const Teuchos::RCP<stk::mesh::MetaData>& metaData_,
                          const Teuchos::RCP<stk::mesh::BulkData>& bulkData_)
 : params (params_)
 , metaData (metaData_)
 , bulkData (bulkData_)
{
  // Nothing to do here
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
addStateStruct(const Teuchos::RCP<StateStruct>& st)
{
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
  } else if(st->stateType()==StateStruct::GlobalState) {
    TEUCHOS_TEST_FOR_EXCEPTION (st->dim.size()>1, std::runtime_error,
        "Unsupported rank (" << st->dim.size() << ") for GlobalState " << st->name << "\n");
    // A single scalar/vector that applies over the entire workset (e.g. time)
    if (st->dim[0]==1) {
      mesh_scalar_states[st->name] = 0;
    } else {
      mesh_vector_states[st->name].resize(st->dim[0],0);
    }
    global_sis.push_back(st);
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
        break;
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

void GenericSTKFieldContainer::createStateArrays (const WorksetArray<int>& worksets_sizes)
{
  const auto ELEM_RANK = stk::topology::ELEM_RANK;
  const auto NODE_RANK = stk::topology::NODE_RANK;

  auto select_owned_part = stk::mesh::Selector(metaData->universal_part()) &
                           stk::mesh::Selector(metaData->locally_owned_part());
  const auto& elem_buckets = bulkData->get_buckets(ELEM_RANK,select_owned_part);
  const auto& node_buckets = bulkData->get_buckets(NODE_RANK,select_owned_part);

  // Sanity checks
  TEUCHOS_TEST_FOR_EXCEPTION (worksets_sizes.size()!=static_cast<int>(elem_buckets.size()), std::logic_error,
      "[GenericSTKFieldContainer::createStateArrays] Error! Input worksets_sizes length does not match mesh num elem buckets.\n"
      " - worksets_sizes length : " << worksets_sizes.size() + "\n"
      " - num mesh elem buckets: " << elem_buckets.size() + "\n");
  for (size_t ws=0; ws<elem_buckets.size(); ++ws) {
    TEUCHOS_TEST_FOR_EXCEPTION (worksets_sizes[ws]!=static_cast<int>(elem_buckets[ws]->size()), std::logic_error,
        "[GenericSTKFieldContainer::createStateArrays] Error! Input workset size does not match mesh bucket size.\n"
        " - workset id        : " << ws << "\n"
        " - input workset_size: " << worksets_sizes[ws] + "\n"
        " - mesh bucket size  : " << elem_buckets[ws]->size() + "\n");
  }

  elemStateArrays.resize(elem_buckets.size());
  nodeStateArrays.resize(node_buckets.size());
  double* data;

  // Elem states
  for (const auto& st : elem_sis) {
    auto f = metaData->get_field<double>(ELEM_RANK,st->name);
    auto dim = st->dim;
    for (size_t ws=0; ws<elem_buckets.size(); ++ws) {
      const auto& b = *elem_buckets[ws];
      data = reinterpret_cast<double*>(b.field_data_location(*f));
      auto& state = elemStateArrays[ws][st->name];
      switch (dim.size()) {
        case 1:
          state.reset_from_host_ptr(data,b.size()); break;
        case 2:
          state.reset_from_host_ptr(data,b.size(),dim[1]); break;
        case 3:
          state.reset_from_host_ptr(data,b.size(),dim[1],dim[2]); break;
        case 4:
          state.reset_from_host_ptr(data,b.size(),dim[1],dim[2],dim[3]); break;
        default:
          throw std::runtime_error("Error! Unsupported rank for elem state '" + st->name + "'.\n");
      }
    }
  }

  // Nodal states
  for (const auto& st : nodal_sis) {
    auto f = metaData->get_field<double>(NODE_RANK,st->name);
    auto dim = st->dim;
    if (st->entity != StateStruct::NodalData) {
      // Add an elem state array, which the SaveStateField/SaveSideSetStateField evaluators will use
      for (size_t ws=0; ws<elem_buckets.size(); ++ws) {
        const auto& b = *elem_buckets[ws];
        auto& state = elemStateArrays[ws][st->name];
        switch (dim.size()) {
          case 2:
            state.resize(st->name,b.size(),dim[1]); break;
          case 3:
            state.resize(st->name,b.size(),dim[1],dim[2]); break;
          case 4:
            state.resize(st->name,b.size(),dim[1],dim[2],dim[3]); break;
          default:
            throw std::runtime_error("Error! Unsupported rank for elem state '" + st->name + "'.\n");
        }
      }
      // Remove <Cell> extent from dim, so we can use for sizing the nodeStateArray
      dim.erase(dim.begin());
    }
    for (size_t ws=0; ws<node_buckets.size(); ++ws) {
      const auto& b = *node_buckets[ws];
      data = reinterpret_cast<double*>(b.field_data_location(*f));
      switch (dim.size()) {
        case 1:
          nodeStateArrays[ws][st->name].reset_from_host_ptr(data,b.size()); break;
        case 2:
          nodeStateArrays[ws][st->name].reset_from_host_ptr(data,b.size(),dim[1]); break;
        case 3:
          nodeStateArrays[ws][st->name].reset_from_host_ptr(data,b.size(),dim[1],dim[2]); break;
        default:
          throw std::runtime_error("Error! Unsupported rank for node state '" + st->name + "'.\n");
      }
    }
  }

  // Global states
  for (const auto& st : global_sis) {
    auto& state = globalStates[st->name];
    if (st->dim.size()==1) {
      state.reset_from_host_ptr(&mesh_scalar_states[st->name],1);
    } else if (st->dim.size()==1) {
      state.reset_from_host_ptr(mesh_vector_states[st->name].data(),st->dim[0]);
    } else {
      throw std::runtime_error("Error! Unsupported rank for global state '" + st->name + "'.\n");
    }
  }
}

void GenericSTKFieldContainer::transferNodeStatesToElemStates ()
{
  const auto ELEM_RANK = stk::topology::ELEM_RANK;
  const auto NODE_RANK = stk::topology::NODE_RANK;

  auto select_owned_part = stk::mesh::Selector(metaData->universal_part()) &
                           stk::mesh::Selector(metaData->locally_owned_part());
  const auto& elem_buckets = bulkData->get_buckets(ELEM_RANK,select_owned_part);

  for (const auto& st : nodal_sis) {
    if (st->entity!=StateStruct::NodalDataToElemNode)
      continue;
    const auto& dim = st->dim;
    const auto rank = st->dim.size();

    auto fn = metaData->get_field<double>(NODE_RANK,st->name);
    for (size_t ws=0; ws<elem_buckets.size(); ++ws) {
      const auto& b = *elem_buckets[ws];
      auto& state = elemStateArrays[ws][st->name];
      auto& state_h = state.host();
      for (size_t i=0; i<b.size(); ++i) {
        const auto& elem  = b[i];
        const auto* nodes = bulkData->begin_nodes(elem);
        for (size_t j=0; j<dim[1]; ++j) {
          const auto* data = stk::mesh::field_data(*fn,nodes[j]);
          switch(rank) {
            case 2:
              state_h(i, j) = *data; break;
            case 3:
              for (size_t k=0; k<dim[2]; ++k) {
                state_h(i,j,k) = data[k];
              } break;
            case 4:
              for (size_t k=0; k<dim[2]; ++k) {
                for (size_t l=0; l<dim[3]; ++l) {
                  // TODO: CHECK THIS. Is the striding correct? Or should it be l*dim[2]+k?
                  state_h(i,j,k,l) = data[k*dim[3]+l];
                }
              } break;
          }
        }
      }
      state.sync_to_dev();
    }
  }
}

void GenericSTKFieldContainer::
transferElemStateToNodeState (const std::string& name)
{
  const auto ELEM_RANK = stk::topology::ELEM_RANK;

  auto select_owned_part = stk::mesh::Selector(metaData->universal_part()) &
                           stk::mesh::Selector(metaData->locally_owned_part());
  const auto& elem_buckets = bulkData->get_buckets(ELEM_RANK,select_owned_part);
  auto* node_field = metaData->get_field<double>(stk::topology::NODE_RANK,name);
  TEUCHOS_TEST_FOR_EXCEPTION(node_field==nullptr, std::runtime_error,
      "Error! Cannot retrieve nodal field '" + name + "\n");

  int num_buckets = elem_buckets.size();
  for (int ws=0; ws<num_buckets; ++ws) {
    auto& e_state_h = elemStateArrays[ws][name].host();

    const auto& bucket = *elem_buckets[ws];
    int nelems = bucket.size();
    int nelem_nodes = e_state_h.extent(1);
    int rank = e_state_h.rank();
    int dim2,dim3;
    for (int ie=0; ie<nelems; ++ie) {
      auto e = bucket[ie];
      const auto& nodes = bulkData->begin_nodes(e);
      for (int in=0; in<nelem_nodes; ++in) {
        auto values = stk::mesh::field_data(*node_field,nodes[in]);
        switch(rank) {
          case 2:
            values[0] = e_state_h(ie,in);
            break;
          case 3:
            dim2 = e_state_h.extent_int(2);
            for (int j=0; j<dim2; ++j) {
              values[j] = e_state_h(ie,in,j);
            }
          case 4:
            dim2 = e_state_h.extent_int(2);
            dim3 = e_state_h.extent_int(3);
            for (int j=0; j<dim2; ++j) {
              for (int k=0; k<dim3; ++k) {
                values[j*dim3+k] = e_state_h(ie,in,j,k);
              }
            }
          default:
            throw std::runtime_error("Unsupported rank for state '" + name + "'\n");
        }
      }
    }
  }
}

void GenericSTKFieldContainer::setGeometryFieldsMetadata ()
{
  int numDim = metaData->spatial_dimension();
  auto& universal_part = metaData->universal_part();

  this->coordinates_field = add_field_to_mesh<double> ("coordinates", true, false, numDim);

  if (numDim == 3) {
    this->coordinates_field3d = this->coordinates_field;
  } else {
    this->coordinates_field3d = add_field_to_mesh<double> ("coordinates3d", true, false, 3);
  }

  // Processor rank field, a scalar
  this->proc_rank_field = add_field_to_mesh<int> ("proc_rank", false, false, 0);
}

template<typename T>
stk::mesh::Field<T>*
GenericSTKFieldContainer::
add_field_to_mesh (const std::string& name,
                   const bool nodal,
                   const bool transient,
                   const int ncmp)
{
  constexpr auto NODE_RANK = stk::topology::NODE_RANK;
  constexpr auto ELEM_RANK = stk::topology::ELEM_RANK;

  auto rank = nodal ? NODE_RANK : ELEM_RANK;

  auto fptr = metaData->get_field<T>(rank,name);
  if (not fptr) {
    fptr = &metaData->declare_field<T>(rank,name);
  }
  auto& universal_part = metaData->universal_part();
  if (ncmp>0)
    stk::mesh::put_field_on_mesh(*fptr,universal_part,ncmp,nullptr);
  else
    stk::mesh::put_field_on_mesh(*fptr,universal_part,nullptr);
#ifdef ALBANY_SEACAS
  if (transient)
    stk::io::set_field_role(*fptr, Ioss::Field::TRANSIENT);
  else
    stk::io::set_field_role(*fptr, Ioss::Field::MESH);
#endif

  return fptr;
}

// Explicit instantiation
template stk::mesh::Field<int>*
GenericSTKFieldContainer::
add_field_to_mesh<int> (const std::string& name,
                        const bool nodal,
                        const bool transient,
                        const int);
template stk::mesh::Field<double>*
GenericSTKFieldContainer::
add_field_to_mesh<double> (const std::string& name,
                           const bool nodal,
                           const bool transient,
                           const int);

} // namespace Albany
