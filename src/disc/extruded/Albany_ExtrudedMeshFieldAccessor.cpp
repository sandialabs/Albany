#include "Albany_ExtrudedMeshFieldAccessor.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_StringUtils.hpp"

namespace Albany {

ExtrudedMeshFieldAccessor::
ExtrudedMeshFieldAccessor (const Teuchos::RCP<AbstractMeshFieldAccessor>& basal_field_accessor,
                           const Teuchos::RCP<LayeredMeshNumbering<LO>>& elem_numbering_lid)
 : m_basal_field_accessor(basal_field_accessor)
 , m_elem_numbering_lid(elem_numbering_lid)
{
  // Nothing to do here
}

  // Add states to mesh (and possibly to nodal_sis/nodal_parameter_sis)
void ExtrudedMeshFieldAccessor::
addStateStruct(const Teuchos::RCP<StateStruct>& st)
{
  all_sis.push_back(st);
  // Add to the proper structure
  switch (st->stateType()) {
    case StateStruct::GlobalState:
      global_sis.push_back(st);
      break;
    case StateStruct::NodeState:
      nodal_sis.push_back(st); break;
      if (st->entity==StateStruct::NodalDistParameter)
        nodal_parameter_sis.push_back(st);
      elem_sis.push_back(st); // We store 3d states as (Cell, Node, ...)
      break;
    case StateStruct::ElemState:
      elem_sis.push_back(st); break;
      break;
    default:
      throw std::logic_error("Error! Unrecognized/unsupported type for state " + st->name + "\n");
  };

  // If not extruded and not interpolated, then it must be an output,
  // so we must add it to the basal mesh, with a layout that is compatible
  // with the basal mesh.
  if (not st->extruded and not st->interpolated) {
    int num_elem_layers = m_elem_numbering_lid->numLayers;
    int num_node_layers = num_elem_layers+1;

    auto bst = Teuchos::rcp(new StateStruct(st->name,st->entity));
    switch(st->stateType()) {
      case StateStruct::GlobalState:
        global_sis.push_back(st);
        break;
      case StateStruct::NodeState:
        break;
      case StateStruct::ElemState:
        bst->entity = StateStruct::ElemData;
        bst->dim.push_back(st->dim[0] / num_elem_layers);
        bst->dim.push_back(num_elem_layers);
        for (size_t i=1; i<st->dim.size(); ++i) {
          bst->dim.push_back(st->dim[i]);
        }
        break;
      default:
        throw std::logic_error("Error! Unrecognized/unsupported type for state " + st->name + "\n");
    }

    m_basal_field_accessor->addStateStruct(bst);
  }
}

void ExtrudedMeshFieldAccessor::
createStateArrays (const WorksetArray<int>& worksets_sizes)
{
  elemStateArrays.resize(worksets_sizes.size());

  // We need to be careful here. Say we have a state X with layout (Node).
  // If X is also a state in the basal mesh, we may have to extrude/interpolate
  // the field. In particular, if
  //  - state X is stored also in the basal mesh as (Node,Layer): interpolate
  //  - state X is stored also in the basal mesh as (Node): extrude
  // The reason we need to be careful is that we only have the basal mesh to
  // store the arrays, and we may not be able to store 2 fields with the same
  // name but different layout.

  auto create_state = [&] (const StateStruct& st, int ws) {
    auto& state = elemStateArrays[ws][st.name];
    int ws_size = worksets_sizes[ws];
    switch (st.dim.size()) {
      case 1:
        state.resize(st.name,ws_size); break;
      case 2:
        state.resize(st.name,ws_size,st.dim[1]); break;
      case 3:
        state.resize(st.name,ws_size,st.dim[1],st.dim[2]); break;
      case 4:
        state.resize(st.name,ws_size,st.dim[1],st.dim[2],st.dim[3]); break;
      default:
        throw std::runtime_error("Error! Unexpected/unsupported rank for state '" + st.name + "'.\n");
    }
  };
  auto view_bstate = [&] (const StateStruct& st, int ws) {
    auto& bstate = m_basal_field_accessor->getElemStates()[ws][st.name];
    auto data_d = bstate.dev().data();
    auto data_h = bstate.host().data();

    auto& state = elemStateArrays[ws][st.name];
    int ws_size = worksets_sizes[ws];
    switch (st.dim.size()) {
      case 1:
        state.reset_from_dev_host_ptr(data_d,data_h,ws_size); break;
      case 2:
        state.reset_from_dev_host_ptr(data_d,data_h,ws_size,st.dim[1]); break;
      case 3:
        state.reset_from_dev_host_ptr(data_d,data_h,ws_size,st.dim[1],st.dim[2]); break;
      case 4:
        state.reset_from_dev_host_ptr(data_d,data_h,ws_size,st.dim[1],st.dim[2],st.dim[3]); break;
      default:
        throw std::runtime_error("Error! Unexpected/unsupported rank for state '" + st.name + "'.\n");
    }
  };
  int num_ws = worksets_sizes.size();
  for (auto st : elem_sis) {
    for (int ws=0; ws<num_ws; ++ws) {
      if (st->extruded or st->interpolated) {
        // This state is not stored in the basal mesh, so we need to allocate the state array here
        create_state(*st,ws);
      } else {
        // There is a state in the basal mesh which we can view
        view_bstate(*st,ws);
      }
    }
  }
  // NOTES: 1. We only store the elem version of nodal_sis, as ElemData in basal mesh.
  //        2. Due to index oredering, we cannot view basal states.
  //           E.g. scalar2d=(nbelem,nbnodes,nlay,2), scalar3d=(nbelem*nlay,nbnode*2),
  //           so in 3d, we do the full 3d elem before moving up a layer, but in the 2d
  //           state, for each 2d node we do all layers.
  for (auto st : nodal_sis) {
    for (int ws=0; ws<num_ws; ++ws) {
      create_state(*st,ws);
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

// Read from mesh methods
void ExtrudedMeshFieldAccessor::
fillSolnVector (Thyra_Vector&        soln,
                const dof_mgr_ptr_t& sol_dof_mgr,
                const bool           overlapped)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::fillSolnVector()");
}

void ExtrudedMeshFieldAccessor::
fillVector (Thyra_Vector&        field_vector,
            const std::string&   field_name,
            const dof_mgr_ptr_t& field_dof_mgr,
            const bool           overlapped)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::fillVector()");
}

void ExtrudedMeshFieldAccessor::
fillSolnMultiVector (Thyra_MultiVector&   /* soln */,
                     const dof_mgr_ptr_t& /* sol_dof_mgr */,
                     const bool           /* overlapped */)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::fillSolnMultiVector()");
}

void ExtrudedMeshFieldAccessor::
fillSolnSensitivity (Thyra_MultiVector&   /* dxdp */,
                     const dof_mgr_ptr_t& /* sol_dof_mgr */,
                     const bool           /* overlapped */)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::fillSolnSensitivity()");
}

  // Write to mesh methods
void ExtrudedMeshFieldAccessor::
saveVector (const Thyra_Vector&  /* field_vector */,
            const std::string&   /* field_name */,
            const dof_mgr_ptr_t& /* field_dof_mgr */,
            const bool           /* overlapped */)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::saveVector()");
}

void ExtrudedMeshFieldAccessor::
saveSolnVector (const Thyra_Vector&  /* soln */,
                const mv_ptr_t&      /* soln_dxdp */,
                const dof_mgr_ptr_t& /* sol_dof_mgr */,
                const bool           /* overlapped */)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::saveSolnVector()");
}

void ExtrudedMeshFieldAccessor::
saveSolnVector (const Thyra_Vector&  /* soln */,
                const mv_ptr_t&      /* soln_dxdp */,
                const Thyra_Vector&  /* soln_dot */,
                const dof_mgr_ptr_t& /* sol_dof_mgr */,
                const bool           /* overlapped */)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::saveSolnVector()");
}

void ExtrudedMeshFieldAccessor::
saveSolnVector (const Thyra_Vector&  /* soln */,
                const mv_ptr_t&      /* soln_dxdp */,
                const Thyra_Vector&  /* soln_dot */,
                const Thyra_Vector&  /* soln_dotdot */,
                const dof_mgr_ptr_t& /* sol_dof_mgr */,
                const bool           /* overlapped */)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::saveSolnVector()");
}

void ExtrudedMeshFieldAccessor::
saveResVector (const Thyra_Vector&  /* res */,
               const dof_mgr_ptr_t& /* dof_mgr */,
               const bool           /* overlapped */)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::saveResVector()");
}

void ExtrudedMeshFieldAccessor::
saveSolnMultiVector (const Thyra_MultiVector& /* soln */,
                     const mv_ptr_t&          /* soln_dxdp */,
                     const dof_mgr_ptr_t&     /* node_vs */,
                     const bool               /* overlapped */)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::saveSolnMultiVector()");
}

void ExtrudedMeshFieldAccessor::extrudeBasalFields (const Teuchos::Array<std::string>& basal_fields)
{
  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  const auto& basal_states = m_basal_field_accessor->getElemStates();
  const int num_ws = basal_states.size();
  const int num_elem_layers = m_elem_numbering_lid->numLayers;
  const int num_node_layers = num_elem_layers+1;
  *out << "[ExtrudedMeshFieldAccessor] Extruding basal fields...\n";
  for (const auto& name : basal_fields) {
    *out << " - Extruding field '" + name + "'...";
    bool nodal = Teuchos::nonnull(nodal_sis.find(name,false));
    for (int ws=0; ws<num_ws; ++ws) {
      const auto& bstate = basal_states[ws].at(name);
      auto& state = elemStateArrays[ws][name];
      auto bview_h = bstate.host();
      auto view_h = state.host();
      std::vector<int> dims;
      bstate.dimensions(dims);
      int rank = view_h.rank();
      TEUCHOS_TEST_FOR_EXCEPTION (
          (nodal and (rank<2 or rank>3)) or (not nodal and (rank<1 or rank>2)), std::runtime_error,
          "[ExtrudedMeshFieldAccessor::extrudeBasalFields] Error! Unsupported rank for state '" + name + "'\n");

      std::vector<int> dims3d;
      state.dimensions(dims3d);
      for (int ie=0; ie<dims[0]; ++ie) {
        for (int il=0; il<num_elem_layers; ++il) {
          int ie3d = m_elem_numbering_lid->getId(ie,il);
          if (nodal) {
            for (int in=0; in<dims[1]; ++in) {
              if (rank==2) {
                view_h(ie3d,in) = bview_h(ie,in);
                view_h(ie3d,in+dims[1]) = bview_h(ie,in);
              } else if (rank==3) {
                for (int j=0; j<dims[2]; ++j) {
                  view_h(ie3d,in,j) = bview_h(ie,in,j);
                  view_h(ie3d,in+dims[1],j) = bview_h(ie,in,j);
                }
              }
            }
          } else {
            if (rank==1) {
              view_h(ie3d) = bview_h(ie);
            } else if (rank==2) {
              for (int j=0; j<dims[1]; ++j) {
                view_h(ie3d,j) = bview_h(ie,j);
              }
            }
          }
        }
      }
      state.sync_to_dev();
    }
    *out << "done!\n";
  }
}

void ExtrudedMeshFieldAccessor::interpolateBasalLayeredFields (const Teuchos::Array<std::string>& basal_fields)
{
  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  const auto& basal_states = m_basal_field_accessor->getElemStates();
  const int num_ws = basal_states.size();
  const int num_elem_layers = m_elem_numbering_lid->numLayers;
  const int num_node_layers = num_elem_layers+1;
  const auto& node_layers_coords = mesh_vector_states.at("node_layers_coords");

  int il0, il1; // used for convex combination of data
  double h0;

  *out << "[ExtrudedMeshFieldAccessor] Interpolating basal fields...\n";
  for (const auto& name : basal_fields) {
    *out << " - Interpolating field '" + name + "'...";
    bool nodal = Teuchos::nonnull(nodal_sis.find(name,false));
    const auto& field_layers_coords = m_basal_field_accessor->getMeshVectorStates().at(name+"_NLC");
    const int num_field_layers = field_layers_coords.size();

    // Will update il0, il1, h0
    auto set_combination_params = [&](int il) {
      // Find where the mesh layer stands in the field layers
      double mesh_layer_coord = nodal ? node_layers_coords[il]
                                      : (node_layers_coords[il] + node_layers_coords[il+1])/2;
      auto where = std::upper_bound(field_layers_coords.begin(),field_layers_coords.end(),mesh_layer_coord);
      if (where==field_layers_coords.begin()) {
        // mesh layer is below the first field layer
        il0 = il1 = 0;
        h0 = 0; // Useless: the 2 values in convex combination are  the same, but for clarity, pick 0
      } else if (where==field_layers_coords.end()) {
        // mesh layer is above the last field layer
        il0 = il1 = num_field_layers-1;
        h0 = 0; // Useless: the 2 values in convex combination are  the same, but for clarity, pick 0
      } else {
        il1 = std::distance(field_layers_coords.begin(),where);
        il0 = il1-1;
        h0 = (field_layers_coords[il1] - mesh_layer_coord) / (field_layers_coords[il1] - field_layers_coords[il0]);
      }
    };

    for (int ws=0; ws<num_ws; ++ws) {
      const auto& bstate = basal_states[ws].at(name);
      auto& state = elemStateArrays[ws][name];
      auto bview_h = bstate.host();
      auto view_h = state.host();
      std::vector<int> dims;
      bstate.dimensions(dims);
      int rank = view_h.rank();
      TEUCHOS_TEST_FOR_EXCEPTION (
          (nodal and (rank<2 or rank>3)) or (not nodal and (rank<1 or rank>2)), std::runtime_error,
          "[ExtrudedMeshFieldAccessor::interpolateBasalLayeredFields] Error!\n"
          "  Unsupported rank (" << rank << ") for state '" + name + "'\n");

      for (int ie=0; ie<dims[0]; ++ie) {
        for (int il=0; il<num_elem_layers; ++il) {
          int ie3d = m_elem_numbering_lid->getId(ie,il);
          if (nodal) {
            for (int side : {0,1}) { // 0=elem-bottom, 1=elem-top
              set_combination_params(il+side);
              for (int in=0; in<dims[1]; ++in) {
                if (rank==2) {
                  view_h(ie3d,in+side*dims[1]) = h0*bview_h(ie,in,il0) + (1-h0)*bview_h(ie,in,il1);
                } else if (rank==3) {
                  for (int j=0; j<dims[2]; ++j) {
                    view_h(ie3d,in+side*dims[1],j) = h0*bview_h(ie,in,j,il0) + (1-h0)*bview_h(ie,in,j,il1);
                  }
                }
              }
            }
          } else {
            set_combination_params(il);
            if (rank==1) {
              view_h(ie3d) = h0*bview_h(ie,il0) + (1-h0)*bview_h(ie,il1);
            } else if (rank==2) {
              for (int j=0; j<dims[1]; ++j) {
                view_h(ie3d,j) = h0*bview_h(ie,j,il0) + (1-h0)*bview_h(ie,j,il1);
              }
            }
          }
        }
      }
      state.sync_to_dev();
    }
    *out << "done!\n";
  }
}

}  // namespace Albany
