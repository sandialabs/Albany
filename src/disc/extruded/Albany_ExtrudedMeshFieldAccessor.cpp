#include "Albany_ExtrudedMeshFieldAccessor.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_StringUtils.hpp"

namespace {
  void print (const Albany::AbstractMeshFieldAccessor& accessor) {
    std::cout << " ELEM SIS:\n";
    for (auto st : accessor.getElemSIS()) {
      std::cout << "   " << st->name << ": (" << util::join(st->dim,",") << ")\n";
    }
    std::cout << " NODE SIS:\n";
    for (auto st : accessor.getNodalSIS()) {
      std::cout << "   " << st->name << ": (" << util::join(st->dim,",") << ")\n";
    }
    std::cout << " GLOB SIS:\n";
    for (auto st : accessor.getGlobalSIS()) {
      std::cout << "   " << st->name << ": (" << util::join(st->dim,",") << ")\n";
    }
  }
}
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
  std::cout << "BASAL SIS:\n";
  print(*m_basal_field_accessor);

  int num_state_layers = -1;
  switch (st->entity) {
    case StateStruct::ElemData:   [[fallthrough]];
    case StateStruct::ElemNode:
      num_state_layers = m_elem_numbering_lid->numLayers;
      elem_sis.push_back(st); break;
    case StateStruct::WorksetValue:
      global_sis.push_back(st); break;
    case StateStruct::NodalData:            [[fallthrough]];
    case StateStruct::NodalDataToElemNode:  [[fallthrough]];
    case StateStruct::NodalDistParameter:
      num_state_layers = m_elem_numbering_lid->numLayers+1;
      nodal_sis.push_back(st); break;
    case StateStruct::QuadPoint:  [[fallthrough]]; // No support for QP states yet
    default:
      throw std::runtime_error("Unrecognized/unsupported StateInfo entity.\n");
  }

  if (find_basal_state(st).is_null()) {
    auto bst = Teuchos::rcp(new StateStruct(st));
    if (num_state_layers>0) {
      bst->dim.push_back(m_elem_numbering_lid->numLayers);
    }
    if (util::starts_with(st->name,"extruded_")) {
      bst->ebName = get_basal_part_name(st->name);
    }

    m_basal_field_accessor->addStateStruct(bst);
  }
  print(*this);
}

void ExtrudedMeshFieldAccessor::
createStateArrays (const WorksetArray<int>& worksets_sizes)
{
  // We need to be careful here. Say we have a state X with layout (Node).
  // If X is also a state in the basal mesh, we may have to extrude/interpolate
  // the field. In particular, if
  //  - state X is stored also in the basal mesh as (Node,Layer): interpolate
  //  - state X is stored also in the basal mesh as (Node): extrude
  // The reason we need to be careful is that we only have the basal mesh to
  // store the arrays, and we may not be able to store 2 fields with the same
  // name but different layout.
  int num_ws = worksets_sizes.size();
  for (auto st : elem_sis) {
    auto bst = find_basal_state(st);
    bool has_bst = bst.is_null();
    // We only allow name clashing if this field was extruded/interpolated from a basal one
    TEUCHOS_TEST_FOR_EXCEPTION (has_bst and not bst->extruded and not bst->layered, std::runtime_error,
        "[ExtrudedMeshFieldAccessor::createStateArrays] Error! State name clashes with basal disc state name.\n"
        "  - state name: " + bst->name + "\n"
        "NOTE: name can clash only if the basal state is interpolated or extruded vertically.\n");
    for (int ws=0; ws<num_ws; ++ws) {
      auto& state = elemStateArrays[ws][st->name];
      auto ws_size = worksets_sizes[ws];
      if (has_bst) {
        // There's a field in the basal mesh (either layered or basal-only), but the 3d field will have
        // different dims. So we cannot "view" the basal mesh field. Instead, create state managed arrays
        switch (st->dim.size()) {
          case 1:
            state.resize(data,ws_size); break;
          case 2:
            state.resize(data,ws_size,dim[1]); break;
          case 3:
            state.resize(data,ws_size,dim[1],dim[2]); break;
          case 4:
            state.resize(data,ws_size,dim[1],dim[2],dim[3]); break;
          default:
            throw std::runtime_error("Error! Unexpected/unsupported rank for elem state '" + st->name + "'.\n");
        }
      } else {
        // The basal field accessor (BFA) did not have a state with this name. So we did register the 3d state as
        // a "n+1 dimensional" state in the BFA. We can view that data here
        auto dev_ptr = state.dev().data();
        auto host_ptr = state.host().data();
        switch (st->dim.size()) {
          case 1:
            state.reset_from_dev_host_ptr(dev_ptr,host_ptr,ws_size); break;
          case 2:
            state.reset_from_dev_host_ptr(dev_ptr,host_ptr,ws_size,dim[1]); break;
          case 3:
            state.reset_from_dev_host_ptr(dev_ptr,host_ptr,ws_size,dim[1],dim[2]); break;
          case 4:
            state.reset_from_dev_host_ptr(dev_ptr,host_ptr,ws_size,dim[1],dim[2],dim[3]); break;
          default:
            throw std::runtime_error("Error! Unsupported rank for elem state '" + st->name + "'.\n");
        }
      }
    }
  }
  print(*m_basal_field_accessor);
  print(*this);
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::createStateArrays()");
}

void ExtrudedMeshFieldAccessor::
transferNodeStatesToElemStates ()
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::transferNodeStatesToElemStates()");
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

Teuchos::RCP<StateStruct>
ExtrudedMeshFieldAccessor::find_basal_state (const Teuchos::RCP<StateStruct>& st)
{
  const auto& bes = m_basal_field_accessor->getElemSIS();
  const auto& bns = m_basal_field_accessor->getNodalSIS();
  const auto& bgs = m_basal_field_accessor->getGlobalSIS();
  const auto& basal_states = st->stateType()==StateStruct::ElemState ? bes : (st->stateType()==StateStruct::NodeState ? bns : bgs);
  Teuchos::RCP<StateStruct> p;
  for (auto bst : basal_states) {
    if (st->name==bst->name)
      p = bst;
      break;
  }
  return p;
}

}  // namespace Albany
