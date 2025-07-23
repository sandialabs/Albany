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

  switch (st->entity) {
    case StateStruct::ElemData:   [[falthrough]];
    case StateStruct::QuadPoint:  [[falthrough]];
    case StateStruct::ElemNode:
      elem_sis.push_back(st); break;
    case StateStruct::WorksetValue:
      global_sis.push_back(st); break;
    case StateStruct::NodalData:            [[fallthrough]];
    case StateStruct::NodalDataToElemNode:  [[fallthrough]];
    case StateStruct::NodalDistParameter:
      nodal_sis.push_back(st); break;
    default:
      throw std::runtime_error("Unrecognized/unsupported StateInfo entity.\n");
  }

  if (st->extruded) {
    // This state is either extruded from basal mesh or interpolated from a layered basal state
    // Either way, we must have a state with the same name
    auto bst = find_basal_state(st);
    if (bst->layered) {
      // We are extruding, so nothing to do. The 3d state will be 
    }
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
  int num_ws = workset_sizes.size();
  for (auto st : elem_sis) {
    auto bst = find_basal_state(st);
    if (not bst.is_null()) {
      // We only allow name clashing if this field was extruded/interpolated from a basal one
      TEUCHOS_TEST_FOR_EXCEPTION (not bst->extruded, std::runtime_error,
          "[ExtrudedMeshFieldAccessor::createStateArrays] Error! Non-extruded state name clashes with basal disc state name.\n"
          "  - state name: " + bst->name + "\n");
      for (int ws=0; ws<num_ws; ++ws) {
        auto& state = elemStateArrays[ws];
        if (bst->layered) {
          // The basal state will be interpolated
        } else {
          // The basal state will be extruded
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
