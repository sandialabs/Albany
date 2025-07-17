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
  // TEUCHOS_TEST_FOR_EXCEPTION (basal_field_accessor.is_null(), std::runtime_error,
  //     "[ExtrudedMeshFieldAccessor::ExtrudedMeshFieldAccessor] Error! Invalid basal field accessor pointer.\n");
}

  // Add states to mesh (and possibly to nodal_sis/nodal_parameter_sis)
void ExtrudedMeshFieldAccessor::
addStateStructs(const StateInfoStruct& sis)
{
  for (auto st : sis) {
    std::cout << "adding " << st->name << ", dims: " << util::join(st->dim,",") << "\n";
  }

  std::cout << "basal elem sis:\n";
  for (auto st : m_basal_field_accessor->getElemSIS()) {
    std::cout << "  " << st->name << ", dims: " << util::join(st->dim,",") << "\n";
  }
}

void ExtrudedMeshFieldAccessor::
createStateArrays ()
{
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
fillSolnMultiVector (Thyra_MultiVector&   soln,
                     const dof_mgr_ptr_t& sol_dof_mgr,
                     const bool           overlapped)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::fillSolnMultiVector()");
}

void ExtrudedMeshFieldAccessor::
fillSolnSensitivity (Thyra_MultiVector&   dxdp,
                     const dof_mgr_ptr_t& sol_dof_mgr,
                     const bool           overlapped)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::fillSolnSensitivity()");
}

  // Write to mesh methods
void ExtrudedMeshFieldAccessor::
saveVector (const Thyra_Vector&  field_vector,
            const std::string&   field_name,
            const dof_mgr_ptr_t& field_dof_mgr,
            const bool           overlapped)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::saveVector()");
}

void ExtrudedMeshFieldAccessor::
saveSolnVector (const Thyra_Vector& soln,
                const mv_ptr_t&     soln_dxdp,
                const dof_mgr_ptr_t& sol_dof_mgr,
                const bool           overlapped)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::saveSolnVector()");
}

void ExtrudedMeshFieldAccessor::
saveSolnVector (const Thyra_Vector&  soln,
                const mv_ptr_t&      soln_dxdp,
                const Thyra_Vector&  soln_dot,
                const dof_mgr_ptr_t& sol_dof_mgr,
                const bool           overlapped)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::saveSolnVector()");
}

void ExtrudedMeshFieldAccessor::
saveSolnVector (const Thyra_Vector&  soln,
                const mv_ptr_t&      soln_dxdp,
                const Thyra_Vector&  soln_dot,
                const Thyra_Vector&  soln_dotdot,
                const dof_mgr_ptr_t& sol_dof_mgr,
                const bool           overlapped)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::saveSolnVector()");
}

void ExtrudedMeshFieldAccessor::
saveResVector (const Thyra_Vector&  res,
               const dof_mgr_ptr_t& dof_mgr,
               const bool          overlapped)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::saveResVector()");
}

void ExtrudedMeshFieldAccessor::
saveSolnMultiVector (const Thyra_MultiVector& soln,
                     const mv_ptr_t&          soln_dxdp,
                     const dof_mgr_ptr_t&     node_vs,
                     const bool          overlapped)
{
  throw NotYetImplemented("ExtrudedMeshFieldAccessor::saveSolnMultiVector()");
}

}  // namespace Albany
