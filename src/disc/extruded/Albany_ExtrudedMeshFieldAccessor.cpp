#include "Albany_ExtrudedMeshFieldAccessor.hpp"

namespace Albany {

ExtrudedMeshFieldAccessor::
ExtrudedMeshFieldAccessor (const Teuchos::RCP<AbstractMeshFieldAccessor>& basal_field_accessor,
                           const Teuchos::RCP<LayeredMeshNumbering<LO>>& elem_numbering_lid)
{
}

  // Add states to mesh (and possibly to nodal_sis/nodal_parameter_sis)
void ExtrudedMeshFieldAccessor::
addStateStructs(const StateInfoStruct& sis)
{
}

  // Read from mesh methods
void ExtrudedMeshFieldAccessor::
fillSolnVector (Thyra_Vector&        soln,
                const dof_mgr_ptr_t& sol_dof_mgr,
                const bool           overlapped)
{
}

void ExtrudedMeshFieldAccessor::
fillVector (Thyra_Vector&        field_vector,
                   const std::string&   field_name,
                   const dof_mgr_ptr_t& field_dof_mgr,
                   const bool           overlapped)
{
}

void ExtrudedMeshFieldAccessor::
fillSolnMultiVector (Thyra_MultiVector&   soln,
                            const dof_mgr_ptr_t& sol_dof_mgr,
                            const bool           overlapped)
{
}

void ExtrudedMeshFieldAccessor::
fillSolnSensitivity (Thyra_MultiVector&   dxdp,
                            const dof_mgr_ptr_t& sol_dof_mgr,
                            const bool           overlapped)
{
}

  // Write to mesh methods
void ExtrudedMeshFieldAccessor::
saveVector (const Thyra_Vector&  field_vector,
                   const std::string&   field_name,
                   const dof_mgr_ptr_t& field_dof_mgr,
                   const bool           overlapped)
{
}

void ExtrudedMeshFieldAccessor::
saveSolnVector (const Thyra_Vector& soln,
                const mv_ptr_t&     soln_dxdp,
                const dof_mgr_ptr_t& sol_dof_mgr,
                const bool           overlapped)
{
}

void ExtrudedMeshFieldAccessor::
saveSolnVector (const Thyra_Vector&  soln,
                const mv_ptr_t&      soln_dxdp,
                const Thyra_Vector&  soln_dot,
                const dof_mgr_ptr_t& sol_dof_mgr,
                const bool           overlapped)
{
}

void ExtrudedMeshFieldAccessor::
saveSolnVector (const Thyra_Vector&  soln,
                const mv_ptr_t&      soln_dxdp,
                const Thyra_Vector&  soln_dot,
                const Thyra_Vector&  soln_dotdot,
                const dof_mgr_ptr_t& sol_dof_mgr,
                const bool           overlapped)
{
}

void ExtrudedMeshFieldAccessor::
saveResVector (const Thyra_Vector&  res,
               const dof_mgr_ptr_t& dof_mgr,
               const bool          overlapped)
{
}

void ExtrudedMeshFieldAccessor::
saveSolnMultiVector (const Thyra_MultiVector& soln,
                     const mv_ptr_t&          soln_dxdp,
                     const dof_mgr_ptr_t&     node_vs,
                     const bool          overlapped)
{
}

}  // namespace Albany
