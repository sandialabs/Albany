//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_EXTRUDED_MESH_FIELD_ACCESSOR_HPP
#define ALBANY_EXTRUDED_MESH_FIELD_ACCESSOR_HPP

#include "Albany_AbstractMeshFieldAccessor.hpp"
#include "Albany_LayeredMeshNumbering.hpp"
#include "Albany_DiscretizationUtils.hpp"

#include "Teuchos_RCP.hpp"

#include <string>
#include <vector>

namespace Albany {

/*
 * Implementation of a mesh field accessor for extruded meshes
 *
 * This class will fully rely on the basal mesh to store all
 * the extruded fields. The layer ordering will determine how
 * the fields are stored in the basal mesh:
 *  - column ordering: we can store a SINGLE vector field,
 *    since all layers values are contiguous for each basal point
 *  - layer ordering: we store NumLayers separate vector fields,
 *    since the data is NOT contiguous for each basal point
 */

class ExtrudedMeshFieldAccessor : public AbstractMeshFieldAccessor
{
public:
  ExtrudedMeshFieldAccessor (const Teuchos::RCP<AbstractMeshFieldAccessor>& basal_field_accessor,
                             const Teuchos::RCP<const LayeredMeshNumbering<LO>>&  elem_numbering_lid);

  virtual ~ExtrudedMeshFieldAccessor () = default;

  // override, do not hide
  using AbstractMeshFieldAccessor::addStateStructs;

  // Add states to mesh (and possibly to nodal_sis/nodal_parameter_sis)
  void addStateStruct (const Teuchos::RCP<StateStruct>& st) override;

  void createStateArrays (const WorksetArray<int>& worksets_sizes);

  // While 3d states are ALL elem states, the basal MFA needs to call this
  void transferNodeStatesToElemStates ();

  // 3d nodal states are stored ONLY as (Cell,Node[,...]) elem states, since there is
  // no 3d mesh database. For 3d nodal states that are neither extruded nor interpolated,
  // this copies the 3d elem state into the basal layered nodal state registered in
  // addStateStruct, and then into the basal mesh nodal field. Otherwise, it's a no-op.
  void transferElemStateToNodeState (const std::string& name) override;

  // Read from mesh methods
  void fillSolnVector (Thyra_Vector&        soln,
                       const dof_mgr_ptr_t& sol_dof_mgr,
                       const bool           overlapped) override;

  void fillVector (Thyra_Vector&        field_vector,
                   const std::string&   field_name,
                   const dof_mgr_ptr_t& field_dof_mgr,
                   const bool           overlapped) override;

  void fillSolnMultiVector (Thyra_MultiVector&   soln,
                            const dof_mgr_ptr_t& sol_dof_mgr,
                            const bool           overlapped) override;

  void fillSolnSensitivity (Thyra_MultiVector&   dxdp,
                            const dof_mgr_ptr_t& sol_dof_mgr,
                            const bool           overlapped) override;

  // Write to mesh methods
  void saveVector (const Thyra_Vector&  field_vector,
                   const std::string&   field_name,
                   const dof_mgr_ptr_t& field_dof_mgr,
                   const bool           overlapped) override;

  void saveSolnVector (const Thyra_Vector& soln,
                       const mv_ptr_t&     soln_dxdp,
                       const dof_mgr_ptr_t& sol_dof_mgr,
                       const bool           overlapped) override;
  void saveSolnVector (const Thyra_Vector&  soln,
                       const mv_ptr_t&      soln_dxdp,
                       const Thyra_Vector&  soln_dot,
                       const dof_mgr_ptr_t& sol_dof_mgr,
                       const bool           overlapped) override;

  void saveSolnVector (const Thyra_Vector&  soln,
                       const mv_ptr_t&      soln_dxdp,
                       const Thyra_Vector&  soln_dot,
                       const Thyra_Vector&  soln_dotdot,
                       const dof_mgr_ptr_t& sol_dof_mgr,
                       const bool           overlapped) override;

  void saveResVector (const Thyra_Vector&  res,
                      const dof_mgr_ptr_t& dof_mgr,
                      const bool          overlapped) override;

  void saveSolnMultiVector (const Thyra_MultiVector& soln,
                            const mv_ptr_t&          soln_dxdp,
                            const dof_mgr_ptr_t&     node_vs,
                            const bool          overlapped) override;

  void setSolutionFieldsMetadata (const int neq) override;

  void extrudeBasalFields (const Teuchos::Array<std::string>& basal_fields);
  // qp_ref_z: vertical reference coords (in [-1,1]) of the cell cubature points,
  // needed only if some of the fields are 3d QuadPoint states
  void interpolateBasalLayeredFields (const Teuchos::Array<std::string>& basal_fields,
                                      const std::vector<double>& qp_ref_z = {});

  void setWorksetElements (const DualView<int**>& workset_elements) { m_workset_elements = workset_elements; }
protected:

  // This class will rely on the basal mesh to store fields
  Teuchos::RCP<AbstractMeshFieldAccessor> m_basal_field_accessor;

  Teuchos::RCP<const LayeredMeshNumbering<LO>>  m_elem_numbering_lid;

  DualView<int**>     m_workset_elements;
  WorksetArray<int>   m_ws_sizes;

  // 3d states (neither extruded nor interpolated) that are stored in the basal mesh as
  // layered states, and whose normalized layers coords (${name}_NLC) we need to set,
  // so that they can be read back (e.g., for a restart) as basal layered inputs
  std::vector<std::string>  m_basal_layered_nodal_outputs;
  std::vector<std::string>  m_basal_layered_elem_outputs;
};

}  // namespace Albany

#endif  // ALBANY_EXTRUDED_MESH_FIELD_ACCESSOR_HPP
