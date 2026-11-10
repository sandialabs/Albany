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

  // 3d nodal states are stored ONLY as (Cell,Node[,...]) elem states, and there is
  // no 3d mesh database holding nodal fields, so there is nothing to transfer.
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

  // --- Layered (3d) solution <-> basal tag packing -------------------------------
  //
  // The whole 3d solution is stored on the BASAL mesh: each basal vertex carries the
  // entire column above it, as a tag of neq*(numLayers+1) components. The mapping
  // between a 3d dof and a basal tag slot is NOT implied by either dof manager --
  // the basal solution dof mgr just has neq*nlev anonymous 'cmp_i' fields (see
  // create_dof_mgr), and Panzer numbers those node-major within a basal element.
  // Handing the 3d solution vector straight to the basal accessor therefore packs
  // dofs belonging to the OTHER nodes of a basal element into a vertex's column.
  //
  // basal_cmp is the single definition of that mapping, used by both directions.
  // Layer-major, to match LayeredMeshOrdering::LAYER.
  int basal_cmp (const int ilev, const int eq) const { return ilev*m_neq + eq; }

  // Dependencies the packing needs, supplied by ExtrudedDiscretization once the
  // extruded dof managers exist (they do not at construction time).
  void setLayeredSolutionInfo (const Teuchos::RCP<const LayeredMeshNumbering<LO>>& node_numbering_lid,
                               const Teuchos::RCP<const DOFManager>& basal_node_dof_mgr,
                               const Teuchos::RCP<const DOFManager>& basal_sol_dof_mgr,
                               const Teuchos::RCP<const DOFManager>& sol_dof_mgr,
                               const int neq);

  // Project a 3d solution vector into the layout of the BASAL solution dof manager.
  // The result is a vector over m_basal_sol_dof_mgr->ov_vs(), which is what the
  // basal discretization's own routines (writeSolutionToMeshDatabase,
  // checkForAdaptation, ...) expect. Handing them the 3d vector instead silently
  // mis-associates dofs with basal nodes -- see the note on basal_cmp above.
  Teuchos::RCP<Thyra_Vector>
  projectSolutionToBasal (const Thyra_Vector& soln,
                          const bool          overlapped) const;

  // Inverse of projectSolutionToBasal. basal_soln may be over either the owned or
  // the overlapped basal solution space; its entries are located by GID.
  void projectBasalToSolution (const Thyra_Vector& basal_soln,
                               Thyra_Vector&       soln,
                               const bool          overlapped) const;

  // Maps a (3d) element LID to the workset that owns it, and its index within that
  // workset. Needed to scatter values computed from the layered numbering (which is
  // defined over all the elements on the rank) into the per-workset state arrays.
  void setElemWorksetIdx (const std::vector<WsIdx>& elem_ws_idx) { m_elem_ws_idx = elem_ws_idx; }
protected:

  // This class will rely on the basal mesh to store fields
  Teuchos::RCP<AbstractMeshFieldAccessor> m_basal_field_accessor;

  Teuchos::RCP<const LayeredMeshNumbering<LO>>  m_elem_numbering_lid;

  // Where a 3d element lives: the workset that owns it, and its index in that workset
  struct WsLoc {
    int ws;
    int idx;
  };
  WsLoc locate3dElem (const int elem_lid, const char* caller) const;

  DualView<int**>     m_workset_elements;
  WorksetArray<int>   m_ws_sizes;
  std::vector<WsIdx>  m_elem_ws_idx;

  // Set by setLayeredSolutionInfo; needed by the basal <-> 3d solution projections.
  Teuchos::RCP<const LayeredMeshNumbering<LO>> m_node_numbering_lid;
  Teuchos::RCP<const DOFManager>               m_basal_node_dof_mgr;
  Teuchos::RCP<const DOFManager>               m_basal_sol_dof_mgr;
  Teuchos::RCP<const DOFManager>               m_sol_dof_mgr;
  int                                          m_neq = -1;

  // Shared body of the basal <-> 3d solution projections: walks (basal elem, layer, eq, elem node)
  // and calls f(dof_lid, basal_node_lid, ilev, eq, ibelem, n) for every 3d dof of the
  // column. Both directions use this one walk, so they can only disagree via
  // basal_cmp -- which is a single expression.
  template<typename Func>
  void forEachColumnDof (const bool overlapped, Func&& f) const;
};

}  // namespace Albany

#endif  // ALBANY_EXTRUDED_MESH_FIELD_ACCESSOR_HPP
