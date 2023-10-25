//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACT_MESH_FIELD_ACCESSOR_HPP
#define ALBANY_ABSTRACT_MESH_FIELD_ACCESSOR_HPP

#include "Albany_ThyraTypes.hpp"
#include "Albany_DOFManager.hpp"
#include "Albany_StateInfoStruct.hpp"

#include "Teuchos_RCP.hpp"

namespace Albany {

/*
 * Base interface class to access data on mesh
 *
 * The user can call the methods of this class to extract/write
 * data from/to te underlying mesh into/from Thyra (multi) vectors.
 * The (multi) vectors may refer to either owned or overlapped
 * partitions.
 */

class AbstractMeshFieldAccessor
{
public:

  using dof_mgr_ptr_t = Teuchos::RCP<const DOFManager>;
  using mv_ptr_t      = Teuchos::RCP<const Thyra_MultiVector>;

  //! Destructor
  virtual ~AbstractMeshFieldAccessor () = default;

  // Add states to mesh (and possibly to nodal_sis/nodal_parameter_sis)
  virtual void
  addStateStructs(const Teuchos::RCP<StateInfoStruct>& sis) = 0;

  const StateInfoStruct& getNodalSIS()          const { return nodal_sis;           }
  const StateInfoStruct& getNodalParameterSIS() const { return nodal_parameter_sis; }

  // Read from mesh methods
  virtual void fillSolnVector (Thyra_Vector&        soln,
                               const dof_mgr_ptr_t& sol_dof_mgr,
                               const bool           overlapped) = 0;

  virtual void fillVector (Thyra_Vector&        field_vector,
                           const std::string&   field_name,
                           const dof_mgr_ptr_t& field_dof_mgr,
                           const bool           overlapped) = 0;

  virtual void fillSolnMultiVector (Thyra_MultiVector&   soln,
                                    const dof_mgr_ptr_t& sol_dof_mgr,
                                    const bool           overlapped) = 0;

  // Write to mesh methods
  virtual void saveVector (const Thyra_Vector&  field_vector,
                           const std::string&   field_name,
                           const dof_mgr_ptr_t& field_dof_mgr,
                           const bool           overlapped) = 0;

  virtual void saveSolnVector (const Thyra_Vector& soln,
                               const mv_ptr_t&     soln_dxdp,
                               const dof_mgr_ptr_t& sol_dof_mgr,
                               const bool           overlapped) = 0;
  virtual void saveSolnVector (const Thyra_Vector&  soln,
                               const mv_ptr_t&      soln_dxdp,
                               const Thyra_Vector&  soln_dot,
                               const dof_mgr_ptr_t& sol_dof_mgr,
                               const bool           overlapped) = 0;

  virtual void saveSolnVector (const Thyra_Vector&  soln,
                               const mv_ptr_t&      soln_dxdp,
                               const Thyra_Vector&  soln_dot,
                               const Thyra_Vector&  soln_dotdot,
                               const dof_mgr_ptr_t& sol_dof_mgr,
                               const bool           overlapped) = 0;

  virtual void saveResVector (const Thyra_Vector&  res,
                              const dof_mgr_ptr_t& dof_mgr,
                              const bool          overlapped) = 0;

  virtual void saveSolnMultiVector (const Thyra_MultiVector& soln,
                                    const mv_ptr_t&          soln_dxdp,
                                    const dof_mgr_ptr_t&     node_vs,
                                    const bool          overlapped) = 0;
protected:
  StateInfoStruct nodal_sis;
  StateInfoStruct nodal_parameter_sis;
};

}  // namespace Albany

#endif  // ALBANY_ABSTRACT_STK_MESH_FIELD_ACCESSOR_HPP
