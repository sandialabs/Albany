#ifndef ALBANY_OMEGAH_MESH_FIELD_ACCESSOR_HPP
#define ALBANY_OMEGAH_MESH_FIELD_ACCESSOR_HPP

#include "Albany_AbstractMeshFieldAccessor.hpp"
#include "Albany_DiscretizationUtils.hpp" // Temporary, for NotYetIMplemented exception

#include "Omega_h_mesh.hpp"

namespace Albany {

class OmegahMeshFieldAccessor : public AbstractMeshFieldAccessor
{
public:
  OmegahMeshFieldAccessor (const Teuchos::RCP<Omega_h::Mesh>& mesh);
  ~OmegahMeshFieldAccessor () = default;

  void addStateStructs(const StateInfoStruct& sis) override;

  // TODO: move this in the base class?
  void addFieldOnMesh (const std::string& name,
                       const int entityDim,
                       const int numComps);

  void setFieldOnMesh (const std::string& name,
                       const int entityDim,
                       const Teuchos::RCP<const Thyra_MultiVector>& mv);

  // Read from mesh methods
  void fillSolnVector (Thyra_Vector&        /* soln */,
                       const dof_mgr_ptr_t& /* sol_dof_mgr */,
                       const bool           /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::fillSolnVector");
  }

  void fillVector (Thyra_Vector&        field_vector,
                   const std::string&   field_name,
                   const dof_mgr_ptr_t& field_dof_mgr,
                   const bool           overlapped) override;

  void fillSolnMultiVector (Thyra_MultiVector&   /* soln */,
                            const dof_mgr_ptr_t& /* sol_dof_mgr */,
                            const bool           /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::fillSolnMultiVector");
  }

  void fillSolnSensitivity(Thyra_MultiVector&                    /* dxdp */,
                           const Teuchos::RCP<const DOFManager>& /* solution_dof_mgr */,
                           const bool                            /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::fillSolnSensitivity");
  }

  // Write to mesh methods
  void saveVector (const Thyra_Vector&  field_vector,
                   const std::string&   field_name,
                   const dof_mgr_ptr_t& field_dof_mgr,
                   const bool           overlapped) override;

  void saveSolnVector (const Thyra_Vector&  /* soln */,
                       const mv_ptr_t&      /* soln_dxdp */,
                       const dof_mgr_ptr_t& /* sol_dof_mgr */,
                       const bool           /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::saveSolnVector");
  }

  void saveSolnVector (const Thyra_Vector&  /* soln */,
                       const mv_ptr_t&      /* soln_dxdp */,
                       const Thyra_Vector&  /* soln_dot */,
                       const dof_mgr_ptr_t& /* sol_dof_mgr */,
                       const bool           /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::saveSolnVector");
  }

  void saveSolnVector (const Thyra_Vector&  /* soln */,
                       const mv_ptr_t&      /* soln_dxdp */,
                       const Thyra_Vector&  /* soln_dot */,
                       const Thyra_Vector&  /* soln_dotdot */,
                       const dof_mgr_ptr_t& /* sol_dof_mgr */,
                       const bool           /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::saveSolnVector");
  }

  void saveResVector (const Thyra_Vector&  /* res */,
                      const dof_mgr_ptr_t& /* dof_mgr */,
                      const bool           /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::saveResVector");
  }

  void saveSolnMultiVector (const Thyra_MultiVector& /* soln */,
                            const mv_ptr_t&          /* soln_dxdp */,
                            const dof_mgr_ptr_t&     /* node_vs */,
                            const bool               /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::saveSolnMultiVector");
  }

protected:
  Teuchos::RCP<Omega_h::Mesh>   m_mesh;
};

} // namespace Albany

#endif // ALBANY_OMEGAH_MESH_FIELD_ACCESSOR_HPP
