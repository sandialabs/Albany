#include "Albany_DistributedParameter.hpp"
#include "Albany_ThyraUtils.hpp"

#include <set>

namespace Albany
{

void DistributedParameter::
compute_elem_dof_lids(const std::string& mesh_part) {
  m_elem_dof_lids = m_dof_mgr->restrict(mesh_part);

  Teuchos::RCP<const Thyra_VectorSpace> owned_vs, overlapped_vs;
  bool restricted = m_elem_dof_lids.dev().data()!=m_dof_mgr->elem_dof_lids().dev().data();
  if (restricted) {
    // This param is defined on a subset of the dof mgr element block.
    // We need to reconstruct the vector spaces
    auto comm = m_dof_mgr->getAlbanyComm();

    auto elem_dof_lids = m_elem_dof_lids.host();

    Teuchos::Array<GO> owned, owned_or_ghosted, ghosted;
    for (int ielem=0; ielem<elem_dof_lids.extent_int(0); ++ielem) {
      const auto& gids = m_dof_mgr->getElementGIDs(ielem);
      for (int idof=0; idof<elem_dof_lids.extent_int(1); ++idof) {
        if (elem_dof_lids(ielem,idof)==-1) {
          continue;
        }

        // Check if this GID is owned or ghosted
        if (m_dof_mgr->indexer()->isLocallyOwnedElement(gids[idof])) {
          owned.push_back(gids[idof]);
          ghosted.push_back(gids[idof]);
        } else {
          ghosted.push_back(gids[idof]);
        }
      }
    }

    // Make sure ghosted come after owned
    owned_or_ghosted = owned;
    for (auto g : ghosted)
      owned_or_ghosted.push_back(g);

    owned_vs = createVectorSpace(comm,owned);
    overlapped_vs = createVectorSpace(comm,owned_or_ghosted);

    // Correct the LIDs stored in m_elem_dof_lids, to reflect the restricted maps
    for (int ielem=0; ielem<elem_dof_lids.extent_int(0); ++ielem) {
      const auto& gids = m_dof_mgr->getElementGIDs(ielem);
      for (int idof=0; idof<elem_dof_lids.extent_int(1); ++idof) {
        if (elem_dof_lids(ielem,idof)==-1) {
          continue;
        }
        m_elem_dof_lids.host()(ielem,idof) = m_dof_mgr->ov_indexer()->getLocalElement(gids[idof]);
      }
    }
  } else {
    // This parameter is defined in the whole element block,
    // so simply use the dof manager vector spaces
    owned_vs = m_dof_mgr->vs();
    overlapped_vs = m_dof_mgr->ov_vs();
  }

  owned_vec = Thyra::createMember(owned_vs);
  overlapped_vec = Thyra::createMember(overlapped_vs);

  lower_bounds_vec = Thyra::createMember(owned_vs);
  upper_bounds_vec = Thyra::createMember(owned_vs);

  cas_manager = createCombineAndScatterManager(owned_vs, overlapped_vs);
}

} // namespace Albany
