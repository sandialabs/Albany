#include "OmegahGhost.hpp"
#include <Omega_h_map.hpp> //unmap, fan_reduce
#include <Omega_h_mark.hpp> //collect_marked
#include <Omega_h_array_ops.hpp> //get_sum

namespace OmegahGhost {

  Omega_h::LO getNumOwnedElms(const Omega_h::Mesh& cmesh) {
    auto mesh = const_cast<Omega_h::Mesh&>(cmesh);
    auto elmDim = mesh.dim();
    auto isElmOwned = mesh.owned(elmDim);
    return Omega_h::get_sum(isElmOwned);
  }

  Omega_h::HostRead<Omega_h::GO> getOwnedEntityGids(const Omega_h::Mesh& cmesh, int dim) {
    auto mesh = const_cast<Omega_h::Mesh&>(cmesh);
    OMEGA_H_CHECK(dim >= 0 && dim <= mesh.dim());
    OMEGA_H_CHECK(mesh.has_tag(dim, "global"));
    auto globals_d = mesh.globals(dim);
    auto owned_d = mesh.owned(dim);
    auto keptIndicies_d = Omega_h::collect_marked(owned_d);
    auto ownedGlobals_d = Omega_h::unmap(keptIndicies_d, globals_d, 1);
    return Omega_h::HostRead<Omega_h::GO>(ownedGlobals_d);
  }

  /**
   * \brief Get entities in the closure of owned elements
   *
   * This function identifies mesh entities that are in the closure of owned elements,
   * matching the behavior of element-based partitions without ghosts. An entity is
   * considered to be in the closure of owned elements if it bounds at least one
   * owned element.
   *
   * \param[in] cmesh The Omega_h mesh (const reference)
   * \param[in] dim The topological dimension of entities to query (0=vertices, 1=edges,
   *                2=faces, 3=regions). Must be in range [0, 3].
   *
   * \return A mask array (Omega_h::Read<Omega_h::I8>) where each entry corresponds to
   *         an entity of dimension \p dim. The value is 1 if the entity is in the
   *         closure of at least one owned element, 0 otherwise. The array has length
   *         equal to the number of entities of dimension \p dim in the mesh.
   *
   * \throws std::logic_error if dim < 0 or dim > 3
   * \throws std::logic_error if the computed mask size doesn't match the number of entities
   *
   * \note For entities with dimension equal to the mesh dimension (elements), this
   *       simply returns the owned mask from the mesh.
   *
   * \note For entities with dimension less than the mesh dimension, the function:
   *       - Queries upward adjacencies (entity-to-element)
   *       - Maps element ownership status to entity adjacencies
   *       - Uses fan_reduce with MAX to determine if any adjacent element is owned
   */
  Omega_h::Read<Omega_h::I8> getEntsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh, int dim) {
    //Omegah isn't very const friendly and the Teuchos RCP returns const pointers/refs
    auto mesh = const_cast<Omega_h::Mesh&>(cmesh);
    std::stringstream ss;
    OMEGA_H_CHECK(dim >= 0 && dim <= mesh.dim());
    const auto elmDim = mesh.dim();
    if( dim == elmDim ) {
      return mesh.owned(elmDim);
    } else {
      auto entToElm = mesh.ask_up(dim, elmDim);
      auto isElmOwned = mesh.owned(elmDim);
      //create an array that replaces the element indices that are adjacent to
      //each ent with the ownership status of the element (1:owned, 0:o.w.)
      auto entToOwned = Omega_h::unmap(entToElm.ab2b, isElmOwned, 1);
      // For each ent of dimension dim, where dim!=meshDim, get max ownership among adjacent elements
      // (will be 1 if any element is owned, 0 otherwise)
      auto entHasOwned = Omega_h::fan_reduce(entToElm.a2ab, Omega_h::read(entToOwned),1,OMEGA_H_MAX);
      OMEGA_H_CHECK(entHasOwned.size() == mesh.nents(dim));
      return entHasOwned;
    }
  }

  //Tell Albany about entities that are in the closure of owned elements.  This
  //matches what was done in an element based partition without ghosts.
  Omega_h::LO getNumEntsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh, int dim) {
    //Omegah isn't very const friendly and the Teuchos RCP returns const pointers/refs
    auto mesh = const_cast<Omega_h::Mesh&>(cmesh);
    OMEGA_H_CHECK(dim >= 0 && dim <= mesh.dim());
    auto entHasOwnedAdjElm = getEntsInClosureOfOwnedElms(cmesh,dim);
    return Omega_h::get_sum(entHasOwnedAdjElm);
  }

  Omega_h::HostRead<Omega_h::GO> getEntGidsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh, int dim) {
    auto mesh = const_cast<Omega_h::Mesh&>(cmesh);
    OMEGA_H_CHECK(dim >= 0 && dim <= mesh.dim());
    auto isInClosure = getEntsInClosureOfOwnedElms(mesh, dim);
    auto globals_d = mesh.globals(dim);
    auto keptIndicies_d = Omega_h::collect_marked(isInClosure);
    auto ownedGlobals_d = Omega_h::unmap(keptIndicies_d, globals_d, 1);
    return Omega_h::HostRead<Omega_h::GO>(ownedGlobals_d);
  }

} //end OmegahGhost namespace
