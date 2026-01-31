#include "OmegahGhost.hpp"
#include <Omega_h_map.hpp> //unmap, fan_reduce
#include <Omega_h_mark.hpp> //collect_marked
#include <Omega_h_array_ops.hpp> //get_sum
#include <Omega_h_element.hpp> //simplex_degree, hypercube_degree
#include <Omega_h_adj.hpp> //invert_adj
#include <Omega_h_int_scan.hpp> //offset_scan
#include <Omega_h_for.hpp> //parallel_for

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

  Omega_h::Read<Omega_h::GO> getEntGidsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh, int dim) {
    auto mesh = const_cast<Omega_h::Mesh&>(cmesh);
    OMEGA_H_CHECK(dim >= 0 && dim <= mesh.dim());
    auto isInClosure = getEntsInClosureOfOwnedElms(mesh, dim);
    auto globals_d = mesh.globals(dim);
    auto keptIndicies_d = Omega_h::collect_marked(isInClosure);
    auto ownedGlobals_d = Omega_h::unmap(keptIndicies_d, globals_d, 1);
    return ownedGlobals_d;
  }

  //return the local id (index) for entities of the specified dimension in the closure of owned elements
  //if an entity is not in the closure its local id is -1
  Omega_h::LOs getEntLidsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh, int dim) {
    auto mesh = const_cast<Omega_h::Mesh&>(cmesh);
    OMEGA_H_CHECK(dim >= 0 && dim <= mesh.dim());
    OMEGA_H_CHECK(mesh.has_tag(dim, "global"));
    auto isEntInClosure = getEntsInClosureOfOwnedElms(cmesh, dim);
    auto offset = Omega_h::offset_scan(isEntInClosure);
    auto lids = Omega_h::Write<Omega_h::LO>(isEntInClosure.size(), -1);
    Omega_h::parallel_for(isEntInClosure.size(), OMEGA_H_LAMBDA(const Omega_h::LO &i) {
      if( isEntInClosure[i] ) {
        lids[i] = offset[i];
      }
    });
    return lids;
  }

  Omega_h::Read<Omega_h::I8> getOwnedEntsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh, int dim) {
    auto mesh = const_cast<Omega_h::Mesh&>(cmesh);
    OMEGA_H_CHECK(dim >= 0 && dim <= mesh.dim());
    auto isInClosure = getEntsInClosureOfOwnedElms(mesh, dim);
    auto owned_d = mesh.owned(dim);
    auto keptIndicies_d = Omega_h::collect_marked(isInClosure);
    auto ownedInClosure_d = Omega_h::unmap(keptIndicies_d, owned_d, 1);
    return ownedInClosure_d;
  }

  Omega_h::Reals getVtxCoordsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh) {
    auto mesh = const_cast<Omega_h::Mesh&>(cmesh);
    auto isInClosure = getEntsInClosureOfOwnedElms(mesh, Omega_h::VERT);
    auto coords_d = mesh.coords();
    auto keptIndicies_d = Omega_h::collect_marked(isInClosure);
    auto ownedCoords_d = Omega_h::unmap(keptIndicies_d, coords_d, mesh.dim());
    return ownedCoords_d;
  }

  //returns downward entity indexing in the unfiltered/ghosted mesh that can be used to access
  // tags/arrays on the ghosted mesh.
  //The implicit element indices are in the filtered/unghosted mesh (i.e., ghost
  //elements are removed).
  Omega_h::LOs getDownAdjacentEntsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh, int dim) {
    auto mesh = const_cast<Omega_h::Mesh&>(cmesh);
    OMEGA_H_CHECK(dim >= 0 && dim < mesh.dim());
    const auto isSimplex = (mesh.family() == OMEGA_H_SIMPLEX);
    const auto entsPerElm = isSimplex ? Omega_h::simplex_degree(mesh.dim(),dim) :
                                        Omega_h::hypercube_degree(mesh.dim(),dim);
    auto elmToDown_d = mesh.ask_down(mesh.dim(),dim).ab2b;
    auto isOwnedElm = mesh.owned(mesh.dim());
    auto keptIndicies_d = Omega_h::collect_marked(isOwnedElm);
    auto ownedElmToDown_d = Omega_h::unmap(keptIndicies_d, elmToDown_d, entsPerElm);
    return ownedElmToDown_d;
  }

  //returns owned upward adjacent elements from *all* entities of dimension dim
  //on the part. If an entity (dimension=dim) is not in the closure of an owned
  //element it will have no upward adjacent elements in the returned Graph.
  //The returned graph has indexing into the unfiltered/ghosted mesh (i.e.,
  //ghost entities are counted).
  Omega_h::Graph getUpAdjacentEntsInClosureOfOwnedElms(const Omega_h::Mesh& cmesh, int dim) {
    auto mesh = const_cast<Omega_h::Mesh&>(cmesh);
    OMEGA_H_CHECK(dim >= 0 && dim < mesh.dim());
    auto entToElm = mesh.ask_up(dim, mesh.dim());
    auto isOwnedElm = mesh.owned(mesh.dim());
    // create marks for each elm to indicate if it should be kept
    auto elmsToKeep = Omega_h::Write<Omega_h::I8>(entToElm.ab2b.size(),0);
    auto ab2b = entToElm.ab2b;
    Omega_h::parallel_for(ab2b.size(), OMEGA_H_LAMBDA(const Omega_h::LO &i) {
      const auto candidateElm = ab2b[i];
      elmsToKeep[i] = isOwnedElm[candidateElm];
    });
    // count number of upward adj elms being kept per entity
    auto oldOffsets = entToElm.a2ab;
    const auto numEnts = oldOffsets.size()-1;
    auto filteredEntDegree = Omega_h::Write<Omega_h::LO>(numEnts,0);
    Omega_h::parallel_for(numEnts, OMEGA_H_LAMBDA(const Omega_h::LO &i) {
      for(int j=oldOffsets[i]; j<oldOffsets[i+1]; j++) {
        filteredEntDegree[i] += elmsToKeep[j];
      }
    });
    // create offset array for filtered elements
    auto offset = Omega_h::offset_scan(Omega_h::Read(filteredEntDegree));
    // copy elms to new 'values' array that are being kept
    auto values = Omega_h::Write<Omega_h::LO>(offset.last(), 0);
    // fill the 'values' array
    Omega_h::parallel_for(numEnts, OMEGA_H_LAMBDA(const Omega_h::LO &i) {
      int newUpIdx = offset[i];
      for(int j=oldOffsets[i]; j<oldOffsets[i+1]; j++) {
        if( elmsToKeep[j] ) {
          auto elm = ab2b[j];
          assert(newUpIdx < offset[i+1]);
          values[newUpIdx] = elm;
          newUpIdx++;
        }
      }
    });
    return Omega_h::Graph({offset,values});
  }

  //return an array sized for the owned elements on this process
  //the ith entry of the array contains the position of the ith owned
  //element in the array of ghosted and non-ghosted elements
  Omega_h::LOs getElemPermutationFromNonGhostedToGhosted(Omega_h::Mesh &mesh) {
    auto elmDim = mesh.dim();
    auto isElmOwned = mesh.owned(elmDim);
    const auto numOwned = getNumOwnedElms(mesh);
    auto exscan = Omega_h::offset_scan(isElmOwned);
    auto perm = Omega_h::Write<Omega_h::LO>(numOwned);
    Omega_h::parallel_for(mesh.nelems(), OMEGA_H_LAMBDA(const Omega_h::LO &i) {
      if( isElmOwned[i] ) {
        const auto idx = exscan[i];
        assert(idx < numOwned);
        perm[idx] = i;
      }
    });
    return Omega_h::read(perm);
  }
} //end OmegahGhost namespace
