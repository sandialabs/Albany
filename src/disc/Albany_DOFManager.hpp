//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DOF_MANAGER_HPP
#define ALBANY_DOF_MANAGER_HPP

#include "Albany_ScalarOrdinalTypes.hpp"

#include "Kokkos_Macros.hpp"

namespace Albany {

/*
 * DOFManager: a lightweight struct to map entity_id->dof_id
 *
 * A DOFManager produces a global/local ID for a DOF given the
 * global/local ID of the entity where the DOF is defined.
 * Depending on whether the dof components are interleaved,
 * the relation between entity and dof id is
 *
 *  - interleaved:     dof_id = entity_id*num_comp + icomp
 *  - non-interleaved: dof_id = entity_id + (max_entity_id+1)*icomp
 *
 * Note: id mapping routines are device friendly
 */

class DOFManager {
public:
  DOFManager ()  = default;

  void setup (const int numComponents, const LO numLocalEntities,
              const GO maxGlobalEntityIdID, const bool interleaved) {
    m_numComponents = numComponents;
    m_numLocalEntities = numLocalEntities;
    m_maxGlobalEntityIdIDp1 = maxGlobalEntityIdID + 1;
    m_interleaved = interleaved;
  }

  KOKKOS_INLINE_FUNCTION
  LO getLocalDOF(LO iEntity, int icomp) const {
    if (m_interleaved) {
      return iEntity*m_numComponents + icomp;
    } else {
      return iEntity + m_numLocalEntities*icomp;
    }
  }

  KOKKOS_INLINE_FUNCTION
  GO getGlobalDOF(GO node, int icomp) const {
    if (m_interleaved) {
      return node*m_numComponents + icomp;
    } else {
      return node + m_maxGlobalEntityIdIDp1*icomp;
    }
  }

  KOKKOS_INLINE_FUNCTION
  int numComponents() const {
    return m_numComponents;
  }

private:
  int       m_numComponents;
  LO        m_numLocalEntities;
  GO        m_maxGlobalEntityIdIDp1;
  bool      m_interleaved;
};

} // namespace Albany

#endif // ALBANY_DOF_MANAGER_HPP
