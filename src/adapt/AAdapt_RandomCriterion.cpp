//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <cassert>

#include <Teuchos_ScalarTraits.hpp>

#include "AAdapt_RandomCriterion.hpp"

namespace AAdapt {

//----------------------------------------------------------------------------
//
// Default constructor
//
RandomCriterion::RandomCriterion(int num_dim,
                                 Albany::STKDiscretization& stk) :
  AbstractFractureCriterion(num_dim, stk::topology::ELEMENT_RANK),
  stk_(stk) {
}

//----------------------------------------------------------------------------
//
// Random fracture criterion function.
//
bool
RandomCriterion::computeFractureCriterion(stk::mesh::Entity entity, double p) {

  // Fracture only defined on the boundary of the elements
  stk::mesh::EntityRank rank = entity.entity_rank();
  assert(rank == num_dim_ - 1);

  stk::mesh::PairIterRelation neighbor_elems =
    entity.relations(stk::topology::ELEMENT_RANK);

  // Need an element on each side
  if(neighbor_elems.size() != 2)
    return false;

  bool is_open = false;

  // All we need to do is generate a number between 0 and 1
  double random = 0.5 + 0.5 * Teuchos::ScalarTraits<double>::random();

  if(random < p) {
    is_open = true;
  }

  return is_open;
}
//----------------------------------------------------------------------------
} // namespace AAdapt

