//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_StressFracture.hpp"

#include <cassert>
#include "Teuchos_ScalarTraits.hpp"

namespace AAdapt {

//----------------------------------------------------------------------------
//
// Default constructor for stress fracture criteria
//
StressFracture::StressFracture(
    int                                     num_dim,
    stk::mesh::EntityRank&                  element_rank,
    const std::vector<std::vector<double>>& stresses,
    double                                  crit_stress,
    Albany::STKDiscretization&              stk)
    : AbstractFailureCriterion(num_dim, element_rank),
      avg_stresses_(stresses),
      critical_stress_(crit_stress),
      stk_(stk)
{
}

//----------------------------------------------------------------------------
//
// Stress fracture criterion function.
//
bool
StressFracture::computeFractureCriterion(stk::mesh::Entity entity, double p)
{
  // Fracture only defined on the boundary of the elements
  stk::mesh::EntityRank rank = entity.entity_rank();
  assert(rank == num_dim_ - 1);

  stk::mesh::PairIterRelation neighbor_elems = entity.relations(element_rank_);

  // Need an element on each side of the edge
  if (neighbor_elems.size() != 2) return false;

  // Note that these are element GIDs

  stk::mesh::EntityId elem_0_Id = neighbor_elems[0].entity()->identifier();
  stk::mesh::EntityId elem_1_Id = neighbor_elems[1].entity()->identifier();

  Albany::WsLIDList& elemGIDws = stk_.getElemGIDws();

  // Have two elements, one on each size of the edge (or
  // face). Check and see if the stresses are such that we want to
  // split the mesh here.

  bool is_open = false;

  // Check criterion
  // If average between cells is above critical, split
  // TODO: Not implemented yet.
  return is_open;
}
//----------------------------------------------------------------------------
}  // namespace AAdapt
