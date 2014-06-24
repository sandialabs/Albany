//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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
StressFracture::StressFracture(int num_dim, stk_classic::mesh::EntityRank& element_rank,
                               const std::vector<std::vector<double> >& stresses,
                               double crit_stress,
                               Albany::STKDiscretization& stk) :
  AbstractFractureCriterion(num_dim, element_rank),
  avg_stresses_(stresses),
  critical_stress_(crit_stress),
  stk_(stk) {
}

//----------------------------------------------------------------------------
//
// Stress fracture criterion function.
//
bool
StressFracture::computeFractureCriterion(stk_classic::mesh::Entity& entity, double p) {
  // Fracture only defined on the boundary of the elements
  stk_classic::mesh::EntityRank rank = entity.entity_rank();
  assert(rank == num_dim_ - 1);

  stk_classic::mesh::PairIterRelation neighbor_elems =
    entity.relations(element_rank_);

  // Need an element on each side of the edge
  if(neighbor_elems.size() != 2)
    return false;

  // Note that these are element GIDs

  stk_classic::mesh::EntityId elem_0_Id =
    neighbor_elems[0].entity()->identifier();
  stk_classic::mesh::EntityId elem_1_Id =
    neighbor_elems[1].entity()->identifier();

  Albany::WsLIDList& elemGIDws = stk_.getElemGIDws();

  // Have two elements, one on each size of the edge (or
  // face). Check and see if the stresses are such that we want to
  // split the mesh here.
  //
  // Initial hack - GAH: if the average stress between two elements
  // is above the input value "Fracture Stress", split them at the
  // edge

  bool is_open = false;

  // Check criterion
  // If average between cells is above crit, split
  //  if (0.5 * (avg_stresses[elemGIDws[elem_0_Id].ws][elemGIDws[elem_0_Id].LID] +
  //    avg_stresses[elemGIDws[elem_1_Id].ws][elemGIDws[elem_1_Id].LID]) >= crit_stress){
  // If stress difference across face it above crit, split
  //  if (fabs(avg_stresses[elemGIDws[elem_0_Id].ws][elemGIDws[elem_0_Id].LID] -
  //    avg_stresses[elemGIDws[elem_1_Id].ws][elemGIDws[elem_1_Id].LID]) >= crit_stress){
  // Just split the doggone elements already!!!
  if(p == 5) {
    if((elem_0_Id - 1 == 35 && elem_1_Id - 1 == 140) ||
        (elem_1_Id - 1 == 35 && elem_0_Id - 1 == 140)) {

      is_open = true;

      std::cout << "Splitting elements " << elem_0_Id - 1 << " and " << elem_1_Id - 1 << std::endl;
      //std::cout << avg_stresses[elemGIDws[elem_0_Id].ws][elemGIDws[elem_0_Id].LID] << " " <<
      //    avg_stresses[elemGIDws[elem_1_Id].ws][elemGIDws[elem_1_Id].LID] << std::endl;

    }
  }

  else if(p == 10) {
    if((elem_0_Id - 1 == 42 && elem_1_Id - 1 == 147) ||
        (elem_1_Id - 1 == 42 && elem_0_Id - 1 == 147)) {

      is_open = true;

      std::cout << "Splitting elements " << elem_0_Id - 1 << " and " << elem_1_Id - 1 << std::endl;
    }
  }

  else if(p == 15) {
    if((elem_0_Id - 1 == 49 && elem_1_Id - 1 == 154) ||
        (elem_1_Id - 1 == 49 && elem_0_Id - 1 == 154)) {

      is_open = true;

      std::cout << "Splitting elements " << elem_0_Id - 1 << " and " << elem_1_Id - 1 << std::endl;
    }
  }

  return is_open;
}
//----------------------------------------------------------------------------
} // namespace LCM

