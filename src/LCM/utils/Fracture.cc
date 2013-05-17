//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <cassert>
#include "Teuchos_ScalarTraits.hpp"

#include "Fracture.h"

namespace LCM{

  //----------------------------------------------------------------------------
  //
  // Default constructor for generic fracture criteria
  //
  GenericFractureCriterion::GenericFractureCriterion(int num_dim, EntityRank& rank) :
    AbstractFractureCriterion(num_dim, rank)
  {
  }

  //----------------------------------------------------------------------------
  //
  // Generic fracture criterion function.
  //
  bool
  GenericFractureCriterion::computeFractureCriterion(Entity& entity,
                                                     double p)
  {
    // Fracture only defined on the boundary of the elements
    EntityRank rank = entity.entity_rank();
    assert( rank == num_dim_-1 );

    stk::mesh::PairIterRelation relations = entity.relations(element_rank_);
    if( relations.size() == 1 )
      return false;

    bool is_open = false;
    // Check criterion
    double random = 0.5 + 0.5*Teuchos::ScalarTraits<double>::random();
    if ( random < p ) {
      is_open = true;
    }
    return is_open;
  }
} // namespace LCM

