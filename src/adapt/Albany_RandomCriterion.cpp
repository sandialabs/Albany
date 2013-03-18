//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_RandomCriterion.hpp"

#include <cassert>
#include "Teuchos_ScalarTraits.hpp"

namespace LCM{

  /**
   * \brief Default constructor for stress fracture criteria
   */

  RandomCriterion::RandomCriterion(int numDim_, 
                                   EntityRank& elementRank_, 
                                   Albany::STKDiscretization& stk_) :
    AbstractFractureCriterion(numDim_, elementRank_), 
    stk(stk_)
  {
  }

  /**
   * \brief Random fracture criterion function.
   *
   * \param[in] entity
   * \param[in] probability
   * \return is criterion met
   *
   * Given an entity and probability, will determine if fracture criterion
   * is met. Will return true if fracture criterion is met, else false.
   */
  bool
  RandomCriterion::fracture_criterion(Entity& entity,
                                      double p)
  {

    // Fracture only defined on the boundary of the elements
    EntityRank rank = entity.entity_rank();
    std::cout << " HELP: rank: " << rank << std::endl;
    assert(rank==numDim-1);

    stk::mesh::PairIterRelation neighbor_elems = entity.relations(elementRank);

    std::cout << " *** HELP: in fracture_criterion" << std::endl;
    std::cout << " *** HELP: number of neighbors: " << neighbor_elems.size() << std::endl;
    // Need an element on each side
    if(neighbor_elems.size() != 2)
      return false;

    bool is_open = false;

    // All we need to do is generate a number between 0 and 1
    double random = 0.5 + 0.5*Teuchos::ScalarTraits<double>::random();
    std::cout << " *** HELP: random: " << random << ", p: " << std::endl;
    if (random < p){
      is_open = true;
    }

    return is_open;
  }

} // namespace LCM

