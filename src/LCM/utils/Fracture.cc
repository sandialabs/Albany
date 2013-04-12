//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Fracture.h"

#include <cassert>
#include "Teuchos_ScalarTraits.hpp"

namespace LCM{

/**
 * \brief Default constructor for generic fracture criteria
 */

GenericFractureCriterion::GenericFractureCriterion(int numDim_, EntityRank& rank) :
  AbstractFractureCriterion(numDim_, rank)
{
}

/**
 * \brief Generic fracture criterion function.
 *
 * \param[in] entity
 * \param[in] probability
 * \return is criterion met
 *
 * Given an entity and probability, will determine if fracture criterion
 * is met. Will return true if fracture criterion is met, else false.
 * Fracture only defined on surface of elements. Thus, input entity
 * must be of rank dimension-1, else error. For 2D, entity rank must = 1.
 * For 3D, entity rank must = 2.
 */
bool
GenericFractureCriterion::fracture_criterion(
		Entity& entity,
		double p)
{

	// Fracture only defined on the boundary of the elements
	EntityRank rank = entity.entity_rank();
	assert(rank==numDim-1);

	stk::mesh::PairIterRelation relations = entity.relations(elementRank);
	if(relations.size()==1)
		return false;

	bool is_open = false;
	// Check criterion
	double random = 0.5 + 0.5*Teuchos::ScalarTraits<double>::random();
	if (random < p){
		is_open = true;
	}

	return is_open;
}

} // namespace LCM

