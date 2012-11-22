//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_StressFracture.hpp"

#include <cassert>
#include "Teuchos_ScalarTraits.hpp"

namespace LCM{

/**
 * \brief Default constructor for stress fracture criteria
 */

StressFracture::StressFracture(int numDim_, EntityRank& elementRank_, 
    const std::vector<std::vector<double> >& stresses, double crit_stress_, Albany::STKDiscretization& stk_) :
  AbstractFractureCriterion(numDim_, elementRank_), avg_stresses(stresses), crit_stress(crit_stress_), stk(stk_)
{
}

/**
 * \brief Stress fracture criterion function.
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
StressFracture::fracture_criterion(
		Entity& entity,
		float p)
{

	// Fracture only defined on the boundary of the elements
	EntityRank rank = entity.entity_rank();
	assert(rank==numDim-1);

	stk::mesh::PairIterRelation neighbor_elems = entity.relations(elementRank);

  // Need an element on each side of the edge
	if(neighbor_elems.size() != 2)
		return false;

// Note that these are element GIDs

  stk::mesh::EntityId elem_0_Id = neighbor_elems[0].entity()->identifier();
  stk::mesh::EntityId elem_1_Id = neighbor_elems[1].entity()->identifier();

  Albany::WsLIDList& elemGIDws = stk.getElemGIDws();

  // Have two elements, one on each size of the edge (or face). Check and see if the stresses
  // are such that we want to split the mesh here.
  //
  // Initial hack - GAH: if the average stress between two elements is above the input value
  // "Fracture Stress", split them at the edge

	bool is_open = false;
	// Check criterion
  // If average between cells is above crit, split
//	if (0.5 * (avg_stresses[elemGIDws[elem_0_Id].ws][elemGIDws[elem_0_Id].LID] +
//    avg_stresses[elemGIDws[elem_1_Id].ws][elemGIDws[elem_1_Id].LID]) >= crit_stress){
  // If stress difference across face it above crit, split
//	if (fabs(avg_stresses[elemGIDws[elem_0_Id].ws][elemGIDws[elem_0_Id].LID] -
//    avg_stresses[elemGIDws[elem_1_Id].ws][elemGIDws[elem_1_Id].LID]) >= crit_stress){
// Just split the doggone elements already!!!
if(p == 5){
	if ((elem_0_Id - 1 == 35 && elem_1_Id - 1 == 140) ||
	(elem_1_Id - 1 == 35 && elem_0_Id - 1 == 140)){

		is_open = true;

std::cout << "Splitting elements " << elem_0_Id - 1 << " and " << elem_1_Id - 1 << std::endl;
//std::cout << avg_stresses[elemGIDws[elem_0_Id].ws][elemGIDws[elem_0_Id].LID] << " " <<
//    avg_stresses[elemGIDws[elem_1_Id].ws][elemGIDws[elem_1_Id].LID] << std::endl;

	}
} else if (p == 10){
	if ((elem_0_Id - 1 == 42 && elem_1_Id - 1 == 147) ||
	(elem_1_Id - 1 == 42 && elem_0_Id - 1 == 147)){

		is_open = true;

std::cout << "Splitting elements " << elem_0_Id - 1 << " and " << elem_1_Id - 1 << std::endl;
}
} else if (p == 15){
	if ((elem_0_Id - 1 == 49 && elem_1_Id - 1 == 154) ||
	(elem_1_Id - 1 == 49 && elem_0_Id - 1 == 154)){

		is_open = true;

std::cout << "Splitting elements " << elem_0_Id - 1 << " and " << elem_1_Id - 1 << std::endl;
}
}

	return is_open;
}

} // namespace LCM

