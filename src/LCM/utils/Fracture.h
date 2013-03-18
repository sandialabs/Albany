//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

/**
 *
 * This class contains the abstract interface for determining if fracture has occurred between
 * two elements, based on some user defined criteria. Here, we present the abstract interface and
 * a generic default implementation that uses a random number generator to determine if fracture has occurred.
 * 
 */

#ifndef FRACTURE_H_
#define FRACTURE_H_

#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>

namespace LCM{

typedef stk::mesh::Entity Entity;
typedef stk::mesh::EntityRank EntityRank;
typedef stk::mesh::RelationIdentifier EdgeId;
typedef stk::mesh::EntityKey EntityKey;

class AbstractFractureCriterion {

public:

	/**
	 * \brief Default constructor for the criterion object
	 */
	AbstractFractureCriterion(int numDim_, EntityRank& elementRank_) :
    numDim(numDim_), elementRank(elementRank_) {}


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
  virtual
  bool
  fracture_criterion(
                     Entity& entity,
                     double p) = 0;

protected:

  int numDim;
  EntityRank elementRank;

private: // None of these work

   AbstractFractureCriterion();
   AbstractFractureCriterion(const AbstractFractureCriterion &);
   AbstractFractureCriterion &operator=(const AbstractFractureCriterion &);


}; // class AbstractFractureCriterion

class GenericFractureCriterion : public AbstractFractureCriterion {

public:

	/**
	 * \brief Default constructor for the criterion object
	 */
	GenericFractureCriterion(int numDim_, EntityRank& rank);


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
  virtual
	bool
	fracture_criterion(
			Entity& entity,
			double p);

private:

   GenericFractureCriterion();
   GenericFractureCriterion(const GenericFractureCriterion &);
   GenericFractureCriterion &operator=(const GenericFractureCriterion &);


}; // class GenericFractureCriterion


} // namespace LCM

#endif /* FRACTURE_H_ */
