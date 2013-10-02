//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

///
/// Fracture criteria classes are required to have a method
/// called check that takes as argument an entity and returns a bool.
///

#if !defined(LCM_Fracture_h)
#define LCM_Fracture_h

#include <cassert>

#include "Teuchos_ScalarTraits.hpp"
#include "Topology_Types.h"

namespace LCM{

///
/// Base class for fracture criteria
///
class AbstractFractureCriterion {

public:

  AbstractFractureCriterion() {}

  virtual
  bool
  check(Entity const & entity) const = 0;

  virtual
  ~AbstractFractureCriterion() {}

private:

  AbstractFractureCriterion(const AbstractFractureCriterion &);
  AbstractFractureCriterion &operator=(const AbstractFractureCriterion &);

};

///
/// Random fracture criterion given a probability of failure
///
class FractureCriterionRandom : public AbstractFractureCriterion {

public:

  explicit
  FractureCriterionRandom(
      EntityRank const space_dimension,
      double const probability) :
  AbstractFractureCriterion(),
  space_dimension_(space_dimension), probability_(probability) {}

  bool
  check(Entity const & entity) const
  {
    EntityRank const rank = entity.entity_rank();
    assert(rank == space_dimension_ - 1);

    stk::mesh::PairIterRelation const
    relations = entity.relations(space_dimension_);

    assert(relations.size() == 2);

    double const
    random = 0.5 * Teuchos::ScalarTraits<double>::random() + 0.5;

    return random < probability_;
  }

private:

  FractureCriterionRandom();
  FractureCriterionRandom(FractureCriterionRandom const &);
  FractureCriterionRandom & operator=(FractureCriterionRandom const &);

private:

  EntityRank space_dimension_;
  double probability_;
};

} // namespace LCM

#endif
