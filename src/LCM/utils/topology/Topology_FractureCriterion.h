//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

///
/// Fracture criteria classes are required to have a method
/// called check that takes as argument an entity and returns a bool.
///

#if !defined(LCM_Topology_FractureCriterion_h)
#define LCM_Topology_FractureCriterion_h

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

  FractureCriterionRandom(double const probability) :
  AbstractFractureCriterion(),
  probability_(probability) {}

  bool
  check(Entity const & entity) const
  {
    EntityRank const
    rank = entity.entity_rank();

    stk::mesh::PairIterRelation const
    relations = entity.relations(rank + 1);

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

  double
  probability_;
};

///
/// Traction fracture criterion
///
class FractureCriterionTraction : public AbstractFractureCriterion {

public:

  FractureCriterionTraction(double const critical_traction) :
  AbstractFractureCriterion(),
  critical_traction_(critical_traction) {}

  bool
  check(Entity const & entity) const
  {
    EntityRank const
    rank = entity.entity_rank();

    stk::mesh::PairIterRelation const
    relations = entity.relations(rank + 1);

    assert(relations.size() == 2);

    double const
    random = 0.5 * Teuchos::ScalarTraits<double>::random() + 0.5;

    return random < critical_traction_;
  }

private:

  FractureCriterionTraction();
  FractureCriterionTraction(FractureCriterionTraction const &);
  FractureCriterionTraction & operator=(FractureCriterionTraction const &);

private:

  double
  critical_traction_;
};

} // namespace LCM

#endif // LCM_Topology_FractureCriterion_h
