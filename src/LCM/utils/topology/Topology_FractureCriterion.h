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

#include <stk_mesh/base/FieldData.hpp>

#include "Teuchos_ScalarTraits.hpp"
#include "Topology_Types.h"
#include "Topology_Utils.h"

namespace LCM{

///
/// Base class for fracture criteria
///
class AbstractFractureCriterion {

public:

  AbstractFractureCriterion() {}

  virtual
  bool
  check(Entity const & entity) = 0;

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
  check(Entity const & entity)
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
/// Fracture criterion that open only once (for debugging)
///
class FractureCriterionOnce : public AbstractFractureCriterion {

public:

  FractureCriterionOnce(double const probability) :
  AbstractFractureCriterion(),
  probability_(probability),
  open_(true) {}

  bool
  check(Entity const & entity)
  {
    EntityRank const
    rank = entity.entity_rank();

    stk::mesh::PairIterRelation const
    relations = entity.relations(rank + 1);

    assert(relations.size() == 2);

    double const
    random = 0.5 * Teuchos::ScalarTraits<double>::random() + 0.5;

    bool const
    is_open = random < probability_ && open_;

    if (is_open == true) open_ = false;

    return is_open;
  }

private:

  FractureCriterionOnce();
  FractureCriterionOnce(FractureCriterionOnce const &);
  FractureCriterionOnce & operator=(FractureCriterionOnce const &);

private:

  double
  probability_;

  bool
  open_;
};

///
/// Traction fracture criterion
///
class FractureCriterionTraction : public AbstractFractureCriterion {

public:

  FractureCriterionTraction(
      stk::mesh::fem::FEMMetaData * meta_data,
      double const critical_traction,
      double const beta) :
  AbstractFractureCriterion(),
  meta_data_(meta_data),
  critical_traction_(critical_traction),
  beta_(beta) {}

  bool
  check(Entity const & entity)
  {
    stk::mesh::PairIterRelation const
    relations_up = relations_one_up(entity);

    assert(relations_up.size() == 2);

    stk::mesh::PairIterRelation const
    relations_node = entity.relations(NODE_RANK);

    std::string const
    stress_name = "nodal_Stress_Cauchy";

    TensorFieldType &
    stress_field = *(meta_data_->get_field<TensorFieldType>(stress_name));

    for (size_t i = 0; i < relations_node.size(); ++i) {

      Entity &
      node = *(relations_node[i].entity());

      std::cout << *(stk::mesh::field_data(stress_field, node));
    }

    double const
    random = 0.5 * Teuchos::ScalarTraits<double>::random() + 0.5;

    return random < critical_traction_;
  }

private:

  FractureCriterionTraction();
  FractureCriterionTraction(FractureCriterionTraction const &);
  FractureCriterionTraction & operator=(FractureCriterionTraction const &);

private:

  stk::mesh::fem::FEMMetaData *
  meta_data_;

  double
  critical_traction_;

  double
  beta_;
};

} // namespace LCM

#endif // LCM_Topology_FractureCriterion_h
