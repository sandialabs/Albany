//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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

#include <stk_mesh/base/FieldBase.hpp>

#include "Teuchos_ScalarTraits.hpp"
#include "Topology.h"
#include "Topology_Types.h"
#include "Topology_Utils.h"

namespace LCM {

///
/// Useful to distinguish among different partitioning schemes.
///
namespace fracture {

  enum Criterion {
    UNKNOWN,
    ONE,
    RANDOM,
    TRACTION};

}

///
/// Base class for fracture criteria
///
class AbstractFractureCriterion {

public:

  AbstractFractureCriterion(Topology & topology) : topology_(topology)
  {
  }

  virtual
  bool
  check(stk::mesh::BulkData & mesh, stk::mesh::Entity interface) = 0;

  virtual
  ~AbstractFractureCriterion()
  {
  }

  Topology &
  get_topology()
  {
    return topology_;
  }

  std::string const &
  get_bulk_block_name()
  {
    return get_topology().get_bulk_block_name();
  }

  std::string const &
  get_interface_block_name()
  {
    return get_topology().get_interface_block_name();
  }

  Albany::STKDiscretization &
  get_stk_discretization()
  {
    return get_topology().get_stk_discretization();
  }

  Albany::AbstractSTKMeshStruct const &
  get_stk_mesh_struct()
  {
    return *(get_topology().get_stk_mesh_struct());
  }

  stk::mesh::BulkData const &
  get_bulk_data()
  {
    return get_topology().get_bulk_data();
  }

  stk::mesh::MetaData const &
  get_meta_data()
  {
    return get_topology().get_meta_data();
  }

  minitensor::Index
  get_space_dimension()
  {
    return get_topology().get_space_dimension();
  }

  stk::mesh::Part &
  get_bulk_part()
  {
    return get_topology().get_bulk_part();
  }

  stk::mesh::Part &
  get_interface_part()
  {
    return get_topology().get_interface_part();
  }

  shards::CellTopology
  get_cell_topology()
  {
    return get_topology().get_cell_topology();
  }

protected:

  Topology &
  topology_;

private:

  AbstractFractureCriterion();
  AbstractFractureCriterion(const AbstractFractureCriterion &);
  AbstractFractureCriterion &operator=(const AbstractFractureCriterion &);

};

///
/// Random fracture criterion given a probability of failure
///
class FractureCriterionRandom: public AbstractFractureCriterion {

public:

  FractureCriterionRandom(Topology & topology, double const probability) :
      AbstractFractureCriterion(topology), probability_(probability)
  {
  }

  bool
  check(stk::mesh::BulkData & bulk_data, stk::mesh::Entity interface)
  {
    stk::mesh::EntityRank const
    rank = bulk_data.entity_rank(interface);

    stk::mesh::EntityRank const
    rank_up = static_cast<stk::mesh::EntityRank>(rank + 1);

    size_t const
    num_connected = bulk_data.num_connectivity(interface, rank_up);

    assert(num_connected == 2);

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
class FractureCriterionOnce: public AbstractFractureCriterion {

public:

  FractureCriterionOnce(Topology & topology, double const probability) :
      AbstractFractureCriterion(topology),
      probability_(probability),
      open_(true)
  {
  }

  bool
  check(stk::mesh::BulkData & bulk_data, stk::mesh::Entity interface)
  {
    stk::mesh::EntityRank const
    rank = bulk_data.entity_rank(interface);

    stk::mesh::EntityRank const
    rank_up = static_cast<stk::mesh::EntityRank>(rank + 1);

    size_t const
    num_connected = bulk_data.num_connectivity(interface, rank_up);

    assert(num_connected == 2);

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
class FractureCriterionTraction: public AbstractFractureCriterion {

public:

  FractureCriterionTraction(
      Topology & topology,
      std::string const & stress_name,
      double const critical_traction,
      double const beta);

  bool
  check(stk::mesh::BulkData & bulk_data, stk::mesh::Entity interface);

private:

  FractureCriterionTraction();
  FractureCriterionTraction(FractureCriterionTraction const &);
  FractureCriterionTraction & operator=(FractureCriterionTraction const &);

  minitensor::Vector<double> const &
  getNormal(stk::mesh::EntityId const entity_id);

  void
  computeNormals();

private:

  TensorFieldType const * const
  stress_field_;

  double
  critical_traction_;

  double
  beta_;

  std::map<stk::mesh::EntityId, minitensor::Vector<double>>
  normals_;
};

} // namespace LCM

#endif // LCM_Topology_FractureCriterion_h
