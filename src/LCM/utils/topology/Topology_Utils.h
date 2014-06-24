//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Topology_Utils_h)
#define LCM_Topology_Utils_h

#include "Topology_Types.h"

namespace LCM {

///
/// \brief Output the mesh connectivity
///
/// Outputs the nodal connectivity of the elements as stored by
/// bulkData. Assumes that relationships between the elements and
/// nodes exist.
///
inline
void
display_connectivity(BulkData * bulk_data, EntityRank cell_rank)
{
  // Create a list of element entities
  EntityVector
  elements;

  stk_classic::mesh::get_entities(*(bulk_data), cell_rank, elements);

  typedef EntityVector::size_type size_type;

  // Loop over the elements
  size_type const
  number_of_elements = elements.size();

  for (size_type i = 0; i < number_of_elements; ++i) {

    PairIterRelation
    relations = elements[i]->relations(NODE_RANK);

    EntityId const
    element_id = elements[i]->identifier();

    std::cout << std::setw(16) << element_id << ":";

    size_t const
    nodes_per_element = relations.size();

    for (size_t j = 0; j < nodes_per_element; ++j) {

      Entity const &
      node = *(relations[j].entity());

      EntityId const
      node_id = node.identifier();

      std::cout << std::setw(16) << node_id;
    }

    std::cout << '\n';
  }

  return;
}

///
/// \brief Output relations associated with entity
///        The entity may be of any rank
///
/// \param[in] entity
///
inline
void
display_relation(Entity const & entity)
{
  std::cout << "Relations for entity (identifier,rank): ";
  std::cout << entity.identifier() << "," << entity.entity_rank();
  std::cout << '\n';

  PairIterRelation
  relations = entity.relations();

  for (size_t i = 0; i < relations.size(); ++i) {
    std::cout << "entity:\t";
    std::cout << relations[i].entity()->identifier() << ",";
    std::cout << relations[i].entity()->entity_rank();
    std::cout << "\tlocal id: ";
    std::cout << relations[i].identifier();
    std::cout << '\n';
  }
  return;
}

///
/// \brief Output relations of a given rank associated with entity
///
/// \param[in] entity
/// \param[in] the rank of the entity
///
inline
void
display_relation(Entity const & entity, EntityRank const rank)
{
  std::cout << "Relations of rank ";
  std::cout << rank;
  std::cout << " for entity (identifier,rank): ";
  std::cout << entity.identifier() << "," << entity.entity_rank();
  std::cout << '\n';

  PairIterRelation
  relations = entity.relations(rank);

  for (size_t i = 0; i < relations.size(); ++i) {
    std::cout << "entity:\t";
    std::cout << relations[i].entity()->identifier() << ",";
    std::cout << relations[i].entity()->entity_rank();
    std::cout << "\tlocal id: ";
    std::cout << relations[i].identifier();
    std::cout << '\n';
  }
  return;
}

inline
bool
is_one_down(Entity const & entity, Relation const & relation)
{
  EntityRank const
  entity_rank = entity.entity_rank();

  EntityRank const
  target_rank = relation.entity_rank();

  return entity_rank - target_rank == 1;
}

inline
bool
is_one_up(Entity const & entity, Relation const & relation)
{
  EntityRank const
  entity_rank = entity.entity_rank();

  EntityRank const
  target_rank = relation.entity_rank();

  return target_rank - entity_rank == 1;
}

///
/// Test whether a given source entity and relation are
/// valid in the sense of the graph representation.
/// Multilevel relations are not valid.
///
inline
bool
is_graph_relation(Entity const & source_entity, Relation const & relation)
{
  return is_one_down(source_entity, relation);
}

///
/// Test whether a given source entity and relation are
/// needed in STK to maintain connectivity information.
/// These are relations that connect cells to points.
///
inline
bool
is_needed_for_stk(
    Entity const & source_entity,
    Relation const & relation,
    EntityRank const cell_rank)
{
  EntityRank const
  source_rank = source_entity.entity_rank();

  EntityRank const
  target_rank = relation.entity_rank();

  return (source_rank == cell_rank) && (target_rank == NODE_RANK);
}

// TODO: returning PairIterRelation(*relation_vector) below
// stores tenporary iterators to relation_vector that are
// invalid outside the scope of these functions.
// Perhaps change to returning the vector itself but this will require
// change of interface for functions that return relations.

///
/// Iterators to all relations.
///
inline
PairIterRelation
relations_all(Entity const & entity)
{
  return entity.relations();
}

///
/// Iterators to relations one level up.
///
inline
PairIterRelation
relations_one_up(Entity const & entity)
{
  return entity.relations(entity.entity_rank() + 1);
}

///
/// Iterators to relations one level down.
///
inline
PairIterRelation
relations_one_down(Entity const & entity)
{
  return entity.relations(entity.entity_rank() - 1);
}

///
/// Add a dash and processor rank to a string. Useful for output
/// file names.
///
inline
std::string
parallelize_string(std::string const & string)
{
  std::ostringstream
  oss;

  oss << string;

  int const
  number_processors = Teuchos::GlobalMPISession::getNProc();

  if (number_processors > 1) {

    int const
    number_digits = static_cast<int>(std::log10(number_processors));

    int const
    processor_id = Teuchos::GlobalMPISession::getRank();

    oss << "-";
    oss << std::setfill('0') << std::setw(number_digits) << processor_id;
  }

  return oss.str();
}

}// namespace LCM

#endif // LCM_Topology_Utils_h
