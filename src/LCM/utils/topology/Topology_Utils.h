//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Topology_Utils_h)
#define LCM_Topology_Utils_h

#include "Topology_Types.h"

//#define DEBUG_LCM_TOPOLOGY

namespace LCM {

///
/// \brief Output the mesh connectivity
///
/// Outputs the nodal connectivity of the elements as stored by
/// bulkData. Assumes that relationships between the elements and
/// nodes exist.
///
void
display_connectivity(
    Topology & topology,
    stk::mesh::EntityRank cell_rank);

///
/// \brief Output relations associated with entity
///        The entity may be of any rank
///
/// \param[in] entity
///
void
display_relation(
    Topology & topology,
    stk::mesh::Entity entity);

///
/// \brief Output relations of a given rank associated with entity
///
/// \param[in] entity
/// \param[in] the rank of the entity
///
void
display_relation(
    Topology & topology,
    stk::mesh::Entity entity,
    stk::mesh::EntityRank const rank);

///
/// Test whether a given source entity and relation are
/// needed in STK to maintain connectivity information.
/// These are relations that connect cells to points.
///
bool
is_needed_for_stk(
    stk::mesh::BulkData & bulk_data,
    stk::mesh::Entity source_entity,
    stk::mesh::EntityRank target_rank,
    stk::mesh::EntityRank const cell_rank);

///
/// Add a dash and processor rank to a string. Useful for output
/// file names.
///
std::string
parallelize_string(std::string const & string);

///
/// Auxiliary for graphviz output
///
std::string
entity_label(stk::mesh::EntityRank const rank);

///
/// Auxiliary for graphviz output
///
std::string
entity_string(
    Topology & topology,
    stk::mesh::Entity entity);

///
/// Auxiliary for graphviz output
///
std::string
entity_color(
    stk::mesh::EntityRank const rank,
    FractureState const fracture_state);

///
/// Auxiliary for graphviz output
///
std::string
dot_header();

///
/// Auxiliary for graphviz output
///
std::string
dot_footer();


///
/// Auxiliary for graphviz output
///
std::string
dot_entity(
    size_t const space_dimension,
    int const parallel_rank,
    stk::mesh::Entity const entity,
    stk::mesh::EntityId const id,
    stk::mesh::EntityRank const rank,
    FractureState const fracture_state);

///
/// Auxiliary for graphviz output
///
std::string
relation_color(unsigned int const relation_id);

///
/// Auxiliary for graphviz output
///
std::string
dot_relation(
    stk::mesh::EntityId const source_id,
    stk::mesh::EntityRank const source_rank,
    stk::mesh::EntityId const target_id,
    stk::mesh::EntityRank const target_rank,
    unsigned int const relation_local_id);

//
// The entity id has now some very high number.
// Change it to something reasonable for debugging purposes.
// See formula for creating high id in CreateFaces.cpp
//
stk::mesh::EntityId
new_id_from_old_id(
    size_t const dimension,
    int const parallel_rank,
    stk::mesh::EntityRank const rank,
    stk::mesh::EntityId const old_id,
    bool const is_low_from_high);
//
//
//
stk::mesh::EntityId
low_id_from_high_id(
    size_t const dimension,
    int const parallel_rank,
    stk::mesh::EntityRank const rank,
    stk::mesh::EntityId const high_id);
//
//
//
stk::mesh::EntityId
high_id_from_low_id(
    size_t const dimension,
    int const parallel_rank,
    stk::mesh::EntityRank const rank,
    stk::mesh::EntityId const low_id);

///
/// Determine surface element topology based on bulk element topology
///
inline
shards::CellTopology
interfaceCellTopogyFromBulkCellTopogy(
    shards::CellTopology const & bulk_cell_topology)
{
  std::string const &
  bulk_cell_topology_name = bulk_cell_topology.getName();

  CellTopologyData const *
  ctd = NULL;

  if (bulk_cell_topology_name == "Triangle_3") {

    ctd = shards::getCellTopologyData<shards::Quadrilateral<4>>();

  } else if (bulk_cell_topology_name == "Quadrilateral_4") {

    ctd = shards::getCellTopologyData<shards::Quadrilateral<4>>();

  } else if (bulk_cell_topology_name == "Tetrahedron_4") {

    ctd = shards::getCellTopologyData<shards::Wedge<6>>();

  } else if (bulk_cell_topology_name == "Hexahedron_8") {

    ctd = shards::getCellTopologyData<shards::Hexahedron<8>>();

  } else {

    TEUCHOS_TEST_FOR_EXCEPTION(
        false,
        std::logic_error,
        "LogicError: Interface cell topology not implemented for:" <<
        bulk_cell_topology_name << '\n');

  }

  shards::CellTopology
  interface_cell_topology = shards::CellTopology(ctd);

  return interface_cell_topology;
}

}  // namespace LCM

#endif // LCM_Topology_Utils_h
