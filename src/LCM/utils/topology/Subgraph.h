//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Topology_Subgraph_h)
#define LCM_Topology_Subgraph_h

#include <stk_mesh/base/FieldBase.hpp>

#include "Topology_Types.h"

namespace LCM {

// Forward declaration
class Topology;

class Subgraph: public Graph {
public:

  ///
  /// \brief Create a subgraph given two vectors: a vertex list and
  ///        a edge list.
  ///
  /// \param[in] bulkData for the stk mesh object
  /// \param[in] start of the vertex list
  /// \param[in] end of the vertex list
  /// \param[in] start of the edge list
  /// \param[in] end of the edge list
  /// \param[in] number of dimensions in the analysis
  ///
  /// Subgraph stored as a boost adjacency list.  Maps are created
  /// to associate the subgraph to the global stk mesh graph.  Any
  /// changes to the subgraph are automatically mirrored in the stk
  /// mesh.
  ///
  Subgraph(
      Topology & topology,
      std::set<stk::mesh::Entity>::iterator first_entity,
      std::set<stk::mesh::Entity>::iterator last_entity,
      std::set<STKEdge>::iterator first_edge,
      std::set<STKEdge>::iterator last_edge);

  ///
  ///\brief Map a vertex in the subgraph to a entity in the stk mesh.
  ///
  ///\param[in] Vertex in the subgraph
  ///\return Global entity for the stk mesh
  ///
  ///Return the global entity (in the stk mesh) given a local
  ///subgraph vertex (in the boost subgraph).
  ///
  stk::mesh::Entity
  entityFromVertex(Vertex vertex);

  ///
  ///\brief Map a entity in the stk mesh to a vertex in the subgraph.
  ///
  ///\param[in] Global entity for the stk mesh
  ///\return Vertex in the subgraph
  ///
  ///Return local vertex (in the boost graph) given global entity (in the
  ///  stk mesh).
  ///
  Vertex
  vertexFromEntity(stk::mesh::Entity entity);

  ///
  ///\brief Add a vertex in the subgraph.
  ///
  ///\param[in] Rank of vertex to be added
  ///\param[in] For articulation points, pass the entity to determine
  /// the parts to which it belongs for propagation to new vertices.
  ///\return New vertex
  ///
  ///  Mirrors the change in the subgraph by adding a corresponding entity
  ///  to the stk mesh. Adds the relationship between the vertex and entity
  ///  to the maps localGlobalVertexMap and globalLocalVertexMap.
  ///
  Vertex
  addVertex(
      stk::mesh::EntityRank vertex_rank,
      stk::mesh::Entity entity = INVALID_ENTITY);

  ///
  /// \brief Remove vertex in subgraph
  ///
  /// \param[in] Vertex to be removed
  ///
  /// When the vertex is removed from the subgraph the corresponding
  /// entity from the stk mesh is also removed.
  ///
  /// Both boost and stk require that all edges to and from the
  /// vertex/entity are removed before deletion. If any edges
  /// remain, will be removed before the vertex/entity deletion.
  ///
  void
  removeVertex(Vertex const vertex);

  ///
  /// \brief Add edge to local graph.
  ///
  /// \param[in] Local ID of the target vertex with respect to the srouce vertex
  /// \param[in] Source vertex in the subgraph
  /// \param[in] Target vertex in the subgraph
  /// \return New edge and boolean value. If true, edge was inserted, if false
  ///  not inserted
  ///
  /// The edge insertion is mirrored in stk mesh. The edge is only
  /// inserted into the stk mesh object if it was inserted into the
  /// subgraph.
  ///
  std::pair<Edge, bool>
  addEdge(
      EdgeId const edge_id,
      Vertex const source_vertex,
      Vertex const target_vertex);

  ///
  /// \brief Remove edge from graph
  ///
  /// \param[in] Source vertex in subgraph
  /// \param[in] Target vertex in subgraph
  ///
  /// Edge removal is mirrored in the stk mesh.
  ///
  ///
  void
  removeEdge(
      Vertex const source_vertex,
      Vertex const target_vertex);

  ///
  /// \param[in] Vertex in subgraph
  ///
  /// \return Rank of vertex
  ///
  stk::mesh::EntityRank
  getVertexRank(Vertex const vertex);

  ///
  /// \param[in] Edge in subgraph
  /// \return Local numbering of edge target with respect to edge source
  ///
  /// In stk mesh, all relationships between entities have a local Id
  /// representing the correct ordering. Need this information to create
  /// or delete relations in the stk mesh.
  ///
  EdgeId
  getEdgeId(Edge const edge);

  ///
  /// \brief Function determines whether the input vertex is an
  ///        articulation point of the subgraph.
  ///
  /// \param[in] Input vertex
  /// \param[out] Number of components
  /// \param[out] map of vertex and associated component number
  ///
  /// Function checks vertex by the boost connected components algorithm to a
  /// copy of the subgraph. The copy does not include the input vertex.
  /// Copy is an undirected graph as required by the connected components
  /// algorithm.
  ///
  /// Returns the number of connected components as well as a map of the
  /// vertex in the subgraph and the component number.
  ///
  void
  testArticulationPoint(
      Vertex const articulation_vertex,
      size_t & number_components,
      VertexComponentMap & vertex_component_map);

  ///
  /// \brief Clones a boundary entity from the subgraph and separates
  /// the in-edges of the entity.
  ///
  /// \param Boundary vertex
  /// \return New boundary vertex
  ///
  /// Boundary entities are on boundary of the elements in the mesh. They
  /// will thus have either 1 or 2 in-edges to elements.
  ///
  /// If there is only 1 in-edge, the entity may be on the exterior of the
  /// mesh and is not a candidate for fracture for this subgraph. The
  /// boundary entity may be a valid candidate in another step. If only 1
  /// in edge: Return.
  ///
  Vertex
  cloneBoundaryVertex(Vertex vertex);

  ///
  /// Restore element to node connectivity needed by STK.
  /// The map contains a list of elements for which the point
  /// was replaced by a new point.
  ///
  void
  updateEntityPointConnectivity(
      stk::mesh::Entity old_point,
      EntityEntityMap & entity_new_point_map);

  ///
  /// \brief Splits an articulation point.
  ///
  /// \param[in] Input vertex
  /// \return Map of element and new node
  ///
  /// An articulation point is defined as a vertex which if removed
  /// yields a graph with more than 1 connected components. Creates
  /// an undirected graph and checks connected components of graph
  /// without vertex. Check if vertex is articulation point.
  ///
  /// Clones articulation point and splits in-edges between original
  /// and new vertices. The out-edges of the vertex are not in the
  /// subgraph. For a consistent global graph, add the out-edges of
  /// the vertex to the new vertex/vertices.
  ///
  /// If the vertex is a node, create a map between the element and
  /// the new node. If the nodal connectivity of an element does not
  /// change, do not add to the map.
  ///
  EntityEntityMap
  splitArticulation(Vertex vertex);

  ///
  /// \brief Clone all out edges of a vertex to a new vertex.
  ///
  /// \param[in] Old vertex
  /// \param[in] New vertex
  ///
  /// The global graph must remain consistent when new vertices are added. In
  /// split_articulation_point and clone_boundary_entity, all out-edges of
  /// the original vertex may not be in the subgraph.
  ///
  /// If there are missing edges in the subgraph, clone them from the original
  /// vertex to the new vertex. Edges not originally in the subgraph are added
  /// to the global graph only.
  ///
  void
  cloneOutEdges(Vertex old_vertex, Vertex new_vertex);

  ///
  /// \brief Output the graph associated with the mesh to graphviz .dot
  /// file for visualization purposes.
  ///
  /// \param[in] output file
  ///
  /// Similar to outputToGraphviz function in Topology class.
  /// If fracture criterion for entity is satisfied, the entity and all
  /// associated lower order entities are marked open. All open entities are
  /// displayed as such in output file.
  ///
  /// To create final output figure, run command below from terminal:
  ///   dot -Tpng <gviz_output>.dot -o <gviz_output>.png
  ///
  void
  outputToGraphviz(std::string const & output_filename);

  ///
  /// Accessors and mutators
  ///
  Topology &
  get_topology();

  size_t
  get_space_dimension();

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> &
  get_stk_mesh_struct();

  stk::mesh::BulkData &
  get_bulk_data();

  stk::mesh::MetaData &
  get_meta_data();

  stk::mesh::EntityId
  get_entity_id(stk::mesh::Entity const entity);

  stk::mesh::EntityRank
  get_boundary_rank();

  IntScalarFieldType &
  get_fracture_state_field(stk::mesh::EntityRank rank);

  void
  set_fracture_state(stk::mesh::Entity e, FractureState const fs);

  FractureState
  get_fracture_state(stk::mesh::Entity e);

  bool
  is_open(stk::mesh::Entity e);

  bool
  is_internal_and_open(stk::mesh::Entity e);

  bool
  is_internal(stk::mesh::Entity e)
  {

    assert(get_bulk_data().entity_rank(e) == get_boundary_rank());

    Vertex
    vertex = vertexFromEntity(e);

    boost::graph_traits<Graph>::degree_size_type
    number_in_edges = boost::in_degree(vertex, *this);

    assert(number_in_edges == 1 || number_in_edges == 2);

    return number_in_edges == 2;
  }

  ///
  /// Auxiliary types
  ///
  typedef std::map<Vertex, stk::mesh::Entity> VertexEntityMap;
  typedef std::map<stk::mesh::Entity, Vertex> EntityVertexMap;

private:

  //! Private to prohibit copying
  Subgraph(const Subgraph&);

  //! Private to prohibit copying
  Subgraph& operator=(const Subgraph&);

private:

  ///
  /// topology
  ///
  Topology &
  topology_;

  ///
  /// map local vertex -> global entity
  ///
  VertexEntityMap
  vertex_entity_map_;

  ///
  /// map global entity -> local vertex
  ///
  EntityVertexMap
  entity_vertex_map_;
};
// class Subgraph

}// namespace LCM

#endif // LCM_Topology_Subgraph_h
