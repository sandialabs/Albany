//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Topology_Subgraph_h)
#define LCM_Topology_Subgraph_h

#include "Topology_Types.h"

namespace LCM {

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
  Subgraph(BulkData * bulk_data,
      std::set<EntityKey>::iterator first_vertex,
      std::set<EntityKey>::iterator last_vertex,
      std::set<stkEdge>::iterator first_edge,
      std::set<stkEdge>::iterator last_edge,
      int dimension);

  ///
  ///\brief Map a vertex in the subgraph to a entity in the stk mesh.
  ///
  ///\param[in] Vertex in the subgraph
  ///\return Global entity key for the stk mesh
  ///
  ///Return the global entity key (in the stk mesh) given a local
  ///subgraph vertex (in the boost subgraph).
  ///
  EntityKey
  localToGlobal(Vertex local_vertex);

  ///
  ///\brief Map a entity in the stk mesh to a vertex in the subgraph.
  ///
  ///\param[in] Global entity key for the stk mesh
  ///\return Vertex in the subgraph
  ///
  ///Return local vertex (in the boost graph) given global entity key (in the
  ///  stk mesh).
  ///
  Vertex
  globalToLocal(EntityKey global_vertex_key);

  ///
  ///\brief Add a vertex in the subgraph.
  ///
  ///\param[in] Rank of vertex to be added
  ///\return New vertex
  ///
  ///  Mirrors the change in the subgraph by adding a corresponding entity
  ///  to the stk mesh. Adds the relationship between the vertex and entity
  ///  to the maps localGlobalVertexMap and globalLocalVertexMap.
  ///
  Vertex
  addVertex(EntityRank vertex_rank);

  Vertex
  cloneVertex(Vertex & old_vertex);

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
  removeVertex(Vertex vertex);

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
  addEdge(const EdgeId edge_id, const Vertex local_source_vertex,
      const Vertex local_target_vertex);

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
  removeEdge(const Vertex & local_source_vertex,
      const Vertex & local_target_vertex);

  ///
  /// \param[in] Vertex in subgraph
  ///
  /// \return Rank of vertex
  ///
  EntityRank &
  getVertexRank(const Vertex vertex);

  ///
  /// \param[in] Edge in subgraph
  /// \return Local numbering of edge target with respect to edge source
  ///
  /// In stk mesh, all relationships between entities have a local Id
  /// representing the correct ordering. Need this information to create
  /// or delete relations in the stk mesh.
  ///
  EdgeId &
  getEdgeId(const Edge edge);

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
  testArticulationPoint(Vertex input_vertex, int & numComponents,
      std::map<Vertex, int> & subComponent);

  ///
  /// \brief Clones a boundary entity from the subgraph and separates the in-edges
  /// of the entity.
  ///
  /// \param[in] Boundary vertex
  /// \param[out] New boundary vertex
  /// \param Map of entity and boolean value is open
  ///
  /// Boundary entities are on boundary of the elements in the mesh. They
  /// will thus have either 1 or 2 in-edges to elements.
  ///
  /// If there is only 1 in-edge, the entity may be on the exterior of the
  /// mesh and is not a candidate for fracture for this subgraph. The
  /// boundary entity may be a valid candidate in another step. If only 1
  /// in edge: Return.
  ///
  /// Entity must have satisfied the fracture criterion and be labeled open
  /// in map is_open. If not open: Return.
  ///
  void
  cloneBoundaryEntity(Vertex & vertex, Vertex & newVertex,
      std::map<EntityKey, bool> & entity_open);

  ///
  /// \brief Splits an articulation point.
  ///
  /// \param[in] Input vertex
  /// \param Map of entity and boolean value is open
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
  std::map<Entity*, Entity*>
  splitArticulationPoint(Vertex vertex,
      std::map<EntityKey, bool> & entity_open);

  ///
  /// \brief Clone all out edges of a vertex to a new vertex.
  ///
  /// \param[in] Original vertex
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
  cloneOutEdges(Vertex & originalVertex, Vertex & newVertex);

  ///
  /// \brief Output the graph associated with the mesh to graphviz
  ///        .dot file for visualization purposes.
  ///
  /// \param[in] output file
  /// \param[in] map of entity and boolean value is open
  ///
  /// Similar to output_to_graphviz function in Topology class.
  /// If fracture criterion for entity is satisfied, the entity and all
  /// associated lower order entities are marked open. All open entities are
  /// displayed as such in output file.
  ///
  /// To create final output figure, run command below from terminal:
  ///   dot -Tpng <gviz_output>.dot -o <gviz_output>.png
  ///
  void
  outputToGraphviz(std::string & gviz_output,
      std::map<EntityKey, bool> entity_open);

private:

  //! Private to prohibit copying
  Subgraph(const Subgraph&);

  //! Private to prohibit copying
  Subgraph& operator=(const Subgraph&);

  ///
  /// Number of dimensions
  ///
  int dimension_;

  ///
  /// stk mesh data
  ///
  BulkData* bulk_data_;

  ///
  /// map local vertex -> global entity key
  ///
  std::map<Vertex, EntityKey> local_global_vertex_map_;

  ///
  /// map global entity key -> local vertex
  ///
  std::map<EntityKey, Vertex> global_local_vertex_map_;

  void
  communicate_and_create_shared_entities(Entity   & node,
      EntityKey   new_node_key);

  void
  bcast_key(unsigned root, EntityKey&   node_key);


};
// class Subgraph

}// namespace LCM

#endif // LCM_Topology_Subgraph_h
