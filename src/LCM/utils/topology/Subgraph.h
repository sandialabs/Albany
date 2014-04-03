//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Topology_Subgraph_h)
#define LCM_Topology_Subgraph_h

#include <stk_mesh/base/FieldData.hpp>

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
  Subgraph(RCP<Albany::AbstractSTKMeshStruct> stk_mesh_struct,
      std::set<EntityKey>::iterator first_vertex,
      std::set<EntityKey>::iterator last_vertex,
      std::set<stkEdge>::iterator first_edge,
      std::set<stkEdge>::iterator last_edge);

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
      Vertex const local_source_vertex,
      Vertex const local_target_vertex);

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
      Vertex const & local_source_vertex,
      Vertex const & local_target_vertex);

  ///
  /// \param[in] Vertex in subgraph
  ///
  /// \return Rank of vertex
  ///
  EntityRank
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
      Vertex const input_vertex,
      size_t & number_components,
      ComponentMap & component_map);

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
  /// Entity must have satisfied the fracture criterion and be labeled open
  /// in map is_open. If not open: Return.
  ///
  Vertex
  cloneBoundaryEntity(Vertex vertex);

  ///
  /// Restore element to node connectivity needed by STK.
  /// The map contains a list of elements for which the point
  /// was replaced by a new point.
  ///
  void
  updateElementNodeConnectivity(Entity & point, ElementNodeMap & map);

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
  std::map<Entity*, Entity*>
  splitArticulationPoint(Vertex vertex);

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
  size_t const
  getSpaceDimension() {return static_cast<size_t>(getSTKMeshStruct()->numDim);}

  RCP<Albany::AbstractSTKMeshStruct> &
  getSTKMeshStruct()
  {return stk_mesh_struct_;}

  BulkData *
  getBulkData()
  {return stk_mesh_struct_->bulkData;}

  stk::mesh::fem::FEMMetaData *
  getMetaData()
  {return stk_mesh_struct_->metaData;}

  EntityRank const
  getCellRank() {return getMetaData()->element_rank();}

  EntityRank const
  getBoundaryRank()
  {
    assert(getCellRank() > 0);
    return getCellRank() - 1;
  }

  IntScalarFieldType &
  getFractureState()
  {return *(stk_mesh_struct_->getFieldContainer()->getFractureState());}

  //
  // Set fracture state. Do nothing for cells (elements).
  //
  void
  setFractureState(Entity const & e, FractureState const fs)
  {
    if (e.entity_rank() < getCellRank()) {
      *(stk::mesh::field_data(getFractureState(), e)) = static_cast<int>(fs);
    }
  }

  //
  // Get fracture state. Return CLOSED for cells (elements).
  //
  FractureState
  getFractureState(Entity const & e)
  {
    return e.entity_rank() >= getCellRank() ?
    CLOSED :
    static_cast<FractureState>(*(stk::mesh::field_data(getFractureState(), e)));
  }

  bool
  isInternal(Entity const & e) {

    assert(e.entity_rank() == getBoundaryRank());

    Vertex
    vertex = globalToLocal(e.key());

    boost::graph_traits<Graph>::degree_size_type
    number_in_edges = boost::in_degree(vertex, *this);

    assert(number_in_edges == 1 || number_in_edges == 2);

    return number_in_edges == 2;
  }

  bool
  isOpen(Entity const & e) {
    return getFractureState(e) == OPEN;
  }

  bool
  isInternalAndOpen(Entity const & e) {
    return isInternal(e) == true && isOpen(e) == true;
  }

private:

  //! Private to prohibit copying
  Subgraph(const Subgraph&);

  //! Private to prohibit copying
  Subgraph& operator=(const Subgraph&);

  ///
  /// stk mesh data
  ///
  RCP<Albany::AbstractSTKMeshStruct> stk_mesh_struct_;

  ///
  /// map local vertex -> global entity key
  ///
  std::map<Vertex, EntityKey> local_global_vertex_map_;

  ///
  /// map global entity key -> local vertex
  ///
  std::map<EntityKey, Vertex> global_local_vertex_map_;
};
// class Subgraph

}// namespace LCM

#endif // LCM_Topology_Subgraph_h
