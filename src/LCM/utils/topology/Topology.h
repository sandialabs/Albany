//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Topology_h)
#define LCM_Topology_h

#include <iterator>

#include <stk_mesh/base/FieldData.hpp>

#include "Topology_Types.h"
#include "Topology_FractureCriterion.h"

namespace LCM {

class Topology {
public:
  ///
  /// \brief Create mesh data structure
  ///
  /// \param[in] input_file is exodus II input file name
  /// \param[in] output_file is exodus II output file name
  ///
  /// Use if want to create new Albany mesh object
  ///
  Topology(std::string const & input_file, std::string const & output_file);

  ///
  /// \brief Create mesh data structure
  ///
  /// \param[in] Albany discretization object
  ///
  /// Use if already have an Albany mesh object
  ///
  Topology(RCP<Albany::AbstractDiscretization> & discretization);

  ///
  /// \brief Create mesh data structure
  ///
  /// \param[in] Albany discretization object
  /// \param[in] Fracture criterion object
  ///
  /// Use if already have an Albany mesh object, and want to
  /// fracture the mesh based on a criterion.
  ///
  Topology(RCP<Albany::AbstractDiscretization> & discretization,
      RCP<AbstractFractureCriterion>& fracture_criterion);

  ///
  /// \brief Iterates over the boundary entities of the mesh of (all entities
  /// of rank dimension-1) and checks fracture criterion.
  ///
  /// \param map of entity and boolean value is entity open
  ///
  /// If fracture_criterion is met, the entity and all lower order entities
  /// associated with it are marked as open.
  ///
  void
  setEntitiesOpen(std::map<EntityKey, bool>& open_entity_map);

  ///
  /// \brief Iterates over the boundary entities contained in the passed-in
  /// vector and opens each edge traversed.
  ///
  /// \param vector of faces to open, map of entity and boolean value is entity opened
  ///
  /// If entity is in the vector, the entity and all lower order entities
  /// associated with it are marked as open.
  ///
  void
  setEntitiesOpen(const EntityVector& fractured_faces,
      std::map<EntityKey, bool>& open_entity_map);

  ///
  /// \brief Output the graph associated with the mesh to graphviz .dot
  /// file for visualization purposes.
  ///
  /// \param[in] output file
  ///
  /// To create final output figure, run command below from terminal:
  ///   dot -Tpng <gviz_output>.dot -o <gviz_output>.png
  ///
  void
  outputToGraphviz(std::string const & output_filename);

  ///
  /// \brief Initializes the default stk mesh object needed by class.
  ///
  /// Creates the full mesh representation of the mesh. Default stk mesh
  /// object has only elements and nodes. Function will delete unneeded
  /// relations between as described in Topology::remove_extra_relations().
  ///
  /// \attention Function must be called before mesh modification begins.
  ///
  /// \attention Call function once. Creation of extra entities and relations
  /// is slow.
  ///
  void
  graphInitialization();

  ///
  /// \brief Removes unneeded relations from the mesh.
  ///
  /// stk::mesh::create_adjacent_entities creates full mesh representation of
  /// the mesh instead of the default of only the elements and nodes. All
  /// entities created by the function are connected through relationships.
  /// Graph algorithms require relationships to only exist between entities
  /// separated by one degree, e.g. elements and faces in a 3D graph.
  /// Function removes all other relationships.
  ///
  /// \note Valid for 2D and 3D meshes.
  ///
  void
  removeExtraRelations();

  ///
  /// \brief Creates temporary nodal connectivity for the elements
  ///        and removes the relationships between the elements and
  ///        nodes.
  ///
  /// \attention Must be called every time before mesh topology
  ///            changes begin.
  ///
  void
  removeNodeRelations();

  ///
  /// Our canonical graph representation has edges (relations) that
  /// connect vertices (entities) with a difference in dimension (rank)
  /// of exactly one.
  /// This method removes all relations that do not conform to the above,
  /// leaving intact those needed for STK (between cells and points).
  /// This is required for the graph fracture algorithm to work.
  ///
  void
  removeMultiLevelRelations();

  ///
  /// \brief Returns array of pointers to Entities for the element to
  ///        node relations
  ///
  std::vector<EntityVector>
  getElementToNodeConnectivity();

  ///
  /// \brief Returns array of pointers to Entities for the element to
  ///        node relations
  ///
  void
  removeElementToNodeConnectivity(std::vector<EntityVector>& v);

  ///
  /// \brief After mesh manipulations are complete, need to recreate
  ///        a stk mesh understood by Albany_STKDiscretization.
  ///
  /// Recreates the nodal connectivity using connectivity_temp_.
  ///
  /// \attention must be called before mesh modification has ended
  ///
  void
  restoreElementToNodeConnectivity();

  ///
  /// \brief After mesh manipulations are complete, need to recreate
  ///        a stk mesh understood by Albany_STKDiscretization.
  void
  restoreElementToNodeConnectivity(std::vector<EntityVector>& v);

  ///
  /// \brief Determine the nodes associated with a face.
  ///
  /// \param[in] Face entity
  ///
  /// \return vector of nodes for the face
  ///
  /// Return an ordered list of nodes which describe the input face. In 2D,
  /// the face of the element is a line segment. In 3D, the face is a surface.
  /// Generalized for all element types valid in stk_mesh. Valid in 2D and 3D.
  ///
  /// \attention Assumes all mesh elements are same type.
  ///
  EntityVector
  getFaceNodes(Entity * entity);

  EntityVector
  getBoundaryEntityNodes(Entity const & boundary_entity);

  ///
  /// \brief Output boundary
  ///
  void
  outputBoundary();

  ///
  /// \brief Create boundary mesh
  ///
  void
  createBoundary();

  ///
  /// \brief Create surface element connectivity
  ///
  /// \param[in] Face 1
  /// \param[in] Face 2
  /// \return Cohesive connectivity
  ///
  /// Given the two faces after insertion process, create the
  /// connectivity of the cohesive element.
  ///
  /// \attention Assumes that all elements have the same topology
  ////
  EntityVector
  createSurfaceElementConnectivity(Entity const & face1, Entity const & face2);

  ///
  /// \brief Struct to store the data needed for creation or
  ///        deletion of an edge in the stk mesh object.
  ///
  /// \param source entity key
  /// \param target entity key
  /// \param local id of the target entity with respect to the source
  ///
  /// To operate on an stk relation between entities (e.g. deleting
  /// a relation), need the source entity, target entity, and local
  /// ID of the relation with respect to the source entity.
  ///
  /// Used to create edges from the stk mesh object in a boost graph
  ///
  struct stkEdge {
    EntityKey source;
    EntityKey target;
    EdgeId local_id;
  };

  ///
  /// Check if edges are the same
  ///
  struct EdgeLessThan
  {
    bool operator()(const stkEdge & a, const stkEdge & b) const
    {
      if (a.source < b.source) return true;
      if (a.source > b.source) return false;
      // source a and b are the same - check target
      return (a.target < b.target);
    }
  };

  ///
  /// \brief Create vectors describing the vertices and edges of the
  ///        star of an entity in the stk mesh.
  ///
  ///  \param list of entities in the star
  ///  \param list of edges in the star
  ///  \param[in] source entity of the star
  ///
  ///   The star of a graph vertex is defined as the vertex and all
  ///   higher order vertices which are connected to it when
  ///   traversing up the graph from the input vertex.
  ///
  ///   \attention Valid for entities of all ranks
  ///
  void
  createStar(
      std::set<EntityKey> & subgraph_entity_list,
      std::set<stkEdge, EdgeLessThan> & subgraph_edge_list,
      Entity & entity);

  ///
  /// \brief Fractures all open boundary entities of the mesh.
  ///
  /// \param[in] map of entity and boolean value is entity open
  ///
  /// Iterate through the faces of the mesh and split into two faces
  /// if marked as open. The elements associated with an open face
  /// are separated. All lower order entities of the face are
  /// updated for a consistent mesh.
  ///
  /// \todo generalize the function for 2D meshes
  ///
  void
  splitOpenFaces(std::map<EntityKey, bool> & open_entity_map);

  void
  splitOpenFaces(
      std::map<EntityKey, bool> & open_entity_map,
      std::vector<EntityVector>& old_connectivity,
      std::vector<EntityVector>& new_connectivity);

  ///
  /// \brief Adds a new entity of rank 3 to the mesh
  ///
  void
  addElement(EntityRank entity_rank);

  ///
  /// \brief creates several entities at a time. The information
  ///        about the type of entity and the amount of entities is
  ///        contained in the input vector called: "requests"
  ///
  void
  addEntities(std::vector<size_t> & requests);

  ///
  /// \brief Removes an entity and all its connections
  ///
  void
  removeEntity(Entity & entity);

  ///
  /// \brief Adds a relation between two entities
  ///
  void
  addRelation(Entity & source_entity, Entity & target_entity,
      EdgeId local_relation_id);

  ///
  /// \brief Removes the relation between two entities
  ///
  void
  removeRelation(Entity & source_entity, Entity & target_entity,
      EdgeId local_relation_id);

  ///
  /// \brief Returns a vector with all the mesh entities of a
  ///        specific rank
  ///
  EntityVector
  getEntitiesByRank(const stk::mesh::BulkData & mesh,
      EntityRank entity_rank);

  ///
  /// \brief Number of entities of a specific rank
  ///
  EntityVector::size_type
  getNumberEntitiesByRank(const stk::mesh::BulkData & mesh,
      EntityRank entity_rank);

  ///
  /// \brief Gets the local relation id (0,1,2,...) between two entities
  ///
  EdgeId
  getLocalRelationId(const Entity & source_entity,
      const Entity & target_entity);

  ///
  /// \brief Returns the total number of lower rank entities
  ///        connected to a specific entity
  ///
  int
  getNumberLowerRankEntities(const Entity & entity);

  ///
  /// \brief Returns a group of entities connected directly to a
  ///        given entity. The input rank is the rank of the
  ///        returned entities.
  ///
  EntityVector
  getDirectlyConnectedEntities(const Entity & entity,
      EntityRank entity_rank);

  ///
  /// \brief Checks if an entity exists inside a specific vector
  ///
  bool
  findEntityInVector(EntityVector & entities,
      Entity * entity);

  ///
  /// \brief Returns a group of entities connected indirectly to a
  ///        given entity.
  ///
  /// e.g. of returns: nodes that belong to a face segments or nodes
  /// that belong to an element The input rank is the rank of the
  /// returned entities.  The input rank must be lower than that of
  /// the input entity
  ///
  ///
  EntityVector
  getBoundaryEntities(const Entity & entity, EntityRank entity_rank);

  ///
  /// \brief Checks if a segment is connected to an input node. Returns "true" if the segment connects to the node.
  ///
  bool
  segmentIsConnected(const Entity & segment, Entity * node);

  ///
  /// \brief Finds the adjacent segments to a given segment. The
  ///        adjacent segments are connected to a given common
  ///        point. it returns adjacent segments
  ///
  EntityVector
  findAdjacentSegments(const Entity & segment, Entity * node);

  ///
  /// \brief Returns all the highest dimensional topology entities
  ///        to which a given face belongs
  ///
  EntityVector
  findCellRelations(const Entity & face);

  ///
  /// \brief Returns all the segments at the boundary of a given
  ///        element. Including those connected between the faces
  ///        barycenters and the faces boundary nodes
  ///
  EntityVector
  findSegmentsFromElement(const Entity & element);

  ///
  /// \brief Returns true if the faces share a segment (two points)
  ///
  bool
  facesShareTwoPoints(const Entity & face1, const Entity & face2);

  ///
  /// \brief returns the adjacent segments from a given face
  ///
  EntityVector
  findAdjacentSegmentsFromFace(const std::vector<EntityVector> & faces_inside_element,
      const Entity & _face,
      int element_number);

  ///
  /// \brief Returns a pointer with the coordinates of a given entity
  ///
  double*
  getPointerOfCoordinates(Entity * entity);

  ///
  /// \brief Returns a vector with the corresponding former boundary
  ///        nodes of an input entity
  ///
  EntityVector
  getFormerElementNodes(const Entity & element,
      const std::vector<EntityVector> & entities);

  ///
  /// \brief Generates the coordinate of a given barycenter
  ///        "entities" is a vector with points that belong to the same
  ///        entity of the barycenter(e.g segment, face, or element)
  ///
  void
  computeBarycentricCoordinates(const EntityVector & entities, Entity * barycenter);

  ///
  /// \brief Barycentric subdivision
  ///
  void
  barycentricSubdivision();

  ///
  /// \brief Finds the closest nodes(Entities of rank 0) to each of the three points in the input vector.
  /// EntityVector
  std::vector<Entity*>
  getClosestNodes(std::vector<std::vector<double> > points);

  ///
  /// \brief Finds the closest nodes(Entities of rank 0) to each
  ///        of the three points in the input vectorThese nodes
  ///        lie over the surface of the mesh
  ///
  std::vector<Entity*>
  getClosestNodesOnSurface(std::vector<std::vector<double> > points);

  ///
  /// \brief calculates the distance between a node and a point
  ///
  double
  getDistanceNodeAndPoint(Entity* node, std::vector<double> point);

  ///
  /// \brief Returns the coordinates of the points that form a equilateral triangle.
  ///        This triangle lies on the plane that intersects the ellipsoid.
  ///
  std::vector<std::vector<double> >
  getCoordinatesOfTriangle(const std::vector<double> normalToPlane);

  ///
  /// \brief Return a random number between two given numbers
  ///
  double
  randomNumber(double valMin, double valMax);

  ///
  /// \brief Returns the distance between two entities of rank 0 (nodes)
  ///
  double
  getDistanceBetweenNodes(Entity * node1, Entity * node2);

  ///
  /// \brief Returns the coordinates of the max and min of x y and z
  ///        in the order max of, min of x, max of y, min of y, max of z, min of z
  ///
  std::vector<double>
  getCoordinatesOfMaxAndMin();

  ///
  /// \brief Returns the edges necessary to compute the shortest path on the outer surface
  ///        of the mesh
  ///
  std::vector<Entity*>
  MeshEdgesShortestPath();

  ///
  /// \brief Returns the shortest path over the boundary faces given three input nodes
  ///        and the edges that belong to the outer surface
  ///
  std::vector<std::vector<int> >
  shortestpathOnBoundaryFaces(const std::vector<Entity*> & nodes,
		  const std::vector<Entity*> & MeshEdgesShortestPath);

  ///
  /// \brief Returns the shortest path between three input nodes
  ///
  std::vector<std::vector<int> >
  shortestpath(const std::vector<Entity*> & nodes);

  ///
  /// \brief Returns the directions of all the edges of the input mesh
  ///
  std::vector<std::vector<int> >
  edgesDirections();

  ///
  /// \brief Returns the directions of all the boundary edges of the input mesh
  ///
  std::vector<std::vector<int> >
  edgesDirectionsOuterSurface();


  ///
  /// \brief Returns the directions of all of the faces of the input mesh
  ///
  std::vector<std::vector<int> >
  facesDirections();

  ///
  /// \brief Returns a vector with the areas of each of the faces of the input mesh
  ///
  std::vector<double>
  facesAreas();

  ///
  /// \brief Returns the boundary operator of the input mesh.
  ///        matrix that has nonzeros only
  ///
  std::vector<std::vector<int> >
  boundaryOperator();

  ///
  /// \brief returns the boundary operator along with the faces areas
  ///        to create the columns of an mps file
  ///
  std::vector<std::vector<double> >
  outputForMpsFile();

  ///
  /// \brief Returns the 1-D boundary required to compute the minimum surface of the
  ///        input mesh. The input to this function is a shortest path (composed by egdes)
  ///        between three nodes
  ///
  std::vector<std::vector<int> >
  boundaryVector(std::vector<std::vector<int> > & shortPath);

  ///
  /// \brief Returns the 1-D boundary required to compute the minimum surface of the input
  ///        mesh boundary faces. The input to this function is a shortest path
  ///        (composed by edges) between three nodes
  ///
  std::vector<std::vector<int> >
  boundaryVectorOuterSurface(std::vector<std::vector<int> > & shortPath);

  ///
  /// \brief Returns the corresponding entities of rank 2 that build the minimum surface.
  ///        It takes as an input the resulting vector taken from the solution of the
  ///        linear programming solver
  ///
  std::vector<Entity*>
  MinimumSurfaceFaces(std::vector<int> VectorFromLPSolver);

  ///
  /// \brief Returns the number of times an entity is repeated in a vector
  ///
  int
  NumberOfRepetitions(std::vector<Entity*> & entities, Entity * entity);

  ///
  /// \brief Returns the coordinates of an input node.
  ///        The input is the identifier of a node
  ///
  std::vector<double>
  findCoordinates(unsigned int nodeIdentifier);

  ///----------------------------------------------------------------------
  ///
  /// \brief Practice creating the barycentric subdivision
  ///
  void
  barycentricSubdivision_();

  ///
  /// \brief Divide former mesh segments by half
  ///
  void
  divideSegmentsHalf();

  void
  addcentroid();

  void
  connectcentroid();

  void
  addnewfaces();

  void
  connectnewfaces();

  ///
  /// Accessors and mutators
  ///
  void
  setSpaceDimension(int const sd) {space_dimension_ = sd;}

  int
  getSpaceDimension() const {return space_dimension_;}

  void
  setNodeRank(EntityRank const nr) {node_rank_ = nr;}

  EntityRank
  getNodeRank() const {return node_rank_;}

  void
  setEdgeRank(EntityRank const er) {edge_rank_ = er;}

  EntityRank
  getEdgeRank() const {return edge_rank_;}

  void
  setFaceRank(EntityRank const fr) {face_rank_ = fr;}

  EntityRank
  getFaceRank() const {return face_rank_;}

  void
  setCellRank(EntityRank const cr) {cell_rank_ = cr;}

  EntityRank
  getCellRank() const {return cell_rank_;}

  IntScalarFieldType &
  getFractureState()
  {return *(stk_mesh_struct_->getFieldContainer()->getFractureState());}

  void
  setFractureCriterion(RCP<AbstractFractureCriterion> const & fc)
  {fracture_criterion_ = fc;}

  RCP<AbstractFractureCriterion> &
  getFractureCriterion()
  {return fracture_criterion_;}

  void
  setSTKMeshStruct(RCP<Albany::AbstractSTKMeshStruct> const & sms)
  {stk_mesh_struct_ = sms;}

  RCP<Albany::AbstractSTKMeshStruct> &
  getSTKMeshStruct()
  {return stk_mesh_struct_;}

  void
  setDiscretization(RCP<Albany::AbstractDiscretization> const & d)
  {discretization_ = d;}

  RCP<Albany::AbstractDiscretization> &
  getDiscretization()
  {return discretization_;}

  stk::mesh::BulkData *
  getBulkData()
  {return stk_mesh_struct_->bulkData;}

  stk::mesh::fem::FEMMetaData *
  getMetaData()
  {return stk_mesh_struct_->metaData;}

  void
  setCellTopology(shards::CellTopology const & ct)
  {cell_topology_ = ct;}

  shards::CellTopology &
  getCellTopology()
  {return cell_topology_;}

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

  ///
  /// Initialization of the open field for fracture
  ///
  void
  initializeFractureState();




private:
  ///
  /// \brief Create Albany discretization
  ///
  /// Called by constructor
  ///
  void
  createDiscretization();

  ///
  /// \brief Assigns Ids to new nodes (not comptabile with STK)
  /// FIXME check this method
  void
  setHighestIds();

  ///
  /// Ranks of all entities of the mesh.
  ///
  EntityRank node_rank_;
  EntityRank edge_rank_;
  EntityRank face_rank_;
  EntityRank cell_rank_;

  int space_dimension_;

  //
  //
  RCP<Albany::AbstractDiscretization> discretization_;

  RCP<Albany::AbstractSTKMeshStruct> stk_mesh_struct_;

  std::vector<EntityVector> connectivity_temp_;

  std::map<int, int> element_global_to_local_ids_;

  std::set<std::pair<Entity*, Entity*> > fractured_faces_;

  std::vector<int> highest_ids_;

  /// \attention Topology of elements in mesh. Only valid if one
  ///            element type used.  Will not give correct results
  ///            if mesh has multiple element types.
  shards::CellTopology cell_topology_;

  /// Pointer to failure criterion object
  RCP<AbstractFractureCriterion> fracture_criterion_;

protected:
  ///
  /// \brief Hide default constructor for Topology
  ///
  Topology();
};
// class Topology

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
  Subgraph(stk::mesh::BulkData* bulk_data,
      std::set<EntityKey>::iterator first_vertex,
      std::set<EntityKey>::iterator last_vertex,
      std::set<Topology::stkEdge>::iterator first_edge,
      std::set<Topology::stkEdge>::iterator last_edge, int num_dim);

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
  removeVertex(Vertex & vertex);

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
  int num_dim_;

  ///
  /// stk mesh data
  ///
  stk::mesh::BulkData* bulk_data_;

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

///
/// \brief Output the mesh connectivity
///
/// Outputs the nodal connectivity of the elements as stored by
/// bulkData. Assumes that relationships between the elements and
/// nodes exist.
///
inline
void
display_connectivity(Topology & topology)
{
  // Create a list of element entities
  EntityVector
  elements;

  stk::mesh::get_entities(
      *(topology.getBulkData()),
      topology.getCellRank(),
      elements);

  typedef EntityVector::size_type size_type;

  // Loop over the elements
  size_type const
  number_of_elements = elements.size();

  for (size_type i = 0; i < number_of_elements; ++i) {

    PairIterRelation
    relations = elements[i]->relations(topology.getNodeRank());

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
is_one_down(Entity const & source_entity, Relation const & relation)
{
  EntityRank const
  source_rank = source_entity.entity_rank();

  EntityRank const
  target_rank = relation.entity_rank();

  return source_rank - target_rank == 1;
}

inline
bool
is_one_up(Entity const & source_entity, Relation const & relation)
{
  EntityRank const
  source_rank = source_entity.entity_rank();

  EntityRank const
  target_rank = relation.entity_rank();

  return target_rank - source_rank == 1;
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

  return (source_rank == cell_rank) && (target_rank == 0);
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

#endif
