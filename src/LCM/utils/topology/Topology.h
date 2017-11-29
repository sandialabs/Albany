//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(LCM_Topology_Topology_h)
#define LCM_Topology_Topology_h

#include <iterator>

#include <stk_mesh/base/FieldBase.hpp>

#include "Topology_Types.h"
#include "Topology_Utils.h"

namespace LCM {

// Forward declaration
class AbstractFractureCriterion;

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
  /// \param[in] Name of bulk block
  /// \param[in] Name of interface block
  ///
  /// Use if already have an Albany mesh object
  ///
  Topology(
      Teuchos::RCP<Albany::AbstractDiscretization> & abstract_disc,
      std::string const & bulk_block_name,
      std::string const & interface_block_name);

  ///
  /// \brief Iterates over the boundary entities of the mesh of (all
  /// entities of rank dimension-1) and checks fracture criterion.
  ///
  /// If fracture_criterion is met, the entity and all lower order
  /// entities associated with it are marked as open.
  ///
  size_t
  setEntitiesOpen();

  ///
  /// \brief Output the graph associated with the mesh to graphviz
  /// .dot file for visualization purposes.
  ///
  /// \param[in] output file
  ///
  /// To create final output figure, run command below from terminal:
  ///   dot -Tpng <gviz_output>.dot -o <gviz_output>.png
  ///
  enum OutputType
  {
    UNIDIRECTIONAL_UNILEVEL,
    UNIDIRECTIONAL_MULTILEVEL,
    BIDIRECTIONAL_UNILEVEL,
    BIDIRECTIONAL_MULTILEVEL
  };

  void
  outputToGraphviz(std::string const & output_filename);

  ///
  /// \brief Initializes the default stk mesh object needed by class.
  ///
  /// Creates the full mesh representation of the mesh. Default stk
  /// mesh object has only elements and nodes. Function will delete
  /// unneeded relations between as described in
  /// Topology::remove_extra_relations().
  ///
  /// \attention Function must be called before mesh modification begins.
  ///
  /// \attention Call function once. Creation of extra entities and relations
  /// is slow.
  ///
  void
  graphInitialization();

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
  /// connect vertices (entities) with a difference in dimension
  /// (rank) of exactly one.
  ///
  /// This method removes all relations that do not conform to the
  /// above, leaving intact those needed for STK (between cells and
  /// points).  This is required for the graph fracture algorithm to
  /// work.
  ///
  void
  removeMultiLevelRelations();

  ///
  /// \brief After mesh manipulations are complete, need to recreate a
  ///        stk mesh understood by Albany::STKDiscretization.
  ///
  /// Recreates the nodal connectivity using connectivity_.
  ///
  /// \attention must be called before mesh modification has ended
  ///
  void
  restoreElementToNodeConnectivity();

  ///
  /// \brief Determine the nodes associated with a boundary entity (face).
  ///
  /// \param[in] Boundary entity
  ///
  /// \return vector of nodes for the face
  ///
  /// Return an ordered list of nodes which describe the input
  /// face. In 2D, the face of the element is a line segment. In 3D,
  /// the face is a surface.  Generalized for all element types valid
  /// in stk_mesh. Valid in 2D and 3D.
  ///
  /// \attention Assumes all mesh elements are same type.
  ///
  stk::mesh::EntityVector
  getBoundaryEntityNodes(stk::mesh::Entity boundary_entity);

  std::vector<minitensor::Vector<double>>
  getNodalCoordinates();

  ///
  /// \brief Output boundary
  ///
  void
  outputBoundary(std::string const & output_filename);

  ///
  /// \brief Get a connectivity list of the boundary
  ///
  Connectivity
  getBoundary();

  ///
  /// \brief Create surface element connectivity
  ///
  /// \param[in] Face top
  /// \param[in] Face bottom
  /// \return Cohesive connectivity
  ///
  /// Given the two faces after insertion process, create the
  /// connectivity of the cohesive element.
  ///
  /// \attention Assumes that all elements have the same topology
  ////
  stk::mesh::EntityVector
  createSurfaceElementConnectivity(
      stk::mesh::Entity face_top,
      stk::mesh::Entity face_bottom);

  ///
  /// \brief Create vectors describing the vertices and edges of the
  ///        star of an entity in the stk mesh.
  ///
  ///  \param[in] source entity of the star
  ///  \param list of entities in the star
  ///  \param list of edges in the star
  ///
  ///   The star of a graph vertex is defined as the vertex and all
  ///   higher order vertices which are connected to it when
  ///   traversing up the graph from the input vertex.
  ///
  ///   \attention Valid for entities of all ranks
  ///
  void
  createStar(
      stk::mesh::Entity entity,
      std::set<stk::mesh::Entity> & subgraph_entities,
      std::set<STKEdge, EdgeLessThan> & subgraph_edges);

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
  splitOpenFaces();

  void
  insertSurfaceElements(std::set<EntityPair> const & fractured_faces);

  ///
  /// \brief Adds a new entity of rank 3 to the mesh
  ///
  void
  addElement(stk::mesh::EntityRank entity_rank);

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
  removeEntity(stk::mesh::Entity entity);

  ///
  /// \brief Adds a relation between two entities
  ///
  void
  addRelation(
      stk::mesh::Entity source_entity,
      stk::mesh::Entity target_entity,
      EdgeId local_relation_id);

  ///
  /// \brief Removes the relation between two entities
  ///
  void
  removeRelation(
      stk::mesh::Entity source_entity,
      stk::mesh::Entity target_entity,
      EdgeId local_relation_id);

  ///
  /// \brief Returns a vector with all the mesh entities of a
  ///        specific rank
  ///
  stk::mesh::EntityVector
  getEntitiesByRank(
      stk::mesh::BulkData const & bulk_data,
      stk::mesh::EntityRank entity_rank);

  ///
  /// \brief Gets the local relation id (0,1,2,...) between two entities
  ///
  EdgeId
  getLocalRelationId(
      stk::mesh::Entity source_entity,
      stk::mesh::Entity target_entity);

  ///
  /// \brief Returns the total number of lower rank entities
  ///        connected to a specific entity
  ///
  int
  getNumberLowerRankEntities(stk::mesh::Entity entity);

  ///
  /// \brief Returns a group of entities connected directly to a
  ///        given entity. The input rank is the rank of the
  ///        returned entities.
  ///
  stk::mesh::EntityVector
  getDirectlyConnectedEntities(
      stk::mesh::Entity entity,
      stk::mesh::EntityRank entity_rank);

  ///
  /// \brief Checks if an entity exists inside a specific vector
  ///
  bool
  findEntityInVector(
      stk::mesh::EntityVector & entities,
      stk::mesh::Entity entity);

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
  stk::mesh::EntityVector
  getBoundaryEntities(
      stk::mesh::Entity entity,
      stk::mesh::EntityRank entity_rank);

  ///
  /// \brief Checks if a segment is connected to an input node.
  /// Returns "true" if the segment connects to the node.
  ///
  bool
  segmentIsConnected(stk::mesh::Entity segment, stk::mesh::Entity node);

  ///
  /// \brief Finds the adjacent segments to a given segment. The
  ///        adjacent segments are connected to a given common
  ///        point. it returns adjacent segments
  ///
  stk::mesh::EntityVector
  findAdjacentSegments(stk::mesh::Entity segment, stk::mesh::Entity node);

  ///
  /// \brief Returns all the highest dimensional topology entities
  ///        to which a given face belongs
  ///
  stk::mesh::EntityVector
  findCellRelations(stk::mesh::Entity face);

  ///
  /// \brief Returns all the segments at the boundary of a given
  ///        element. Including those connected between the faces
  ///        barycenters and the faces boundary nodes
  ///
  stk::mesh::EntityVector
  findSegmentsFromElement(stk::mesh::Entity element);

  ///
  /// \brief Returns true if the faces share a segment (two points)
  ///
  bool
  facesShareTwoPoints(stk::mesh::Entity face1, stk::mesh::Entity face2);

  ///
  /// \brief returns the adjacent segments from a given face
  ///
  stk::mesh::EntityVector
  findAdjacentSegmentsFromFace(
      std::vector<stk::mesh::EntityVector> const & faces_inside_element,
      stk::mesh::Entity face,
      int element_number);

  ///
  /// \brief Returns a pointer with the coordinates of a given entity
  ///
  double*
  getEntityCoordinates(stk::mesh::Entity entity);

  ///
  /// \brief Returns a vector with the corresponding former boundary
  ///        nodes of an input entity
  ///
  stk::mesh::EntityVector
  getFormerElementNodes(
      stk::mesh::Entity element,
      std::vector<stk::mesh::EntityVector> const & entities);

  ///
  /// \brief Generates the coordinate of a given barycenter
  ///        "entities" is a vector with points that belong to the same
  ///        entity of the barycenter(e.g segment, face, or element)
  ///
  void
  computeBarycentricCoordinates(
      stk::mesh::EntityVector const & entities,
      stk::mesh::Entity barycenter);

  ///
  /// \brief Barycentric subdivision
  ///
  void
  barycentricSubdivision();

  ///
  /// \brief Finds the closest nodes(Entities of rank 0) to each of
  /// the three points in the input vector.  stk::mesh::EntityVector
  std::vector<stk::mesh::Entity>
  getClosestNodes(std::vector<std::vector<double>> points);

  ///
  /// \brief Finds the closest nodes(Entities of rank 0) to each
  ///        of the three points in the input vectorThese nodes
  ///        lie over the surface of the mesh
  ///
  std::vector<stk::mesh::Entity>
  getClosestNodesOnSurface(std::vector<std::vector<double>> points);

  ///
  /// \brief calculates the distance between a node and a point
  ///
  double
  getDistanceNodeAndPoint(stk::mesh::Entity node, std::vector<double> point);

  ///
  /// \brief Returns the coordinates of the points that form a
  ///        equilateral triangle.  This triangle lies on the plane
  ///        that intersects the ellipsoid.
  ///
  std::vector<std::vector<double>>
  getCoordinatesOfTriangle(std::vector<double> const normalToPlane);

  ///
  /// \brief Return a random number between two given numbers
  ///
  double
  randomNumber(double valMin, double valMax);

  ///
  /// \brief Returns the distance between two entities of rank 0 (nodes)
  ///
  double
  getDistanceBetweenNodes(stk::mesh::Entity node1, stk::mesh::Entity node2);

  ///
  /// \brief Returns the coordinates of the max and min of x y and z
  ///        in the order max of, min of x, max of y, min of y, max of
  ///        z, min of z
  ///
  std::vector<double>
  getCoordinatesOfMaxAndMin();

  ///
  /// \brief Returns the edges necessary to compute the shortest path
  ///        on the outer surface of the mesh
  ///
  std::vector<stk::mesh::Entity>
  meshEdgesShortestPath();

  ///
  /// \brief Returns the shortest path over the boundary faces given
  ///        three input nodes and the edges that belong to the outer
  ///        surface
  ///
  std::vector<std::vector<int>>
  shortestpathOnBoundaryFaces(
      std::vector<stk::mesh::Entity> const & nodes,
      std::vector<stk::mesh::Entity> const & MeshEdgesShortestPath);

  ///
  /// \brief Returns the shortest path between three input nodes
  ///
  std::vector<std::vector<int>>
  shortestpath(std::vector<stk::mesh::Entity> const & nodes);

  ///
  /// \brief Returns the directions of all the edges of the input mesh
  ///
  std::vector<std::vector<int>>
  edgesDirections();

  ///
  /// \brief Returns the directions of all the boundary edges of the
  /// input mesh
  ///
  std::vector<std::vector<int>>
  edgesDirectionsOuterSurface();

  ///
  /// \brief Returns the directions of all of the faces of the input
  /// mesh
  ///
  std::vector<std::vector<int>>
  facesDirections();

  ///
  /// \brief Returns a vector with the areas of each of the faces of
  /// the input mesh
  ///
  std::vector<double>
  facesAreas();

  ///
  /// \brief Returns the boundary operator of the input mesh.
  ///        matrix that has nonzeros only
  ///
  std::vector<std::vector<int>>
  boundaryOperator();

  ///
  /// \brief returns the boundary operator along with the faces areas
  ///        to create the columns of an mps file
  ///
  std::vector<std::vector<double>>
  outputForMpsFile();

  ///
  /// \brief Returns the 1-D boundary required to compute the minimum
  ///        surface of the input mesh. The input to this function is
  ///        a shortest path (composed by egdes) between three nodes
  ///
  std::vector<std::vector<int>>
  boundaryVector(std::vector<std::vector<int>> & shortPath);

  ///
  /// \brief Returns the 1-D boundary required to compute the minimum
  ///        surface of the input mesh boundary faces. The input to
  ///        this function is a shortest path (composed by edges)
  ///        between three nodes
  ///
  std::vector<std::vector<int>>
  boundaryVectorOuterSurface(std::vector<std::vector<int>> & shortPath);

  ///
  /// \brief Returns the corresponding entities of rank 2 that build
  ///        the minimum surface.  It takes as an input the resulting
  ///        vector taken from the solution of the linear programming
  ///        solver
  ///
  std::vector<stk::mesh::Entity>
  minimumSurfaceFaces(std::vector<int> VectorFromLPSolver);

  ///
  /// \brief Returns the number of times an entity is repeated in a vector
  ///
  int
  numberOfRepetitions(
      std::vector<stk::mesh::Entity> & entities,
      stk::mesh::Entity entity);

  ///
  /// \brief Returns the coordinates of an input node.
  ///        The input is the id of a node
  ///
  std::vector<double>
  findCoordinates(unsigned int nodeIdentifier);

  ///
  /// Check fracture criterion
  ///
  bool
  checkOpen(stk::mesh::Entity e);

  ///
  /// Initialization of the open field for fracture
  ///
  void
  initializeFractureState();

  ///----------------------------------------------------------------------
  ///
  /// \brief Practice creating the barycentric subdivision
  ///
  void
  barycentricSubdivisionAlt();

  ///
  /// \brief Divide former mesh segments by half
  ///
  void
  divideSegmentsHalf();

  void
  addCentroid();

  void
  connectCentroid();

  void
  addNewFaces();

  void
  connectNewFaces();

  ///
  /// \brief Place the entity in the root part that has the stk::topology
  /// associated with the given rank.
  ///
  void
  AssignTopology(
      stk::mesh::EntityRank const rank,
      stk::mesh::Entity const entity);

  ///
  /// Accessors and mutators
  ///
  Topology &
  get_topology()
  {
    return *this;
  }

  stk::mesh::EntityId
  get_entity_id(stk::mesh::Entity const entity);

  void
  set_stk_mesh_struct(Teuchos::RCP<Albany::AbstractSTKMeshStruct> const & sms)
  {
    stk_mesh_struct_ = sms;
  }

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> &
  get_stk_mesh_struct()
  {
    return stk_mesh_struct_;
  }

  void
  set_discretization(Teuchos::RCP<Albany::AbstractDiscretization> const & d)
  {
    discretization_ = d;
  }

  Teuchos::RCP<Albany::AbstractDiscretization> &
  get_discretization()
  {
    return discretization_;
  }

  Albany::STKDiscretization &
  get_stk_discretization()
  {
    return static_cast<Albany::STKDiscretization &>(
        *(get_discretization().get()));
  }

  stk::mesh::BulkData &
  get_bulk_data()
  {
    return *(get_stk_mesh_struct()->bulkData);
  }

  stk::mesh::MetaData &
  get_meta_data()
  {
    return *(get_stk_mesh_struct()->metaData);
  }

  size_t
  get_space_dimension()
  {
    return static_cast<size_t>(get_meta_data().spatial_dimension());
  }

  stk::mesh::EntityRank
  get_boundary_rank()
  {
    return get_meta_data().side_rank();
  }

  void
  set_bulk_block_name(std::string const & bn)
  {
    bulk_block_name_ = bn;
  }

  void
  set_interface_block_name(std::string const & in)
  {
    interface_block_name_ = in;
  }

  std::string const &
  get_bulk_block_name()
  {
    return bulk_block_name_;
  }

  std::string const &
  get_interface_block_name()
  {
    return interface_block_name_;
  }

  IntScalarFieldType &
  get_fracture_state_field(stk::mesh::EntityRank rank)
  {
    return
        *(get_stk_mesh_struct()->getFieldContainer()->getFractureState(rank));
  }

  void
  set_fracture_criterion(Teuchos::RCP<AbstractFractureCriterion> const & fc)
  {
    fracture_criterion_ = fc;
  }

  Teuchos::RCP<AbstractFractureCriterion> &
  get_fracture_criterion()
  {
    return fracture_criterion_;
  }

  stk::mesh::Part &
  get_bulk_part()
  {
    return *(get_meta_data().get_part(get_bulk_block_name()));
  }

  stk::mesh::Part &
  get_interface_part()
  {
    return *(get_meta_data().get_part(get_interface_block_name()));
  }

  shards::CellTopology
  get_cell_topology()
  {
    return get_meta_data().get_cell_topology(get_bulk_part());
  }

  stk::mesh::Part &
  get_local_part()
  {
    return get_meta_data().locally_owned_part();
  }

  stk::mesh::Selector
  get_local_bulk_selector()
  {
    return stk::mesh::Selector(get_local_part() & get_bulk_part());
  }

  stk::mesh::Selector
  get_local_interface_selector()
  {
    return stk::mesh::Selector(get_local_part() & get_interface_part());
  }

  int
  get_parallel_rank()
  {
    return get_bulk_data().parallel_rank();
  }

  bool
  is_local_entity(stk::mesh::Entity e)
  {
    return get_parallel_rank() == get_bulk_data().parallel_owner_rank(e);
  }

  bool
  is_in_bulk(stk::mesh::Entity e)
  {
    return get_bulk_data().bucket(e).member(get_bulk_part());
  }

  bool
  is_in_part(stk::mesh::Part & part, stk::mesh::Entity e)
  {
    return get_bulk_data().bucket(e).member(part);
  }

  bool
  is_bulk_cell(stk::mesh::Entity e)
  {
    return (get_bulk_data().entity_rank(e) == stk::topology::ELEMENT_RANK)
        && is_in_bulk(e);
  }

  bool
  is_in_interface(stk::mesh::Entity e)
  {
    return get_bulk_data().bucket(e).member(get_interface_part());
  }

  bool
  is_interface_cell(stk::mesh::Entity e)
  {
    return (get_bulk_data().entity_rank(e) == stk::topology::ELEMENT_RANK)
        && is_in_interface(e);
  }

  //
  // Set fracture state. Do nothing for cells (elements).
  //
  void
  set_fracture_state(stk::mesh::Entity e, FractureState const fs)
  {
    stk::mesh::EntityRank const rank = get_bulk_data().entity_rank(e);
    if (rank < stk::topology::ELEMENT_RANK) {
      *(stk::mesh::field_data(get_fracture_state_field(rank), e)) =
          static_cast<int>(fs);
    }
  }

  //
  // Get fracture state. Return CLOSED for cells (elements).
  //
  FractureState
  get_fracture_state(stk::mesh::Entity e)
  {
    stk::mesh::EntityRank const rank = get_bulk_data().entity_rank(e);
    return
    rank >= stk::topology::ELEMENT_RANK ?
        CLOSED :
        static_cast<FractureState>(*(stk::mesh::field_data(
            get_fracture_state_field(rank),
            e)));
  }

  bool
  is_internal(stk::mesh::Entity e)
  {

    assert(get_bulk_data().entity_rank(e) == get_boundary_rank());

    size_t const
    number_in_edges = get_bulk_data().num_elements(e);

    assert(number_in_edges == 1 || number_in_edges == 2);

    return number_in_edges == 2;
  }

  bool
  is_open(stk::mesh::Entity e)
  {
    return get_fracture_state(e) == OPEN;
  }

  bool
  is_internal_and_open(stk::mesh::Entity e)
  {
    return is_internal(e) == true && is_open(e) == true;
  }

  void
  set_output_type(OutputType const ot)
  {
    output_type_ = ot;
  }

  OutputType
  get_output_type()
  {
    return output_type_;
  }

  ///
  /// \brief Number of entities of a specific rank
  ///
  EntityVectorIndex
  get_num_entities(stk::mesh::EntityRank const entity_rank);

  stk::mesh::EntityId
  get_highest_id(stk::mesh::EntityRank const rank);

  void
  increase_highest_id(stk::mesh::EntityRank const rank)
  {
    ++highest_ids_[rank];
  }

  stk::topology
  get_rank_topology(stk::mesh::EntityRank const rank)
  {
    assert(rank < topologies_.size());
    return topologies_[rank];
  }

  ///
  /// Compute normal using first 3 nodes of boundary entity.
  ///
  minitensor::Vector<double>
  get_normal(stk::mesh::Entity boundary_entity);

private:
  ///
  /// \brief Create Albany discretization
  ///
  /// Called by constructor
  ///
  void
  createDiscretization();

  void
  initializeHighestIds();

  void
  initializeTopologies();

  //
  //
  Teuchos::RCP<Albany::AbstractDiscretization>
  discretization_;

  Teuchos::RCP<Albany::AbstractSTKMeshStruct>
  stk_mesh_struct_;

  std::vector<stk::mesh::EntityVector>
  connectivity_;

  std::set<EntityPair>
  fractured_faces_;

  std::vector<stk::topology>
  topologies_;

  std::vector<stk::mesh::EntityId>
  highest_ids_;

  std::string
  bulk_block_name_;

  std::string
  interface_block_name_;

  /// Pointer to failure criterion object
  Teuchos::RCP<AbstractFractureCriterion>
  fracture_criterion_;

  OutputType
  output_type_;

private:
  ///
  /// \brief Hide default constructor for Topology
  ///
  Topology();
};
// class Topology

}// namespace LCM

#endif // LCM_Topology_Topology_h
