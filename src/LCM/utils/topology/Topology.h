//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(LCM_Topology_Topology_h)
#define LCM_Topology_Topology_h

#include <iterator>

#include <stk_mesh/base/FieldData.hpp>

#include "Topology_Types.h"
#include "Topology_FractureCriterion.h"
#include "Topology_Utils.h"

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
  /// \brief Iterates over the boundary entities of the mesh of (all entities
  /// of rank dimension-1) and checks fracture criterion.
  ///
  /// If fracture_criterion is met, the entity and all lower order entities
  /// associated with it are marked as open.
  ///
  size_t
  setEntitiesOpen();

  ///
  /// \brief Output the graph associated with the mesh to graphviz .dot
  /// file for visualization purposes.
  ///
  /// \param[in] output file
  ///
  /// To create final output figure, run command below from terminal:
  ///   dot -Tpng <gviz_output>.dot -o <gviz_output>.png
  ///
  enum OutputType {
    UNIDIRECTIONAL_UNILEVEL,
    UNIDIRECTIONAL_MULTILEVEL,
    BIDIRECTIONAL_UNILEVEL,
    BIDIRECTIONAL_MULTILEVEL
  };

  void
  outputToGraphviz(
      std::string const & output_filename,
      OutputType const output_type = UNIDIRECTIONAL_UNILEVEL);

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
  /// \brief After mesh manipulations are complete, need to recreate
  ///        a stk mesh understood by Albany_STKDiscretization.
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
  /// Return an ordered list of nodes which describe the input face. In 2D,
  /// the face of the element is a line segment. In 3D, the face is a surface.
  /// Generalized for all element types valid in stk_mesh. Valid in 2D and 3D.
  ///
  /// \attention Assumes all mesh elements are same type.
  ///
  EntityVector
  getBoundaryEntityNodes(Entity const & boundary_entity);

  std::vector<Intrepid::Vector<double> >
  getNodalCoordinates();

  ///
  /// \brief Output boundary
  ///
  void
  outputBoundary(std::string const & output_filename);

  ///
  /// \brief Create boundary mesh
  ///
  void
  createBoundary();

  ///
  /// \brief Get a connectivity list of the boundary
  ///
  std::vector<std::vector<EntityId> >
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
  EntityVector
  createSurfaceElementConnectivity(
      Entity const & face_top, Entity const & face_bottom);

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
      Entity & entity,
      std::set<EntityKey> & subgraph_entities,
      std::set<stkEdge, EdgeLessThan> & subgraph_edges);

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
  getEntitiesByRank(BulkData const & mesh, EntityRank entity_rank);

  ///
  /// \brief Number of entities of a specific rank
  ///
  EntityVector::size_type
  getNumberEntitiesByRank(BulkData const & mesh, EntityRank entity_rank);

  ///
  /// \brief Gets the local relation id (0,1,2,...) between two entities
  ///
  EdgeId
  getLocalRelationId(Entity const & source_entity,
      Entity const & target_entity);

  ///
  /// \brief Returns the total number of lower rank entities
  ///        connected to a specific entity
  ///
  int
  getNumberLowerRankEntities(Entity const & entity);

  ///
  /// \brief Returns a group of entities connected directly to a
  ///        given entity. The input rank is the rank of the
  ///        returned entities.
  ///
  EntityVector
  getDirectlyConnectedEntities(
      Entity const & entity,
      EntityRank entity_rank);

  ///
  /// \brief Checks if an entity exists inside a specific vector
  ///
  bool
  findEntityInVector(EntityVector & entities, Entity * entity);

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
  getBoundaryEntities(Entity const & entity, EntityRank entity_rank);

  ///
  /// \brief Checks if a segment is connected to an input node.
  /// Returns "true" if the segment connects to the node.
  ///
  bool
  segmentIsConnected(Entity const & segment, Entity * node);

  ///
  /// \brief Finds the adjacent segments to a given segment. The
  ///        adjacent segments are connected to a given common
  ///        point. it returns adjacent segments
  ///
  EntityVector
  findAdjacentSegments(Entity const & segment, Entity * node);

  ///
  /// \brief Returns all the highest dimensional topology entities
  ///        to which a given face belongs
  ///
  EntityVector
  findCellRelations(Entity const & face);

  ///
  /// \brief Returns all the segments at the boundary of a given
  ///        element. Including those connected between the faces
  ///        barycenters and the faces boundary nodes
  ///
  EntityVector
  findSegmentsFromElement(Entity const & element);

  ///
  /// \brief Returns true if the faces share a segment (two points)
  ///
  bool
  facesShareTwoPoints(Entity const & face1, Entity const & face2);

  ///
  /// \brief returns the adjacent segments from a given face
  ///
  EntityVector
  findAdjacentSegmentsFromFace(
      std::vector<EntityVector> const & faces_inside_element,
      Entity const & face,
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
  getFormerElementNodes(Entity const & element,
      std::vector<EntityVector> const & entities);

  ///
  /// \brief Generates the coordinate of a given barycenter
  ///        "entities" is a vector with points that belong to the same
  ///        entity of the barycenter(e.g segment, face, or element)
  ///
  void
  computeBarycentricCoordinates(
      EntityVector const & entities,
      Entity * barycenter);

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
  shortestpathOnBoundaryFaces(
      std::vector<Entity*> const & nodes,
		  std::vector<Entity*> const & MeshEdgesShortestPath);

  ///
  /// \brief Returns the shortest path between three input nodes
  ///
  std::vector<std::vector<int> >
  shortestpath(std::vector<Entity*> const & nodes);

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
  /// Accessors and mutators
  ///
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

  Albany::STKDiscretization *
  getSTKDiscretization()
  {return static_cast<Albany::STKDiscretization*>(discretization_.get());}

  BulkData *
  getBulkData()
  {return stk_mesh_struct_->bulkData;}

  stk_classic::mesh::fem::FEMMetaData *
  getMetaData()
  {return stk_mesh_struct_->metaData;}

  void
  setCellTopology(shards::CellTopology const & ct)
  {cell_topology_ = ct;}

  shards::CellTopology &
  getCellTopology()
  {return cell_topology_;}

  size_t const
  getSpaceDimension() {return static_cast<size_t>(getSTKMeshStruct()->numDim);}

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

  void
  setFractureCriterion(RCP<AbstractFractureCriterion> const & fc)
  {fracture_criterion_ = fc;}

  RCP<AbstractFractureCriterion> &
  getFractureCriterion()
  {return fracture_criterion_;}

  bool
  isLocalEntity(Entity const & e)
  {return getBulkData()->parallel_rank() == e.owner_rank();}

  //
  // Set fracture state. Do nothing for cells (elements).
  //
  void
  setFractureState(Entity const & e, FractureState const fs)
  {
    if (e.entity_rank() < getCellRank()) {
      *(stk_classic::mesh::field_data(getFractureState(), e)) = static_cast<int>(fs);
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
    static_cast<FractureState>(*(stk_classic::mesh::field_data(getFractureState(), e)));
  }

  bool
  isInternal(Entity const & e) {

    assert(e.entity_rank() == getBoundaryRank());

    PairIterRelation
    relations = relations_one_up(e);

    size_t const
    number_in_edges = std::distance(relations.begin(), relations.end());

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

  bool
  checkOpen(Entity const & e)
  {
    return fracture_criterion_->check(e);
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

  //
  //
  RCP<Albany::AbstractDiscretization> discretization_;

  RCP<Albany::AbstractSTKMeshStruct> stk_mesh_struct_;

  std::vector<EntityVector> connectivity_;

  std::map<int, int> element_global_to_local_ids_;

  std::set<EntityPair> fractured_faces_;

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

}// namespace LCM

#endif // LCM_Topology_Topology_h
