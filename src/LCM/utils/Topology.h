/**
 * Topology.h
 *
 *  Created on: Jul 11, 2011
 *      Author: jrthune
 */

#include <Teuchos_CommandLineProcessor.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/fem/CreateAdjacentEntities.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Shards_CellTopology.hpp>
#include <Shards_BasicTopologies.hpp>
#include <stk/Albany_AbstractDiscretization.hpp>
#include <stk/Albany_DiscretizationFactory.hpp>
#include <stk/Albany_STKDiscretization.hpp>
#include <Albany_Utils.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graphviz.hpp>

#ifndef TOPOLOGY_H_
#define TOPOLOGY_H_

namespace LCM{

typedef stk::mesh::Entity Entity;
typedef stk::mesh::EntityRank EntityRank;
typedef stk::mesh::RelationIdentifier EdgeId;
typedef stk::mesh::EntityKey EntityKey;
typedef boost::vertex_name_t VertexName;
typedef boost::edge_name_t EdgeName;
typedef boost::property<VertexName, EntityRank> VertexProperty;
typedef boost::property<EdgeName, EdgeId> EdgeProperty;
typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::bidirectionalS, VertexProperty, EdgeProperty> boostGraph;
typedef boost::property_map<boostGraph,VertexName>::type VertexNamePropertyMap;
typedef boost::property_map<boostGraph,EdgeName>::type EdgeNamePropertyMap;
typedef boost::graph_traits<boostGraph>::vertex_descriptor Vertex;
typedef boost::graph_traits<boostGraph>::edge_descriptor Edge;
typedef boost::graph_traits<boostGraph>::vertex_iterator VertexIterator;
typedef boost::graph_traits<boostGraph>::edge_iterator EdgeIterator;
typedef boost::graph_traits<boostGraph>::out_edge_iterator out_edge_iterator;
typedef boost::graph_traits<boostGraph>::in_edge_iterator in_edge_iterator;

class topology {
public:
	/**
	 * \brief Default constructor for topology
	 */
	topology();

	/**
	 * \brief Create mesh data structure
	 *
	 * \param[in] input_file is exodus II input file name
	 * \param[in] output_file is exodus II output file name
	 *
	 * Use if want to create new Albany mesh object
	 */
	topology(
			std::string const & input_file,
			std::string const & output_file);

	/**
	 * \brief Create mesh data structure
	 *
	 * \param[in] Albany discretization object
	 *
	 * Use if already have an Albany mesh object
	 */
	topology(Teuchos::RCP<Albany::AbstractDiscretization> &
	  discretization_ptr);

	/**
	 * \brief Output relations associated with entity
	 *        The entity may be of any rank
	 *
	 * \param[in] entity
	 */
	void
	disp_relation(
			Entity const & entity);

	/**
	 * \brief Output relations of rank entityRank associated with entity
	 *        the entity may be of any rank
	 *
	 * \param[in] entity
	 * \param[in] the rank of the entity
	 */
	void
	disp_relation(
			Entity const & entity,
			EntityRank const entityRank);

	/**
	 * \brief Output the mesh connectivity
	 *
	 * Outputs the nodal connectivity of the elements as stored by
	 * bulkData. Assumes that relationships between the elements and
	 * nodes exist.
	 */
	void
	disp_connectivity();

	/**
	 * \brief Generic fracture criterion function.
	 *
	 * \param[in] entity
	 * \param[in] probability
	 * \return is criterion met
	 *
	 * Given an entity and probability, will determine if fracture criterion
	 * is met. Will return true if fracture criterion is met, else false.
	 * Fracture only defined on surface of elements. Thus, input entity
	 * must be of rank dimension-1, else error. For 2D, entity rank must = 1.
	 * For 3D, entity rank must = 2.
	 */
	bool
	fracture_criterion(
			Entity& entity,
			float p);

	/**
	 * \brief Iterates over the boundary entities of the mesh of (all entities
	 * of rank dimension-1) and checks fracture criterion.
	 *
	 * \param map of entity and boolean value is entity open
	 *
	 * If fracture_criterion is met, the entity and all lower order entities
	 * associated with it are marked as open.
	 */
	void
	set_entities_open(
			std::map<EntityKey,bool>& entity_open);

	/**
	 * \brief Output the graph associated with the mesh to graphviz .dot
	 * file for visualization purposes.
	 *
	 * \param[in] output file
	 * \param[in] map of entity and boolean value is open
	 *
	 * If fracture criterion for entity is satisfied, the entity and all
	 * associated lower order entities are marked open. All open entities are
	 * displayed as such in output file.
	 *
	 * To create final output figure, run command below from terminal:
	 *   dot -Tpng <gviz_output>.dot -o <gviz_output>.png
	 */
	void
	output_to_graphviz(
			std::string & gviz_output,
			std::map<EntityKey, bool> & entity_open);

	/**
	 * \brief Initializes the default stk mesh object needed by class.
	 *
	 * Creates the full mesh representation of the mesh. Default stk mesh
	 * object has only elements and nodes. Function will delete unneeded
	 * relations between as described in topology::remove_extra_relations().
	 *
	 * \attention Function must be called before mesh modification begins.
	 *
	 * \attention Call function once. Creation of extra entities and relations
	 * is slow.
	 */
	void
	graph_initialization();

	/**
	 * \brief Removes unneeded relations from the mesh.
	 *
	 * stk::mesh::create_adjacent_entities creates full mesh representation of
	 * the mesh instead of the default of only the elements and nodes. All
	 * entities created by the function are connected through relationships.
	 * Graph algorithms require relationships to only exist between entities
	 * separated by one degree, e.g. elements and faces in a 3D graph.
	 * Function removes all other relationships.
	 *
	 * \note Valid for 2D and 3D meshes.
	 */
	void
	remove_extra_relations();

	/**
	 * \brief Creates temporary nodal connectivity for the elements and removes the
	 * relationships between the elements and nodes.
	 *
	 * \attention Must be called every time before mesh topology changes begin.
	 */
	void
	remove_node_relations();

	/**
	 * \brief After mesh manipulations are complete, need to recreate a stk
	 * mesh understood by Albany_STKDiscretization.
	 *
	 * Recreates the nodal connectivity using connectivity_temp.
	 *
	 * \attention must be called before mesh modification has ended
	 */
	void
	graph_cleanup();

	/**
	 * \brief Determine the nodes associated with a face.
	 *
	 * \param[in] Face entity
	 * \return vector of nodes for the face
	 *
	 * Return an ordered list of nodes which describe the input face. In 2D,
	 * the face of the element is a line segment. In 3D, the face is a surface.
	 * Generalized for all element types valid in stk_mesh. Valid in 2D and 3D.
	 *
	 * \attention Assumes all mesh elements are same type.
	 */
	std::vector<Entity*>
	get_face_nodes(Entity * entity);

	Teuchos::RCP<Albany::AbstractDiscretization>
	get_Discretization(){return discretization_ptr_;};

	stk::mesh::BulkData*
	get_BulkData(){return bulkData_;};

	Teuchos::RCP<Albany::AbstractSTKMeshStruct>
	get_stkMeshStruct(){return stkMeshStruct_;};

	/**
	 * \brief Creates a mesh of the fractured surfaces only.
	 *
	 *  Outputs the mesh as an exodus file for visual representation of split faces.
	 *
	 *  \todo output the exodus file
	 */
	void output_surface_mesh();

	/**
	 * \brief Create cohesive connectivity
	 *
	 * \param[in] Face 1
	 * \param[in] Face 2
	 * \return Cohesive connectivity
	 *
	 * Given the two faces after insertion process, create the connectivity
	 * of the cohesive element.
	 *
	 * \attention Assumes that all elements have the same topology
	 */
	std::vector<Entity*>
	create_cohesive_conn(Entity* face1, Entity* face2);

	///
	/// Ranks of all entities of the mesh.
	///
	EntityRank nodeRank;
	EntityRank edgeRank;
	EntityRank faceRank;
	EntityRank elementRank;

	///
	/// Number of dimensions in the mesh
	///
	int numDim;

	/**
	 * \brief Struct to store the data needed for creation or deletion of an
	 * edge in the stk mesh object.
	 *
	 * \param source entity key
	 * \param target entity key
	 * \param local id of the target entity with respect to the source
	 *
	 * To operate on an stk relation between entities (e.g. deleting a
	 * relation), need the source entity, target entity, and local ID of
	 * the relation with respect to the source entity.
	 *
	 * Used to create edges from the stk mesh object in a boost graph
	 */
	struct stkEdge {
		EntityKey source;
		EntityKey target;
		EdgeId localId;
	};

	/**
	 * Check if edges are the same
	 */
	struct EdgeLessThan:
		std::binary_function<stkEdge,stkEdge,bool>{
			bool operator()(const stkEdge & a, const stkEdge & b) const
			{
				if (a.source  < b.source)
					return true;
				if (a.source > b.source)
					return false;
				// source a and b are the same check target
				return (a.target < b.target);
			}
		};

	/**
	 * \brief Create vectors describing the vertices and edges of the star of an entity
	 *   in the stk mesh.
	 *
	 *   \param list of entities in the star
	 *   \param list of edges in the star
	 *   \param[in] source entity of the star
	 *
	 *   The star of a graph vertex is defined as the vertex and all higher order
	 *   vertices which are connected to it when traversing up the graph from the
	 *   input vertex.
	 *
	 *   \attention Valid for entities of all ranks
	 */
	void
	star(std::set<EntityKey> & subgraph_entity_lst,
			std::set<stkEdge,EdgeLessThan> & subgraph_edge_lst,
			Entity & entity);

	/**
	 * \brief Fractures all open boundary entities of the mesh.
	 *
	 * \param[in] map of entity and boolean value is entity open
	 *
	 * Iterate through the faces of the mesh and split into two faces if marked
	 * as open. The elements associated with an open face are separated. All
	 * lower order entities of the face are updated for a consistent mesh.
	 *
	 * \todo generalize the function for 2D meshes
	 */
	void
	fracture_boundary(std::map<EntityKey, bool> & entity_open);

private:

	/**
	 * \brief Create Albany discretization
	 *
	 * Called by constructor
	 */
	void
	create_discretization();

	Teuchos::RCP<Albany::AbstractDiscretization>
	  discretization_ptr_;

	stk::mesh::BulkData* bulkData_;

	Teuchos::RCP<Albany::AbstractSTKMeshStruct>
	stkMeshStruct_;

	std::vector<std::vector<Entity*> > connectivity_temp;

	std::set<std::pair<Entity*,Entity*> > fractured_face;

	///
	/// \attention Topology of elements in mesh. Only valid if one element type used.
	/// Will not give correct results if mesh has multiple element types.
	///
	shards::CellTopology element_topology;


}; // class topology

class Subgraph: public boostGraph {
public:
	// Default constructor
	Subgraph();

	/**
	 * \brief Create a subgraph given two vectors: a vertex list and a edge list.
	 *
	 * \param[in] bulkData for the stk mesh object
	 * \param[in] start of the vertex list
	 * \param[in] end of the vertex list
	 * \param[in] start of the edge list
	 * \param[in] end of the edge list
	 * \param[in] number of dimensions in the analysis
	 *
	 * Subgraph stored as a boost adjacency list.
	 * Maps are created to associate the subgraph to the global stk mesh graph.
	 * Any changes to the subgraph are automatically mirrored in the stk mesh.
	 */
	Subgraph(
			stk::mesh::BulkData* bulkData,
			std::set<EntityKey>::iterator firstVertex,
			std::set<EntityKey>::iterator lastVertex,
			std::set<topology::stkEdge>::iterator firstEdge,
			std::set<topology::stkEdge>::iterator lastEdge,
			int numDim);

	/**
	 * \brief Map a vertex in the subgraph to a entity in the stk mesh.
	 *
	 * \param[in] Vertex in the subgraph
	 * \return Global entity key for the stk mesh
	 *
	 * Return the global entity key (in the stk mesh) given a local
	 * subgraph vertex (in the boost subgraph).
	 */
	EntityKey
	local_to_global(Vertex localVertex);

	/**
	 * \brief Map a entity in the stk mesh to a vertex in the subgraph.
	 *
	 * \param[in] Global entity key for the stk mesh
	 * \return Vertex in the subgraph
	 *
	 * Return local vertex (in the boost graph) given global entity key (in the
	 *   stk mesh).
	 */
	Vertex
	global_to_local(EntityKey globalVertexKey);

	/**
	 * \brief Add a vertex in the subgraph.
	 *
	 * \param[in] Rank of vertex to be added
	 * \return New vertex
	 *
	 *   Mirrors the change in the subgraph by adding a corresponding entity
	 *   to the stk mesh. Adds the relationship between the vertex and entity
	 *   to the maps localGlobalVertexMap and globalLocalVertexMap.
	 */
	Vertex
	add_vertex(EntityRank vertex_rank);

	/**
	 * \brief Remove vertex in subgraph
	 *
	 * \param[in] Vertex to be removed
	 *
	 * When the vertex is removed from the subgraph the corresponding entity
	 * from the stk mesh is also removed.
	 *
	 * Both boost and stk require that all edges to and from the vertex/entity
	 * are removed before deletion. If any edges remain, will be removed
	 * before the vertex/entity deletion.
	 */
	void
	remove_vertex(Vertex & vertex);

	/**
	 * \brief Add edge to local graph.
	 *
	 * \param[in] Local ID of the target vertex with respect to the srouce vertex
	 * \param[in] Source vertex in the subgraph
	 * \param[in] Target vertex in the subgraph
	 * \return New edge and boolean value. If true, edge was inserted, if false
	 *  not inserted
	 *
	 * The edge insertion is mirrored in stk mesh. The edge is only inserted
	 * into the stk mesh object if it was inserted into the subgraph.
	 */
	std::pair<Edge,bool>
	add_edge(const EdgeId edge_id,
			const Vertex localSourceVertex,
			const Vertex localTargetVertex);


	/**
	 * \brief Remove edge from graph
	 *
	 * \param[in] Source vertex in subgraph
	 * \param[in] Target vertex in subgraph
	 *
	 * Edge removal is mirrored in the stk mesh.
	 *
	 */
	void
	remove_edge(
			const Vertex & localSourceVertex,
			const Vertex & localTargetVertex);

	/**
	 * \param[in] Vertex in subgraph
	 * \return Rank of vertex
	 */
	EntityRank &
	get_vertex_rank(const Vertex vertex);

	/**
	 * \param[in] Edge in subgraph
	 * \return Local numbering of edge target with respect to edge source
	 *
	 * In stk mesh, all relationships between entities have a local Id
	 * representing the correct ordering. Need this information to create
	 * or delete relations in the stk mesh.
	 */
	EdgeId &
	get_edge_id(const Edge edge);

	/**
	 * \brief Function determines whether the input vertex is an articulation
	 * point of the subgraph.
	 *
	 * \param[in] Input vertex
	 * \param[out] Number of components
	 * \param[out] map of vertex and associated component number
	 *
	 * Function checks vertex by the boost connected components algorithm to a
	 * copy of the subgraph. The copy does not include the input vertex.
	 * Copy is an undirected graph as required by the connected components
	 * algorithm.
	 *
	 * Returns the number of connected components as well as a map of the
	 * vertex in the subgraph and the component number.
	 */
	void
	undirected_graph(Vertex input_vertex,
			int & numComponents,
			std::map<Vertex,int> & subComponent);

	/**
	 * \brief Clones a boundary entity from the subgraph and separates the in-edges
	 * of the entity.
	 *
	 * \param[in] Boundary vertex
	 * \param[out] New boundary vertex
	 * \param Map of entity and boolean value is open
	 *
	 * Boundary entities are on boundary of the elements in the mesh. They
	 * will thus have either 1 or 2 in-edges to elements.
	 *
	 * If there is only 1 in-edge, the entity may be on the exterior of the
	 * mesh and is not a candidate for fracture for this subgraph. The
	 * boundary entity may be a valid candidate in another step. If only 1
	 * in edge: Return.
	 *
	 * Entity must have satisfied the fracture criterion and be labeled open
	 * in map is_open. If not open: Return.
	 */
	void
	clone_boundary_entity(Vertex & vertex,
			Vertex & newVertex,
			std::map<EntityKey,bool> & entity_open);

	/**
	 * \brief Splits an articulation point.
	 *
	 * \param[in] Input vertex
	 * \param Map of entity and boolean value is open
	 * \return Map of element and new node
	 *
	 * An articulation point is defined as a vertex which if removed
	 * yields a graph with more than 1 connected components. Creates
	 * an undirected graph and checks connected components of graph without
	 * vertex. Check if vertex is articulation point.
	 *
	 * Clones articulation point and splits in-edges between original and new
	 * vertices. The out-edges of the vertex are not in the subgraph. For
	 * a consistent global graph, add the out-edges of the vertex to the new
	 * vertex/vertices.
	 *
	 * If the vertex is a node, create a map between the element and the new
	 * node. If the nodal connectivity of an element does not change, do not
	 * add to the map.
	 */
	std::map<Entity*,Entity* >
	split_articulation_point(Vertex vertex,std::map<EntityKey,bool> & entity_open);

	/**
	 * \brief Clone all out edges of a vertex to a new vertex.
	 *
	 * \param[in] Original vertex
	 * \param[in] New vertex
	 *
	 * The global graph must remain consistent when new vertices are added. In
	 * split_articulation_point and clone_boundary_entity, all out-edges of
	 * the original vertex may not be in the subgraph.
	 *
	 * If there are missing edges in the subgraph, clone them from the original
	 * vertex to the new vertex. Edges not originally in the subgraph are added
	 * to the global graph only.
	 */
	void
	clone_out_edges(Vertex & originalVertex, Vertex & newVertex);

	/**
	 * \brief Output the graph associated with the mesh to graphviz .dot
	 * file for visualization purposes.
	 *
	 * \param[in] output file
	 * \param[in] map of entity and boolean value is open
	 *
	 * Similar to output_to_graphviz function in topology class.
	 * If fracture criterion for entity is satisfied, the entity and all
	 * associated lower order entities are marked open. All open entities are
	 * displayed as such in output file.
	 *
	 * To create final output figure, run command below from terminal:
	 *   dot -Tpng <gviz_output>.dot -o <gviz_output>.png
	 */
	void
	output_to_graphviz(
			std::string & gviz_output,
			std::map<EntityKey,bool> entity_open);


private:
	///
	/// Number of dimensions
	///
	int numDim_;

	///
	/// stk mesh data
	///
	stk::mesh::BulkData* bulkData_;

	///
	/// map local vertex -> global entity key
	///
	std::map<Vertex,EntityKey> localGlobalVertexMap;

	///
	/// map global entity key -> local vertex
	///
	std::map<EntityKey, Vertex> globalLocalVertexMap;

};  // class Subgraph

} // namespace LCM

#endif /* TOPOLOGY_H_ */
