/*
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

typedef stk::mesh::EntityRank EntityRank;
typedef stk::mesh::RelationIdentifier EdgeId;
typedef stk::mesh::EntityKey entityKey;
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

class topology {
public:
	/*
	 * Default constructor for topology
	 */
	topology();

	/*
	 * Create mesh data structure
	 *   Inputs:
	 *     input_file is exodus II input file name
	 *     output_file is exodus II output file name
	 */
	topology(
			std::string const & input_file,
			std::string const & output_file);

	/*
	 * Output relations associated with entity
	 */
	void
	disp_relation(
			stk::mesh::Entity & entity);

	/*
	 * Output relations of rank entityRank associated with entity
	 */
	void
	disp_relation(
			stk::mesh::Entity & entity,
			stk::mesh::EntityRank entityRank);

	/*
	 * Output the mesh connectivity
	 *   stk mesh must contain relations between the elements and the nodes (as in
	 *   the original stk mesh object)
	 */
	void
	disp_connectivity();

	/*
	 * Generic fracture criterion function. Given an entity and
	 *   probability, will determine if criterion is met.
	 *   Will return true if fracture criterion is met, else false.
	 *   Fracture only defined on surface of elements. Thus, input entity
	 *   must be of rank dimension-1, else error.
	 */
	bool
	fracture_criterion(
			stk::mesh::Entity& entity,
			float p);

	/*
	 * Iterates over the boundary entities of the mesh of (all entities of rank
	 *   dimension-1) and checks fracture criterion. If fracture_criterion is
	 *   met, the entity and all lower order entities associated with it are
	 *   marked as open.
	 */
	void
	set_entities_open(
			std::map<stk::mesh::EntityKey,bool>& entity_open);

	/*
	 * Output the graph associated with the mesh to graphviz .dot
	 *   file for visualization purposes. If fracture criterion for
	 *   entity is satisfied, the entity and all associated lower order
	 *   entities are marked open. All open entities are displayed as
	 *   such in output file.
	 * Can create figure using:
	 *   dot -Tpng output.dot -o output.png
	 */
	void
	output_to_graphviz(
			std::map<stk::mesh::EntityKey, bool> & entity_open);

	/*
	 * Creates the full graph representation of the mesh. Default graph has only
	 *   elements and nodes. The original node connectivity will be deleted in
	 *   later steps, store the connectivity in temporary array.
	 *
	 * Note: Function must be called before mesh modification begins
	 */
	void
	graph_initialization(
			std::vector<std::vector<stk::mesh::Entity*> >  & connectivity_temp);

	/*
	 * stk::mesh::create_adjacent_entities creates all entities in
	 *   graph instead of default elements and nodes. All entities are
	 *   connected through relationships. Graph algorithms require
	 *   relationships only between entities separated by one degree, e.g. elements
	 *   and faces in a 3D graph.
	 *   Function removes all other relationships, e.g. between elements
	 *   and nodes.
	 *
	 * Note: Valid for 2D and 3D meshes.
	 */
	void
	remove_extra_relations();

	/*
	 * After mesh manipulations are complete, need to recreate the original
	 *   mesh representation as expected by Albany_STKDiscretization. Remove
	 *   all extra entities (faces and edges for a 3D mesh) and recreate relationships
	 *   between elements and nodes. Nodal connectivity data for each element is stored
	 *   in connectivity_temp.
	 *
	 * Note: must be called before mesh modification has ended
	 */
	void
	graph_cleanup(
			std::vector<std::vector<stk::mesh::Entity*> >  & connectivity_temp);

	/*
	 * Used to split elements of a mesh along faces.
	 *   Duplicates old_entity and removes connection between in_entity and old_entity.
	 *   A new entity with the same out_edges (i.e. connectivity to lower order
	 *   entities) is created.
	 *   old_entity is the original entity before the duplication
	 *   in_entity is a higher order entity with an out_edge connected to old_entity
	 *
	 * Note: Must be called after mesh modification has begun.
	 *
	 *       Does *Not* utilize the method from Mota, et al. 2008. Proof of concept for
	 *       mesh modification only.
	 */
	void
	duplicate_entity(
			stk::mesh::Entity & in_entity,
			stk::mesh::Entity & old_entity,
			std::vector<std::vector<stk::mesh::Entity*> > & connectivity_temp,
			int local_id,
			stk::mesh::Entity & element);

	Teuchos::RCP<Albany::AbstractDiscretization>
	get_Discretization(){return discretization_ptr_;};

	stk::mesh::BulkData*
	get_BulkData(){return bulkData_;};

	Teuchos::RCP<Albany::AbstractSTKMeshStruct>
	get_stkMeshStruct(){return stkMeshStruct_;};

	// Ranks of all entities of the mesh.
	EntityRank nodeRank;
	EntityRank edgeRank;
	EntityRank faceRank;
	EntityRank elementRank;

	int numDim;

	/*
	 * To operate on an stk relation between entities (e.g. deleting a
	 *   relation), need the source entity, target entity, and local ID of
	 *   the relation with respect to the source entity. Store information
	 *   for creation of an associated edge in a boost graph.
	 */
	struct stkEdge {
		entityKey source;
		entityKey target;
		EdgeId localId;
	};

	// Check if edges are the same
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

	/*
	 * Create vectors describing the vertices and edges of the star of an entity
	 *   in the stk mesh. Currently, only allowed input is a node.
	 *   TODO: allow arbitrary rank input.
	 *
	 *   The star of a graph vertex is defined as the vertex and all higher order
	 *   vertices which are connected to it when traversing up the graph from the
	 *   input vertex.
	 */
	void
	star(std::set<entityKey> & subgraph_entity_lst,
			std::set<stkEdge,EdgeLessThan> & subgraph_edge_lst,
			stk::mesh::Entity & entity);

private:

	Teuchos::RCP<Albany::AbstractDiscretization>
	  discretization_ptr_;

	stk::mesh::BulkData* bulkData_;

	Teuchos::RCP<Albany::AbstractSTKMeshStruct>
	stkMeshStruct_;


}; // class topology

class Subgraph: public boostGraph {
public:
	// Default constructor
	Subgraph();

	/*
	 * Create a subgraph given two vectors: a vertex list and a edge list.
	 *   Subgraph stored as a boost adjacency list.
	 *   Maps the subgraph to the global stk mesh graph.
	 */
	Subgraph(
			stk::mesh::BulkData* bulkData,
			std::set<entityKey>::iterator firstVertex,
			std::set<entityKey>::iterator lastVertex,
			std::set<topology::stkEdge>::iterator firstEdge,
			std::set<topology::stkEdge>::iterator lastEdge,
			int numDim);

	/*
	 * Return the global entity key given a local subgraph vertex.
	 */
	entityKey
	local_to_global(Vertex localVertex);

	/*
	 * Return local vertex given global entity key.
	 */
	Vertex
	global_to_local(entityKey globalVertexKey);

	/*
	 * Add a vertex in the subgraph.
	 *   Mirrors the change in the stk mesh
	 *   Input: rank of vertex (entity) to be created
	 *   Output: new vertex
	 */
	Vertex
	add_vertex(EntityRank vertex_rank);

	/*
	 * Remove vertex in subgraph
	 *   Mirror in stk mesh
	 */
	void
	remove_vertex(Vertex & vertex);

	/*
	 * Add edge to local graph.
	 *   Mirror change in stk mesh
	 *   Return true if edge inserted in local graph, else false.
	 *   If false, will not insert edge into stk mesh.
	 */
	std::pair<Edge,bool>
	add_edge(const EdgeId edge_id,
			const Vertex localSourceVertex,
			const Vertex localTargetVertex);


	/*
	 * Remove edge from graph
	 *   Mirror change in stk mesh
	 */
	void
	remove_edge(
			const Vertex & localSourceVertex,
			const Vertex & localTargetVertex);

	EntityRank &
	get_vertex_rank(const Vertex vertex);

	EdgeId &
	get_edge_id(const Edge edge);

	/*
	 * The connected components boost algorithm requires an undirected graph.
	 *   Subgraph creates a directed graph. Copy all vertices and edges from the
	 *   subgraph into the undirected graph.
	 */
	void
	undirected_graph();

	/*
	 * Clones a boundary entity from the subgraph and separates the in-edges
	 *   of the entity.
	 *   Boundary entities are on boundary of the elements in the mesh. They
	 *   will thus have either 1 or 2 in-edges to elements.
	 *
	 *   If there is only 1 in-edge, the entity is an exterior entity of the
	 *   mesh and is not a candidate for fracture. If only 1 in-edge: Return error.
	 *
	 *   Entity must have satisfied the fracture criterion and be labeled open
	 *   in is_open. If not open: Return error.
	 */
	void
	clone_boundary_entity(Vertex vertex);

	/*
	 * Splits an articulation point.
	 *   An articulation point is defined as a vertex which if removed
	 *   yields a graph with more than 1 connected components. Creates
	 *   an undirected graph and checks connected components of graph without
	 *   vertex. Check if vertex is articulation point
	 *
	 *   Clones articulation point and splits in-edges between original and new
	 *   vertices.
	 */
	void
	split_articulation_point(Vertex vertex);

	/*
	 * Similar to the function in the topology class. Will output the subgraph in a
	 * .dot file readable by the graphviz program dot.
	 *
	 * To create a figure from the file:
	 *   dot -Tpng output.dot -o output.png
	 */
	void
	output_to_graphviz(std::map<entityKey,bool> entity_open);

	int numDim_;

private:
	// stk mesh data
	stk::mesh::BulkData* bulkData_;

	// map local vertex -> global entity key
	std::map<Vertex,stk::mesh::EntityKey> localGlobalVertexMap;

	// map global entity key -> local vertex
	std::map<stk::mesh::EntityKey, Vertex> globalLocalVertexMap;

};  // class Subgraph

} // namespace LCM

#endif /* TOPOLOGY_H_ */
