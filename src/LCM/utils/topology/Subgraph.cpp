//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Subgraph.h"
#include "Topology.h"
#include "Topology_Utils.h"

namespace LCM {

//
// Create a subgraph given a vertex list and an edge list.
//
Subgraph::Subgraph(
    Topology&                             topology,
    std::set<stk::mesh::Entity>::iterator first_entity,
    std::set<stk::mesh::Entity>::iterator last_entity,
    std::set<STKEdge>::iterator           first_edge,
    std::set<STKEdge>::iterator           last_edge)
    : topology_(topology)
{
  // Insert vertices and create the vertex map
  for (std::set<stk::mesh::Entity>::iterator entity_iterator = first_entity;
       entity_iterator != last_entity;
       ++entity_iterator) {
    // get global vertex
    stk::mesh::Entity entity = *entity_iterator;

    // get entity rank
    stk::mesh::EntityRank entity_rank = get_bulk_data().entity_rank(entity);

    // create new local vertex
    Vertex vertex = boost::add_vertex(*this);

    std::pair<Vertex, stk::mesh::Entity> vertex_entity_pair =
        std::make_pair(vertex, entity);

    std::pair<stk::mesh::Entity, Vertex> entity_vertex_pair =
        std::make_pair(entity, vertex);

    vertex_entity_map_.insert(vertex_entity_pair);

    entity_vertex_map_.insert(entity_vertex_pair);

    // store entity rank to vertex property
    VertexNamePropertyMap vertex_property_map = boost::get(VertexName(), *this);

    boost::put(vertex_property_map, vertex, entity_rank);
  }

  // Add edges to the subgraph
  for (std::set<STKEdge>::iterator edge_iterator = first_edge;
       edge_iterator != last_edge;
       ++edge_iterator) {
    // Get the edge
    STKEdge stk_edge = *edge_iterator;

    // Get global source and target entities
    stk::mesh::Entity source_entity = stk_edge.source;

    stk::mesh::Entity target_entity = stk_edge.target;

    // Get local source and target vertices
    Vertex source_vertex = entity_vertex_map_.find(source_entity)->second;

    Vertex target_vertex = entity_vertex_map_.find(target_entity)->second;

    EdgeId edge_id = stk_edge.local_id;

    Edge graph_edge;

    bool is_inserted = false;

    boost::tie(graph_edge, is_inserted) =
        boost::add_edge(source_vertex, target_vertex, *this);

    assert(is_inserted == true);

    // Add edge id to edge property
    EdgeNamePropertyMap edge_property_map = boost::get(EdgeName(), *this);

    boost::put(edge_property_map, graph_edge, edge_id);
  }

  return;
}

//
// Accessors and mutators
//
Topology&
Subgraph::get_topology()
{
  return topology_;
}

size_t
Subgraph::get_space_dimension()
{
  return get_topology().get_space_dimension();
}

Teuchos::RCP<Albany::AbstractSTKMeshStruct>&
Subgraph::get_stk_mesh_struct()
{
  return get_topology().get_stk_mesh_struct();
}

stk::mesh::BulkData&
Subgraph::get_bulk_data()
{
  return get_topology().get_bulk_data();
}

stk::mesh::MetaData&
Subgraph::get_meta_data()
{
  return get_topology().get_meta_data();
}

stk::mesh::EntityId
Subgraph::get_entity_id(stk::mesh::Entity const entity)
{
  return get_topology().get_entity_id(entity);
}

stk::mesh::EntityRank
Subgraph::get_boundary_rank()
{
  return get_topology().get_boundary_rank();
}

IntScalarFieldType&
Subgraph::get_fracture_state_field(stk::mesh::EntityRank rank)
{
  return get_topology().get_fracture_state_field(rank);
}

void
Subgraph::set_fracture_state(stk::mesh::Entity e, FractureState const fs)
{
  get_topology().set_fracture_state(e, fs);
}

FractureState
Subgraph::get_fracture_state(stk::mesh::Entity e)
{
  return get_topology().get_fracture_state(e);
}

bool
Subgraph::is_open(stk::mesh::Entity e)
{
  return get_topology().is_open(e);
}

bool
Subgraph::is_internal_and_open(stk::mesh::Entity e)
{
  return get_topology().is_internal_and_open(e);
}

//
// Map a vertex in the subgraph to a entity in the stk mesh.
//
stk::mesh::Entity
Subgraph::entityFromVertex(Vertex vertex)
{
  VertexEntityMap::const_iterator vertex_entity_map_iterator =
      vertex_entity_map_.find(vertex);

  assert(vertex_entity_map_iterator != vertex_entity_map_.end());

  stk::mesh::Entity entity = (*vertex_entity_map_iterator).second;

  return entity;
}

//
// Map a entity in the stk mesh to a vertex in the subgraph.
//
Vertex
Subgraph::vertexFromEntity(stk::mesh::Entity entity)
{
  EntityVertexMap::const_iterator entity_vertex_map_iterator =
      entity_vertex_map_.find(entity);

  assert(entity_vertex_map_iterator != entity_vertex_map_.end());

  Vertex vertex = (*entity_vertex_map_iterator).second;

  return vertex;
}

//
// Add a vertex in the subgraph.
//
Vertex
Subgraph::addVertex(stk::mesh::EntityRank vertex_rank, stk::mesh::Entity entity)
{
  get_topology().increase_highest_id(vertex_rank);

  stk::mesh::EntityId low_id = get_topology().get_highest_id(vertex_rank);

  size_t const dimension = get_space_dimension();

  int const parallel_rank = get_bulk_data().parallel_rank();

  stk::mesh::EntityId high_id =
      high_id_from_low_id(dimension, parallel_rank, vertex_rank, low_id);

  stk::mesh::PartVector add_parts;

  // If the entity passed is a node, propagate parts to new entities.
  bool const propagate_parts =
      entity != INVALID_ENTITY && vertex_rank == stk::topology::NODE_RANK;

  if (propagate_parts == true) {
    std::map<std::string, stk::mesh::Part*>& ns_parts =
        get_stk_mesh_struct()->nsPartVec;

    stk::mesh::Selector select_local =
        stk::mesh::Selector(get_meta_data().locally_owned_part());

    for (auto it = ns_parts.begin(); it != ns_parts.end(); ++it) {
      stk::mesh::Part& ns_part = *(it->second);

      stk::mesh::Selector select_in_nodeset = stk::mesh::Selector(ns_part);

      stk::mesh::Selector select_local_in_nodeset =
          select_local & select_in_nodeset;

      std::vector<stk::mesh::Entity> ns_nodes;

      stk::mesh::get_selected_entities(
          select_local_in_nodeset,
          get_bulk_data().buckets(stk::topology::NODE_RANK),
          ns_nodes);

      bool const is_local_and_in_nodeset =
          std::find(ns_nodes.begin(), ns_nodes.end(), entity) != ns_nodes.end();

      if (is_local_and_in_nodeset == true) { add_parts.push_back(&ns_part); }
    }
  }

  stk::mesh::Entity new_entity =
      get_bulk_data().declare_entity(vertex_rank, high_id, add_parts);

  // Add the vertex to the subgraph
  Vertex new_vertex = boost::add_vertex(*this);

  // Update maps
  std::pair<Vertex, stk::mesh::Entity> vertex_entity_pair =
      std::make_pair(new_vertex, new_entity);

  std::pair<stk::mesh::Entity, Vertex> entity_vertex_pair =
      std::make_pair(new_entity, new_vertex);

  vertex_entity_map_.insert(vertex_entity_pair);

  entity_vertex_map_.insert(entity_vertex_pair);

  // store entity rank to the vertex property
  VertexNamePropertyMap vertex_property_map = boost::get(VertexName(), *this);

  boost::put(vertex_property_map, new_vertex, vertex_rank);

  return new_vertex;
}

//
// Remove vertex in subgraph
//
void
Subgraph::removeVertex(Vertex const vertex)
{
  // get the global entity of vertex
  stk::mesh::Entity entity = entityFromVertex(vertex);

  // remove the vertex and entity from global_local_vertex_map_ and
  // local_global_vertex_map_
  entity_vertex_map_.erase(entity);
  vertex_entity_map_.erase(vertex);

  // remove vertex from subgraph
  // first have to ensure that there are no edges in or out of the vertex
  boost::clear_vertex(vertex, *this);
  // remove the vertex
  boost::remove_vertex(vertex, *this);

  // remove the entity from stk mesh
  bool const deleted = get_bulk_data().destroy_entity(entity);

  assert(deleted == true);

  return;
}

//
// Add edge to local graph.
//
std::pair<Edge, bool>
Subgraph::addEdge(
    EdgeId const edge_id,
    Vertex const source_vertex,
    Vertex const target_vertex)
{
  // get global entities
  stk::mesh::Entity source_entity = entityFromVertex(source_vertex);

  stk::mesh::Entity target_entity = entityFromVertex(target_vertex);

  assert(
      get_bulk_data().entity_rank(source_entity) -
          get_bulk_data().entity_rank(target_entity) ==
      1);

  // Add edge to local graph
  std::pair<Edge, bool> graph_edge =
      boost::add_edge(source_vertex, target_vertex, *this);

  // If edge exists in graph, skip adding it to STK.
  if (graph_edge.second == false) return graph_edge;

  // Add edge to stk mesh
  get_bulk_data().declare_relation(source_entity, target_entity, edge_id);

  // Add edge id to edge property
  EdgeNamePropertyMap edge_property_map = boost::get(EdgeName(), *this);

  boost::put(edge_property_map, graph_edge.first, edge_id);

  return graph_edge;
}

//
//
//
void
Subgraph::removeEdge(Vertex const source_vertex, Vertex const target_vertex)
{
  // Get the local id of the edge in the subgraph
  Edge edge;

  bool inserted = false;

  boost::tie(edge, inserted) = boost::edge(source_vertex, target_vertex, *this);

  assert(inserted == true);

  EdgeId edge_id = getEdgeId(edge);

  // remove local edge
  boost::remove_edge(source_vertex, target_vertex, *this);

  // remove relation from stk mesh
  stk::mesh::Entity source_entity = entityFromVertex(source_vertex);

  stk::mesh::Entity target_entity = entityFromVertex(target_vertex);

  get_bulk_data().destroy_relation(source_entity, target_entity, edge_id);

  return;
}

//
//
//
stk::mesh::EntityRank
Subgraph::getVertexRank(Vertex const vertex)
{
  VertexNamePropertyMap vertex_property_map = boost::get(VertexName(), *this);
  return boost::get(vertex_property_map, vertex);
}

//
//
//
EdgeId
Subgraph::getEdgeId(Edge const edge)
{
  EdgeNamePropertyMap edge_property_map = boost::get(EdgeName(), *this);
  return boost::get(edge_property_map, edge);
}

// Here we use the connected components algorithm, which requires
// and undirected graph. These types are needed to build that graph
// without the overhead that was used for the subgraph.
// The adjacency list type must use a vector container for the vertices
// so that they can be converted to indices to determine the components.
typedef boost::adjacency_list<VectorS, VectorS, Undirected> UGraph;
typedef boost::graph_traits<UGraph>::vertex_descriptor      UVertex;
typedef boost::graph_traits<UGraph>::edge_descriptor        UEdge;

namespace {

void
writeGraphviz(std::string const& output_filename, UGraph const& graph)
{
  // Open output file
  std::ofstream gviz_out;

  gviz_out.open(output_filename.c_str(), std::ios::out);

  if (gviz_out.is_open() == false) {
    std::cout << "Unable to open graphviz output file :";
    std::cout << output_filename << '\n';
    return;
  }

  boost::write_graphviz(gviz_out, graph);

  gviz_out.close();

  return;
}

}  // anonymous namespace

//
// Function determines whether the input vertex is an articulation
// point of the subgraph.
//
void
Subgraph::testArticulationPoint(
    Vertex const        articulation_vertex,
    size_t&             number_components,
    VertexComponentMap& vertex_component_map)
{
  // Map to and from undirected graph and subgraph
  std::map<UVertex, Vertex> u_sub_vertex_map;

  std::map<Vertex, UVertex> sub_u_vertex_map;

  UGraph graph;

  VertexIterator vertex_begin;

  VertexIterator vertex_end;

  boost::tie(vertex_begin, vertex_end) = boost::vertices(*this);

  // First add the vertices to the graph
  for (VertexIterator i = vertex_begin; i != vertex_end; ++i) {
    Vertex vertex = *i;

    // If this is the vertex that is subjected to the articulation point test
    // skip it.
    if (vertex == articulation_vertex) continue;

    UVertex uvertex = boost::add_vertex(graph);

    // Add to maps
    std::pair<UVertex, Vertex> u_sub = std::make_pair(uvertex, vertex);

    std::pair<Vertex, UVertex> sub_u = std::make_pair(vertex, uvertex);

    u_sub_vertex_map.insert(u_sub);
    sub_u_vertex_map.insert(sub_u);
  }

  // Then add the edges
  for (VertexIterator i = vertex_begin; i != vertex_end; ++i) {
    Vertex source = *i;

    // If this is the vertex that is subjected to the articulation point test
    // skip it.
    if (source == articulation_vertex) continue;

    std::map<Vertex, UVertex>::const_iterator source_map_iterator =
        sub_u_vertex_map.find(source);

    UVertex usource = (*source_map_iterator).second;

    OutEdgeIterator out_edge_begin;

    OutEdgeIterator out_edge_end;

    boost::tie(out_edge_begin, out_edge_end) = boost::out_edges(source, *this);

    for (OutEdgeIterator j = out_edge_begin; j != out_edge_end; ++j) {
      Edge edge = *j;

      Vertex target = boost::target(edge, *this);

      // If this is the vertex that is subjected to the articulation point test
      // skip it.
      if (target == articulation_vertex) continue;

      std::map<Vertex, UVertex>::const_iterator target_map_iterator =
          sub_u_vertex_map.find(target);

      UVertex utarget = (*target_map_iterator).second;

      boost::add_edge(usource, utarget, graph);
    }
  }

#if defined(DEBUG_LCM_TOPOLOGY)
  {
    stk::mesh::Entity entity = entityFromVertex(articulation_vertex);

    stk::mesh::BulkData& bulk_data = get_bulk_data();

    std::string file_name =
        "undirected-" + entity_string(get_topology(), entity) + ".dot";

    writeGraphviz(file_name, graph);
  }
#endif  // DEBUG_LCM_TOPOLOGY

  std::vector<size_t> components(boost::num_vertices(graph));

  number_components = boost::connected_components(graph, &components[0]);

  for (std::map<UVertex, Vertex>::iterator i = u_sub_vertex_map.begin();
       i != u_sub_vertex_map.end();
       ++i) {
    Vertex vertex = (*i).second;

    size_t u_vertex = static_cast<size_t>((*i).first);

    std::pair<Vertex, size_t> component =
        std::make_pair(vertex, components[u_vertex]);

    vertex_component_map.insert(component);
  }

  return;
}

//
// Clones a boundary vertex from the subgraph and separates the in-edges
// of the entity.
//
Vertex
Subgraph::cloneBoundaryVertex(Vertex boundary_vertex)
{
  stk::mesh::EntityRank boundary_rank = getVertexRank(boundary_vertex);

  assert(boundary_rank == get_boundary_rank());

  Vertex vertex_clone = addVertex(boundary_rank);

  // Copy the out_edges of vertex to new_vertex
  OutEdgeIterator out_edge_begin;

  OutEdgeIterator out_edge_end;

  boost::tie(out_edge_begin, out_edge_end) =
      boost::out_edges(boundary_vertex, *this);

  for (OutEdgeIterator i = out_edge_begin; i != out_edge_end; ++i) {
    Edge edge = *i;

    EdgeId edge_id = getEdgeId(edge);

    Vertex target = boost::target(edge, *this);

    addEdge(edge_id, vertex_clone, target);
  }

  // Copy all out edges not in the subgraph to the new vertex
  cloneOutEdges(boundary_vertex, vertex_clone);

  // Remove one of the edges from vertex, copy to new_vertex
  InEdgeIterator in_edge_begin;

  InEdgeIterator in_edge_end;

  boost::tie(in_edge_begin, in_edge_end) =
      boost::in_edges(boundary_vertex, *this);

  // Arbitrarily remove the first edge from original vertex
  Edge edge = *(in_edge_begin);

  EdgeId edge_id = getEdgeId(edge);

  Vertex source = boost::source(edge, *this);

  removeEdge(source, boundary_vertex);

  // Add edge to new vertex
  addEdge(edge_id, source, vertex_clone);

  // Assign topology here, otherwise STK complains when we add and
  // remove edges from entities that their count is not correct.
  stk::mesh::Entity entity_clone = entityFromVertex(vertex_clone);

  get_topology().AssignTopology(boundary_rank, entity_clone);

  return vertex_clone;
}

//
// Restore element to node connectivity needed by STK.
//
void
Subgraph::updateEntityPointConnectivity(
    stk::mesh::Entity old_point,
    EntityEntityMap&  entity_new_point_map)
{
  for (EntityEntityMap::iterator i = entity_new_point_map.begin();
       i != entity_new_point_map.end();
       ++i) {
    stk::mesh::Entity entity = i->first;

    // Identify relation id and remove
    stk::mesh::Entity const* relations = get_bulk_data().begin_nodes(entity);

    stk::mesh::ConnectivityOrdinal const* ords =
        get_bulk_data().begin_node_ordinals(entity);

    size_t const num_relations = get_bulk_data().num_nodes(entity);

    EdgeId edge_id;

    bool found = false;

    for (size_t j = 0; j < num_relations; ++j) {
      if (relations[j] == old_point) {
        edge_id = ords[j];
        found   = true;
        break;
      }
    }

    assert(found == true);

    get_bulk_data().destroy_relation(entity, old_point, edge_id);

    stk::mesh::Entity new_point = i->second;

    get_bulk_data().declare_relation(entity, new_point, edge_id);
  }
  return;
}

//
// Splits an articulation point.
//
EntityEntityMap
Subgraph::splitArticulation(Vertex articulation_vertex)
{
  stk::mesh::EntityRank articulation_rank = getVertexRank(articulation_vertex);

  size_t num_components;

  VertexComponentMap vertex_component_map;

  testArticulationPoint(
      articulation_vertex, num_components, vertex_component_map);

  assert(num_components > 0);

  // The function returns an updated connectivity map.
  // If the vertex rank is not node, then this map will be empty.
  EntityEntityMap entity_split_point_map;

  if (num_components == 1) return entity_split_point_map;

  // If more than one component, split vertex in subgraph and stk mesh.
  std::vector<Vertex> split_vertices(num_components - 1);

  stk::mesh::Entity articulation_entity = entityFromVertex(articulation_vertex);

  for (std::vector<Vertex>::size_type i = 0; i < split_vertices.size(); ++i) {
    Vertex split_vertex = addVertex(articulation_rank, articulation_entity);

    split_vertices[i] = split_vertex;
  }

  // Create a map of entities to new node numbers
  // only if the articulation vertex is a node
  if (articulation_rank == stk::topology::NODE_RANK) {
    stk::mesh::Entity articulation_point =
        entityFromVertex(articulation_vertex);

    for (VertexComponentMap::iterator i = vertex_component_map.begin();
         i != vertex_component_map.end();
         ++i) {
      Vertex component_vertex = (*i).first;

      size_t component_number = (*i).second;

      stk::mesh::EntityRank rank = getVertexRank(component_vertex);

      bool const is_edge_or_lower = rank <= stk::topology::EDGE_RANK;

      if (is_edge_or_lower == true) continue;

      if (component_number == num_components - 1) continue;

      stk::mesh::Entity component_entity = entityFromVertex(component_vertex);

      Vertex split_vertex = split_vertices[component_number];

      stk::mesh::Entity split_point = entityFromVertex(split_vertex);

      std::pair<stk::mesh::Entity, stk::mesh::Entity> nc =
          std::make_pair(component_entity, split_point);

      entity_split_point_map.insert(nc);
    }

    updateEntityPointConnectivity(articulation_point, entity_split_point_map);
  }

  // Copy the out edges of the original vertex to the new vertex
  for (std::vector<Vertex>::size_type i = 0; i < split_vertices.size(); ++i) {
    cloneOutEdges(articulation_vertex, split_vertices[i]);
  }

  // Vector for edges to be removed. Vertex is source and edgeId the
  // local id of the edge
  std::vector<std::pair<Vertex, EdgeId>> edges_to_remove;

  // Iterate over the in edges of the vertex to determine which will
  // be removed
  InEdgeIterator in_edge_begin;

  InEdgeIterator in_edge_end;

  boost::tie(in_edge_begin, in_edge_end) =
      boost::in_edges(articulation_vertex, *this);

  for (InEdgeIterator i = in_edge_begin; i != in_edge_end; ++i) {
    Edge edge = *i;

    Vertex source_vertex = boost::source(edge, *this);

    VertexComponentMap::const_iterator component_iterator =
        vertex_component_map.find(source_vertex);

    assert(component_iterator != vertex_component_map.end());

    size_t source_component = (*component_iterator).second;

    stk::mesh::Entity source_entity = entityFromVertex(source_vertex);

    if (source_component < num_components - 1) {
      EdgeId edge_id = getEdgeId(edge);

      edges_to_remove.push_back(std::make_pair(source_vertex, edge_id));
    }
  }

  // Remove all edges in vector removed and replace with new edges
  for (std::vector<std::pair<Vertex, EdgeId>>::iterator i =
           edges_to_remove.begin();
       i != edges_to_remove.end();
       ++i) {
    std::pair<Vertex, EdgeId> edge = *i;

    Vertex source_vertex = edge.first;

    EdgeId edge_id = edge.second;

    VertexComponentMap::const_iterator component_iterator =
        vertex_component_map.find(source_vertex);

    assert(component_iterator != vertex_component_map.end());

    size_t source_component = (*component_iterator).second;

    assert(source_component < num_components - 1);

    removeEdge(source_vertex, articulation_vertex);

    Vertex split_vertex = split_vertices[source_component];

    std::pair<Edge, bool> inserted =
        addEdge(edge_id, source_vertex, split_vertex);

    assert(inserted.second == true);

    // Assign topology here, otherwise STK complains when we add and
    // remove edges from entities that their count is not correct.
    stk::mesh::Entity split_entity = entityFromVertex(split_vertex);

    get_topology().AssignTopology(articulation_rank, split_entity);
  }

  return entity_split_point_map;
}

//
// Clone all out edges of a vertex to a new vertex.
//
void
Subgraph::cloneOutEdges(Vertex old_vertex, Vertex new_vertex)
{
  // Get the entity for the old and new vertices
  stk::mesh::Entity old_entity = entityFromVertex(old_vertex);

  stk::mesh::Entity new_entity = entityFromVertex(new_vertex);

  // Iterate over the out edges of the old vertex and check against the
  // out edges of the new vertex. If the edge does not exist, add.
  stk::mesh::EntityRank const top_rank =
      get_bulk_data().entity_rank(old_entity);

  assert(get_bulk_data().entity_rank(new_entity) == top_rank);

  for (stk::mesh::EntityRank rank = stk::topology::NODE_RANK; rank <= top_rank;
       ++rank) {
    stk::mesh::Entity const* old_relations =
        get_bulk_data().begin(old_entity, rank);

    size_t const num_old_relations =
        get_bulk_data().num_connectivity(old_entity, rank);

    stk::mesh::ConnectivityOrdinal const* old_relation_ords =
        get_bulk_data().begin_ordinals(old_entity, rank);

    for (size_t i = 0; i < num_old_relations; ++i) {
      stk::mesh::Entity const* new_relations =
          get_bulk_data().begin(new_entity, rank);

      size_t const num_new_relations =
          get_bulk_data().num_connectivity(new_entity, rank);

      // Assume that the edge does not exist
      bool exists = false;

      for (size_t j = 0; j < num_new_relations; ++j) {
        if (old_relations[i] == new_relations[j]) {
          exists = true;
          break;
        }
      }

      if (exists == false) {
        EdgeId edge_id = old_relation_ords[i];

        stk::mesh::Entity target = old_relations[i];

        get_bulk_data().declare_relation(new_entity, target, edge_id);
      }
    }
  }

  return;
}

//
// \brief Output the graph associated with the mesh to graphviz .dot
// file for visualization purposes.
//
// \param[in] output file
//
// Similar to outputToGraphviz function in Topology class.
// If fracture criterion for entity is satisfied, the entity and all
// associated lower order entities are marked open. All open entities are
// displayed as such in output file.
//
// To create final output figure, run command below from terminal:
//   dot -Tpng <gviz_output>.dot -o <gviz_output>.png
//
void
Subgraph::outputToGraphviz(std::string const& output_filename)
{
  // Open output file
  std::ofstream gviz_out;

  gviz_out.open(output_filename.c_str(), std::ios::out);

  if (gviz_out.is_open() == false) {
    std::cout << "Unable to open graphviz output file: ";
    std::cout << output_filename << '\n';
    return;
  }

  std::cout << "Write graph to graphviz dot file: ";
  std::cout << output_filename << '\n';

  // Write beginning of file
  gviz_out << dot_header();

  VertexIterator vertices_begin;

  VertexIterator vertices_end;

  boost::tie(vertices_begin, vertices_end) = vertices(*this);

  for (VertexIterator i = vertices_begin; i != vertices_end; ++i) {
    Vertex vertex = *i;

    stk::mesh::Entity entity = entityFromVertex(vertex);

    stk::mesh::EntityRank const rank = get_bulk_data().entity_rank(entity);

    FractureState const fracture_state = get_fracture_state(entity);

    stk::mesh::EntityId const entity_id = get_entity_id(entity);

    gviz_out << dot_entity(
        get_space_dimension(),
        get_topology().get_parallel_rank(),
        entity,
        entity_id,
        rank,
        fracture_state);

    // write the edges in the subgraph
    OutEdgeIterator out_edge_begin;

    OutEdgeIterator out_edge_end;

    boost::tie(out_edge_begin, out_edge_end) = boost::out_edges(*i, *this);

    for (OutEdgeIterator j = out_edge_begin; j != out_edge_end; ++j) {
      Edge out_edge = *j;

      Vertex source = boost::source(out_edge, *this);

      Vertex target = boost::target(out_edge, *this);

      stk::mesh::Entity global_source = entityFromVertex(source);

      stk::mesh::Entity global_target = entityFromVertex(target);

      EdgeId edge_id = getEdgeId(out_edge);

      gviz_out << dot_relation(
          get_entity_id(global_source),
          get_bulk_data().entity_rank(global_source),
          get_entity_id(global_target),
          get_bulk_data().entity_rank(global_target),
          edge_id);
    }
  }

  // File end
  gviz_out << dot_footer();

  gviz_out.close();

  return;
}

}  // namespace LCM
