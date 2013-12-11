//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Subgraph.h"

namespace LCM {

//
// Create a subgraph given a vertex list and an edge list.
//
Subgraph::Subgraph(BulkData * bulk_data,
    std::set<EntityKey>::iterator first_vertex,
    std::set<EntityKey>::iterator last_vertex,
    std::set<stkEdge>::iterator first_edge,
    std::set<stkEdge>::iterator last_edge,
    int dimension) : bulk_data_(bulk_data), dimension_(dimension)
{
  // Insert vertices and create the vertex map
  for (std::set<EntityKey>::iterator vertex_iterator = first_vertex;
      vertex_iterator != last_vertex;
      ++vertex_iterator) {

    // get global vertex
    EntityKey
    global_vertex = *vertex_iterator;

    // get entity rank
    EntityRank
    vertex_rank = bulk_data_->get_entity(global_vertex)->entity_rank();

    // create new local vertex
    Vertex
    local_vertex = boost::add_vertex(*this);

    std::pair<Vertex, EntityKey>
    local_to_global = std::make_pair(local_vertex, global_vertex);

    std::pair<EntityKey, Vertex>
    global_to_local = std::make_pair(global_vertex, local_vertex);

    local_global_vertex_map_.insert(local_to_global);

    global_local_vertex_map_.insert(global_to_local);

    // store entity rank to vertex property
    VertexNamePropertyMap
    vertex_property_map = boost::get(VertexName(), *this);

    boost::put(vertex_property_map, local_vertex, vertex_rank);
  }

  // Add edges to the subgraph
  for (std::set<stkEdge>::iterator edge_iterator = first_edge;
      edge_iterator != last_edge;
      ++edge_iterator) {

    // Get the edge
    stkEdge
    global_edge = *edge_iterator;

    // Get global source and target vertices
    EntityKey
    global_source_vertex = global_edge.source;

    EntityKey
    global_target_vertex = global_edge.target;

    // Get local source and target vertices
    Vertex
    local_source_vertex =
        global_local_vertex_map_.find(global_source_vertex)->second;

    Vertex
    local_target_vertex =
        global_local_vertex_map_.find(global_target_vertex)->second;

    EdgeId
    edge_id = global_edge.local_id;

    Edge
    local_edge;

    bool
    is_inserted;

    boost::tie(local_edge, is_inserted) =
        boost::add_edge(local_source_vertex, local_target_vertex, *this);

    assert(is_inserted == true);

    // Add edge id to edge property
    EdgeNamePropertyMap
    edge_property_map = boost::get(EdgeName(), *this);

    boost::put(edge_property_map, local_edge, edge_id);
  }

  return;
}

//
// Map a vertex in the subgraph to a entity in the stk mesh.
//
EntityKey
Subgraph::localToGlobal(Vertex local_vertex)
{
  std::map<Vertex, EntityKey>::const_iterator
  vertex_map_iterator = local_global_vertex_map_.find(local_vertex);

  assert(vertex_map_iterator != local_global_vertex_map_.end());

  return (*vertex_map_iterator).second;
}

//
// Map a entity in the stk mesh to a vertex in the subgraph.
//
Vertex
Subgraph::globalToLocal(EntityKey global_vertex_key)
{
  std::map<EntityKey, Vertex>::const_iterator
  vertex_map_iterator = global_local_vertex_map_.find(global_vertex_key);

  assert(vertex_map_iterator != global_local_vertex_map_.end());

  return (*vertex_map_iterator).second;
}

//
// Add a vertex in the subgraph.
//
Vertex
Subgraph::addVertex(EntityRank vertex_rank)
{
  // Insert the vertex into the stk mesh
  // First have to request a new entity of rank N
  // number of entity ranks: 1 + number of dimensions
  std::vector<size_t>
  requests(dimension_ + 1, 0);

  requests[vertex_rank] = 1;

  EntityVector
  new_entities;

  bulk_data_->generate_new_entities(requests, new_entities);

  EntityKey
  global_vertex = new_entities[0]->key();

  // Add the vertex to the subgraph
  Vertex
  local_vertex = boost::add_vertex(*this);

  // Update maps
  std::pair<Vertex, EntityKey>
  local_to_global = std::make_pair(local_vertex, global_vertex);

  std::pair<EntityKey, Vertex>
  global_to_local = std::make_pair(global_vertex, local_vertex);

  local_global_vertex_map_.insert(local_to_global);

  global_local_vertex_map_.insert(global_to_local);

  // store entity rank to the vertex property
  VertexNamePropertyMap
  vertex_property_map = boost::get(VertexName(), *this);

  boost::put(vertex_property_map, local_vertex, vertex_rank);

  return local_vertex;
}


void
Subgraph::communicate_and_create_shared_entities(Entity   & node,
    EntityKey   new_node_key){

  stk::CommAll comm(bulk_data_->parallel());

  {
    stk::mesh::PairIterEntityComm entity_comm = node.sharing();

    for (; entity_comm.first != entity_comm.second; ++entity_comm.first) {

      unsigned proc = entity_comm.first->proc;
      comm.send_buffer(proc).pack<EntityKey>(node.key())
                                  .pack<EntityKey>(new_node_key);

    }
  }

  comm.allocate_buffers(bulk_data_->parallel_size()/4 );

  {
    stk::mesh::PairIterEntityComm entity_comm = node.sharing();

    for (; entity_comm.first != entity_comm.second; ++entity_comm.first) {

      unsigned proc = entity_comm.first->proc;
      comm.send_buffer(proc).pack<EntityKey>(node.key())
                                  .pack<EntityKey>(new_node_key);

    }
  }

  comm.communicate();

  const stk::mesh::PartVector no_parts;

  for (size_t process = 0; process < bulk_data_->parallel_size(); ++process) {
    EntityKey old_key;
    EntityKey new_key;

    while ( comm.recv_buffer(process).remaining()) {

      comm.recv_buffer(process).unpack<EntityKey>(old_key)
                                     .unpack<EntityKey>(new_key);

      Entity * new_entity = & bulk_data_->declare_entity(new_key.rank(), new_key.id(), no_parts);
      //std::cout << " Proc: " << bulk_data_->parallel_rank() << " created entity: (" << new_entity->identifier() << ", " <<
      //new_entity->entity_rank() << ")." << '\n';

    }
  }

}

void
Subgraph::bcast_key(unsigned root, EntityKey&   node_key){

  stk::CommBroadcast comm(bulk_data_->parallel(), root);

  unsigned rank = bulk_data_->parallel_rank();

  if(rank == root)

    comm.send_buffer().pack<EntityKey>(node_key);

  comm.allocate_buffer();

  if(rank == root)

    comm.send_buffer().pack<EntityKey>(node_key);

  comm.communicate();

  comm.recv_buffer().unpack<EntityKey>(node_key);

}

Vertex
Subgraph::cloneVertex(Vertex & vertex)
{

  // Get the vertex rank
  EntityRank vertex_rank = Subgraph::getVertexRank(vertex);
  EntityKey vertex_key = Subgraph::localToGlobal(vertex);

  // Determine which processor should create the new vertex
  Entity *  old_vertex = bulk_data_->get_entity(vertex_key);

  //    if(!oldVertex){
  //std::cout << "oldVertex is NULL at line " << __LINE__ << " in file " << __FILE__ << '\n';
  //    }

  // For now, the owner of the new vertex is the same as the owner of the old one
  int owner_proc = old_vertex->owner_rank();

  // The owning processor inserts a new vertex into the stk mesh
  // First have to request a new entity of rank N
  std::vector<size_t> requests(dimension_ + 1, 0); // number of entity ranks. 1 + number of dimensions
  EntityVector new_entity;
  const stk::mesh::PartVector no_parts;

  int my_proc = bulk_data_->parallel_rank();
  int source;
  Entity *global_vertex;
  EntityKey global_vertex_key;
  EntityKey::raw_key_type gvertkey;

  if(my_proc == owner_proc){

    // Insert the vertex into the stk mesh
    // First have to request a new entity of rank N
    requests[vertex_rank] = 1;

    // have stk build the new entity, then broadcast the key

    bulk_data_->generate_new_entities(requests, new_entity);
    global_vertex = new_entity[0];
    //std::cout << " Proc: " << bulk_data_->parallel_rank() << " created entity: (" << global_vertex->identifier() << ", " <<
    //global_vertex->entity_rank() << ")." << '\n';
    global_vertex_key = global_vertex->key();
    gvertkey = global_vertex_key.raw_key();

  }
  else {

    // All other processors do a no-op

    bulk_data_->generate_new_entities(requests, new_entity);

  }

  Subgraph::bcast_key(owner_proc, global_vertex_key);

  if(my_proc != owner_proc){ // All other processors receive the key

    // Get the vertex from stk

    const stk::mesh::PartVector no_parts;
    Entity * new_entity = & bulk_data_->declare_entity(global_vertex_key.rank(), global_vertex_key.id(), no_parts);

  }

  // Insert the vertex into the subgraph
  Vertex local_vertex = boost::add_vertex(*this);

  // Update maps
  local_global_vertex_map_.insert(
      std::map<Vertex, EntityKey>::value_type(local_vertex,
          global_vertex_key));
  global_local_vertex_map_.insert(
      std::map<EntityKey, Vertex>::value_type(global_vertex_key,
          local_vertex));

  // store entity rank to the vertex property
  VertexNamePropertyMap vertex_property_map = boost::get(VertexName(), *this);
  boost::put(vertex_property_map, local_vertex, vertex_rank);

  return local_vertex;
}

//----------------------------------------------------------------------------
//
// Remove vertex in subgraph
//
void Subgraph::removeVertex(Vertex vertex)
{
  // get the global entity key of vertex
  EntityKey key = localToGlobal(vertex);

  // look up entity from key
  Entity* entity = bulk_data_->get_entity(key);

  // remove the vertex and key from global_local_vertex_map_ and
  // local_global_vertex_map_
  global_local_vertex_map_.erase(key);
  local_global_vertex_map_.erase(vertex);

  // remove vertex from subgraph
  // first have to ensure that there are no edges in or out of the vertex
  boost::clear_vertex(vertex, *this);
  // remove the vertex
  boost::remove_vertex(vertex, *this);

  // destroy all relations to or from the entity
  PairIterRelation relations = entity->relations();
  for (int i = 0; i < relations.size(); ++i) {
    EdgeId edgeId = relations[i].identifier();

    Entity & target = *(relations[i].entity());

    bulk_data_->destroy_relation(*entity, target, edgeId);
  }
  // remove the entity from stk mesh
  bool deleted = bulk_data_->destroy_entity(entity);
  assert(deleted);

  return;
}

//----------------------------------------------------------------------------
//
// Add edge to local graph.
//
std::pair<Edge, bool>
Subgraph::addEdge(const EdgeId edge_id,
    const Vertex local_source_vertex,
    const Vertex local_target_vertex)
{
  // Add edge to local graph
  std::pair<Edge, bool> local_edge =
      boost::add_edge(local_source_vertex, local_target_vertex, *this);

  if (local_edge.second == false) return local_edge;

  // get global entities
  EntityKey global_source_key = localToGlobal(local_source_vertex);
  EntityKey global_target_key = localToGlobal(local_target_vertex);
  Entity* global_source_vertex = bulk_data_->get_entity(global_source_key);
  Entity* global_target_vertex = bulk_data_->get_entity(global_target_key);

  //testing
  if (global_source_vertex->entity_rank() - global_target_vertex->entity_rank()
      != 1) {
    std::cout << "add edge:" << global_source_vertex->entity_rank() << ","
        << global_source_vertex->identifier() << " "
        << global_target_vertex->entity_rank() << ","
        << global_target_vertex->identifier() << "\n";
  }

  // Add edge to stk mesh
  bulk_data_->declare_relation(*(global_source_vertex),
      *(global_target_vertex),
      edge_id);

  // Add edge id to edge property
  EdgeNamePropertyMap edge_property_map = boost::get(EdgeName(), *this);
  boost::put(edge_property_map, local_edge.first, edge_id);

  return local_edge;
}

//----------------------------------------------------------------------------
void Subgraph::removeEdge(const Vertex & local_source_vertex,
    const Vertex & local_target_vertex)
{
  // Get the local id of the edge in the subgraph

  Edge edge;
  bool inserted;
  boost::tie(edge, inserted) =
      boost::edge(local_source_vertex,
          local_target_vertex, *this);

  assert(inserted);

  EdgeId edge_id = getEdgeId(edge);

  // remove local edge
  boost::remove_edge(local_source_vertex, local_target_vertex, *this);

  // remove relation from stk mesh
  EntityKey global_source_id = Subgraph::localToGlobal(local_source_vertex);
  EntityKey global_target_id = Subgraph::localToGlobal(local_target_vertex);
  Entity* global_source_vertex = bulk_data_->get_entity(global_source_id);
  Entity* global_target_vertex = bulk_data_->get_entity(global_target_id);

  bulk_data_->destroy_relation(*(global_source_vertex),
      *(global_target_vertex),
      edge_id);

  return;
}

//----------------------------------------------------------------------------
EntityRank &
Subgraph::getVertexRank(const Vertex vertex)
{
  VertexNamePropertyMap vertexPropertyMap = boost::get(VertexName(), *this);
  return boost::get(vertexPropertyMap, vertex);
}

//----------------------------------------------------------------------------
EdgeId &
Subgraph::getEdgeId(const Edge edge)
{
  EdgeNamePropertyMap edgePropertyMap = boost::get(EdgeName(), *this);
  return boost::get(edgePropertyMap, edge);
}

//----------------------------------------------------------------------------
//
// Function determines whether the input vertex is an articulation
// point of the subgraph.
//
void
Subgraph::testArticulationPoint(Vertex input_vertex, int & num_components,
    std::map<Vertex, int> & subComponent)
{
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> undirectedGraph;
  typedef boost::graph_traits<undirectedGraph>::vertex_descriptor uVertex;
  typedef boost::graph_traits<undirectedGraph>::edge_descriptor uEdge;
  // Map to and from undirected graph and subgraph
  std::map<uVertex, Vertex> undirectedSubVertexMap;
  std::map<Vertex, uVertex> subUndirectedVertexMap;
  undirectedGraph g;
  VertexIterator vertex_begin;
  VertexIterator vertex_end;
  boost::tie(vertex_begin, vertex_end) = vertices(*this);

  // First add the vertices to the graph
  for (VertexIterator i = vertex_begin; i != vertex_end; ++i) {
    Vertex vertex = *i;
    if (vertex != input_vertex) {
      uVertex uvertex = boost::add_vertex(g);
      undirectedSubVertexMap[uvertex] = vertex;
      // Add to maps
      undirectedSubVertexMap.insert(
          std::map<uVertex, Vertex>::value_type(uvertex, vertex));
      subUndirectedVertexMap.insert(
          std::map<Vertex, uVertex>::value_type(vertex, uvertex));
    }
  }

  // Then add the edges
  for (VertexIterator i = vertex_begin; i != vertex_end; ++i) {
    Vertex source = *i;

    if (source != input_vertex) {
      std::map<Vertex, uVertex>::const_iterator sourceMapIterator =
          subUndirectedVertexMap.find(source);

      uVertex usource = (*sourceMapIterator).second;

      // write the edges in the subgraph
      OutEdgeIterator out_edge_begin;
      OutEdgeIterator out_edge_end;
      boost::tie(out_edge_begin, out_edge_end) = out_edges(*i, *this);

      for (OutEdgeIterator j = out_edge_begin; j != out_edge_end; ++j) {
        Vertex target;
        Edge edge = *j;
        target = boost::target(edge, *this);

        if (target != input_vertex) {
          std::map<Vertex, uVertex>::const_iterator targetMapIterator =
              subUndirectedVertexMap.find(target);

          uVertex utarget = (*targetMapIterator).second;

          boost::add_edge(usource, utarget, g);
        }
      }
    }
  }

  std::vector<int> component(boost::num_vertices(g));
  num_components = boost::connected_components(g, &component[0]);

  for (std::map<uVertex, Vertex>::iterator i = undirectedSubVertexMap.begin();
      i != undirectedSubVertexMap.end(); ++i) {
    Vertex vertex = (*i).second;
    subComponent.insert(
        std::map<Vertex, int>::value_type(vertex, component[(*i).first]));
  }

  return;
}

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
Subgraph::cloneBoundaryEntity(Vertex & vertex, Vertex & new_vertex,
    std::map<EntityKey, bool> & entity_open)
{
  // Check that number of in_edges = 2
  boost::graph_traits<Graph>::degree_size_type num_in_edges =
      boost::in_degree(vertex, *this);
  if (num_in_edges != 2) return;

  // Check that vertex = open
  EntityKey vertex_key = Subgraph::localToGlobal(vertex);
  assert(entity_open[vertex_key]==true);

  // Get the vertex rank
  //    EntityRank vertex_rank = Subgraph::getVertexRank(vertex);

  // Create a new vertex of same rank as vertex
  //    newVertex = Subgraph::add_vertex(vertex_rank);
  new_vertex = Subgraph::cloneVertex(vertex);

  // Copy the out_edges of vertex to new_vertex
  OutEdgeIterator out_edge_begin;
  OutEdgeIterator out_edge_end;
  boost::tie(out_edge_begin, out_edge_end) = boost::out_edges(vertex, *this);
  for (OutEdgeIterator i = out_edge_begin; i != out_edge_end; ++i) {
    Edge edge = *i;
    EdgeId edgeId = Subgraph::getEdgeId(edge);
    Vertex target = boost::target(edge, *this);
    Subgraph::addEdge(edgeId, new_vertex, target);
  }

  // Copy all out edges not in the subgraph to the new vertex
  Subgraph::cloneOutEdges(vertex, new_vertex);

  // Remove one of the edges from vertex, copy to new_vertex
  // Arbitrarily remove the first edge from original vertex
  InEdgeIterator in_edge_begin;
  InEdgeIterator in_edge_end;
  boost::tie(in_edge_begin, in_edge_end) = boost::in_edges(vertex, *this);
  Edge edge = *(in_edge_begin);
  EdgeId edgeId = Subgraph::getEdgeId(edge);
  Vertex source = boost::source(edge, *this);
  Subgraph::removeEdge(source, vertex);

  // Add edge to new vertex
  Subgraph::addEdge(edgeId, source, new_vertex);

  // Have to clone the out edges of the original entity to the new entity.
  // These edges are not in the subgraph

  // Clone process complete, set entity_open to false
  entity_open[vertex_key] = false;

  return;
}

//----------------------------------------------------------------------------
//
// Splits an articulation point.
//
std::map<Entity*, Entity*>
Subgraph::splitArticulationPoint(Vertex vertex,
    std::map<EntityKey, bool> & entity_open)
{
  // Check that vertex = open
  EntityKey vertex_key = Subgraph::localToGlobal(vertex);
  assert(entity_open[vertex_key]==true);

  // get rank of vertex
  EntityRank vertex_rank = Subgraph::getVertexRank(vertex);

  // Create undirected graph
  int num_components;
  std::map<Vertex, int> components;
  Subgraph::testArticulationPoint(vertex, num_components, components);

  // The function returns an updated connectivity map. If the vertex
  //   rank is not node, then this map will be of size 0.
  std::map<Entity*, Entity*> new_connectivity;

  // Check number of connected components in undirected graph. If =
  // 1, return
  if (num_components == 1) return new_connectivity;

  // If number of connected components > 1, split vertex in subgraph and stk mesh
  // number of new vertices = numComponents - 1
  std::vector<Vertex> new_vertex;
  for (int i = 0; i < num_components - 1; ++i) {
    //      Vertex newVert = Subgraph::add_vertex(vertex_rank);
    Vertex new_vert = Subgraph::cloneVertex(vertex);
    new_vertex.push_back(new_vert);
  }

  // create a map of elements to new node numbers
  // only do this if the input vertex is a node (don't require otherwise)
  if (vertex_rank == 0) {
    for (std::map<Vertex, int>::iterator i = components.begin();
        i != components.end(); ++i) {
      int component_num = (*i).second;
      Vertex current_vertex = (*i).first;
      EntityRank current_rank = Subgraph::getVertexRank(current_vertex);
      // Only add to map if the vertex is an element
      if (current_rank == dimension_ && component_num != 0) {
        Entity* element =
            bulk_data_->get_entity(Subgraph::localToGlobal(current_vertex));
        Entity* new_node =
            bulk_data_->
            get_entity(Subgraph::localToGlobal(new_vertex[component_num - 1]));
        new_connectivity.
        insert(std::map<Entity*, Entity*>::value_type(element, new_node));
      }
    }
  }

  // Copy the out edges of the original vertex to the new vertex
  for (int i = 0; i < new_vertex.size(); ++i) {
    Subgraph::cloneOutEdges(vertex, new_vertex[i]);
  }

  // vector for edges to be removed. Vertex is source and edgeId the
  // local id of the edge
  std::vector<std::pair<Vertex, EdgeId> > removed;

  // Iterate over the in edges of the vertex to determine which will
  // be removed
  InEdgeIterator in_edge_begin;
  InEdgeIterator in_edge_end;
  boost::tie(in_edge_begin, in_edge_end) = boost::in_edges(vertex, *this);
  for (InEdgeIterator i = in_edge_begin; i != in_edge_end; ++i) {
    Edge edge = *i;
    Vertex source = boost::source(edge, *this);

    std::map<Vertex, int>::const_iterator componentIterator =
        components.find(source);
    int vertComponent = (*componentIterator).second;
    Entity& entity =
        *(bulk_data_->get_entity(Subgraph::localToGlobal(source)));
    // Only replace edge if vertex not in component 0
    if (vertComponent != 0) {
      EdgeId edgeId = Subgraph::getEdgeId(edge);
      removed.push_back(std::make_pair(source, edgeId));
    }
  }

  // remove all edges in vector removed and replace with new edges
  for (std::vector<std::pair<Vertex, EdgeId> >::iterator i = removed.begin();
      i != removed.end(); ++i) {
    std::pair<Vertex, EdgeId> edge = *i;
    Vertex source = edge.first;
    EdgeId edgeId = edge.second;
    std::map<Vertex, int>::const_iterator componentIterator =
        components.find(source);
    int vertComponent = (*componentIterator).second;

    Subgraph::removeEdge(source, vertex);
    std::pair<Edge, bool> inserted =
        Subgraph::addEdge(edgeId, source,new_vertex[vertComponent - 1]);
    assert(inserted.second==true);
  }

  // split process complete, set entity_open to false
  entity_open[vertex_key] = false;

  return new_connectivity;
}

//----------------------------------------------------------------------------
//
// Clone all out edges of a vertex to a new vertex.
//
void Subgraph::cloneOutEdges(Vertex & original_vertex, Vertex & new_vertex)
{
  // Get the entity for the original and new vertices
  EntityKey original_key = Subgraph::localToGlobal(original_vertex);
  EntityKey new_key = Subgraph::localToGlobal(new_vertex);
  Entity & original_entity = *(bulk_data_->get_entity(original_key));
  Entity & new_entity = *(bulk_data_->get_entity(new_key));

  // Iterate over the out edges of the original vertex and check against the
  //   out edges of the new vertex. If the edge does not exist, add.
  PairIterRelation original_relations =
      original_entity.relations(original_entity.entity_rank() - 1);
  for (int i = 0; i < original_relations.size(); ++i) {
    PairIterRelation new_relations =
        new_entity.relations(new_entity.entity_rank() - 1);
    // assume the edge doesn't exist
    bool exists = false;
    for (int j = 0; j < new_relations.size(); ++j) {
      if (original_relations[i].entity() == new_relations[j].entity()) {
        exists = true;
      }
    }
    if (exists == false) {
      EdgeId edgeId = original_relations[i].identifier();
      Entity& target = *(original_relations[i].entity());
      bulk_data_->declare_relation(new_entity, target, edgeId);
    }
  }

  return;
}

/**
 * \brief Output the graph associated with the mesh to graphviz .dot
 * file for visualization purposes.
 *
 * \param[in] output file
 * \param[in] map of entity and boolean value is open
 *
 * Similar to outputToGraphviz function in Topology class.
 * If fracture criterion for entity is satisfied, the entity and all
 * associated lower order entities are marked open. All open entities are
 * displayed as such in output file.
 *
 * To create final output figure, run command below from terminal:
 *   dot -Tpng <gviz_output>.dot -o <gviz_output>.png
 */
void Subgraph::outputToGraphviz(std::string & gviz_output,
    std::map<EntityKey, bool> entity_open)
{
  // Open output file
  std::ofstream gviz_out;
  gviz_out.open(gviz_output.c_str(), std::ios::out);

  std::cout << "Write graph to graphviz dot file\n";

  if (gviz_out.is_open()) {
    // Write beginning of file
    gviz_out << "digraph mesh {\n" << "  node [colorscheme=paired12]\n"
        << "  edge [colorscheme=paired12]\n";

    VertexIterator vertex_begin;
    VertexIterator vertex_end;
    boost::tie(vertex_begin, vertex_end) = vertices(*this);

    for (VertexIterator i = vertex_begin; i != vertex_end; ++i) {
      EntityKey key = localToGlobal(*i);
      Entity & entity = *(bulk_data_->get_entity(key));
      std::string label;
      std::string color;

      // Write the entity name
      switch (entity.entity_rank()) {
      // nodes
      case 0:
        label = "Node";
        if (entity_open[entity.key()] == false)
          color = "6";
        else
          color = "5";
        break;
        // segments
      case 1:
        label = "Segment";
        if (entity_open[entity.key()] == false)
          color = "4";
        else
          color = "3";
        break;
        // faces
      case 2:
        label = "Face";
        if (entity_open[entity.key()] == false)
          color = "2";
        else
          color = "1";
        break;
        // volumes
      case 3:
        label = "Element";
        if (entity_open[entity.key()] == false)
          color = "8";
        else
          color = "7";
        break;
      }
      gviz_out << "  \"" << entity.identifier() << "_" << entity.entity_rank()
                     << "\" [label=\" " << label << " " << entity.identifier()
                     << "\",style=filled,fillcolor=\"" << color << "\"]\n";

      // write the edges in the subgraph
      OutEdgeIterator out_edge_begin;
      OutEdgeIterator out_edge_end;
      boost::tie(out_edge_begin, out_edge_end) = out_edges(*i, *this);

      for (OutEdgeIterator j = out_edge_begin; j != out_edge_end; ++j) {
        Edge out_edge = *j;
        Vertex source = boost::source(out_edge, *this);
        Vertex target = boost::target(out_edge, *this);

        EntityKey sourceKey = localToGlobal(source);
        Entity & global_source = *(bulk_data_->get_entity(sourceKey));

        EntityKey targetKey = localToGlobal(target);
        Entity & global_target = *(bulk_data_->get_entity(targetKey));

        EdgeId edgeId = getEdgeId(out_edge);

        switch (edgeId) {
        case 0:
          color = "6";
          break;
        case 1:
          color = "4";
          break;
        case 2:
          color = "2";
          break;
        case 3:
          color = "8";
          break;
        case 4:
          color = "10";
          break;
        case 5:
          color = "12";
          break;
        default:
          color = "9";
        }
        gviz_out << "  \"" << global_source.identifier() << "_"
            << global_source.entity_rank() << "\" -> \""
            << global_target.identifier() << "_"
            << global_target.entity_rank() << "\" [color=\"" << color << "\"]"
            << "\n";
      }

    }

    // File end
    gviz_out << "}";
    gviz_out.close();
  } else
    std::cout << "Unable to open graphviz output file 'output.dot'\n";

  return;
}

} // namespace LCM
