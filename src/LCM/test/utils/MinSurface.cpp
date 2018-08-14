//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include "topology/Topology.h"

using namespace boost;

int
main(int ac, char* av[])
{
  typedef adjacency_list<
      listS,
      vecS,
      undirectedS,
      no_property,
      property<edge_weight_t, int>>
                                                   graph_t;
  typedef graph_traits<graph_t>::vertex_descriptor vertex_descriptor;
  typedef stk::mesh::Entity                        Entity;
  typedef std::pair<int, int>                      Edge;

  //---------------------------------------------------------------------------------------------------------
  //
  // Create a command line processor and parse command line options
  //
  Teuchos::CommandLineProcessor command_line_processor;

  command_line_processor.setDocString(
      "Test of barycentric subdivision.\n"
      "Reads in a mesh and applies the barycentric subdivision algorithm.\n"
      "Restricted to simplicial complexes.\n");

  std::string input_file = "input.e";
  command_line_processor.setOption("input", &input_file, "Input File Name");

  std::string output_file = "output.e";
  command_line_processor.setOption("output", &output_file, "Output File Name");

  // Throw a warning and not error for unrecognized options
  command_line_processor.recogniseAllOptions(true);

  // Don't throw exceptions for errors
  command_line_processor.throwExceptions(false);

  // Parse command line
  Teuchos::CommandLineProcessor::EParseCommandLineReturn parse_return =
      command_line_processor.parse(ac, av);

  if (parse_return == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
    return 0;
  }

  if (parse_return != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return 1;
  }

  //
  // Read the mesh
  //
  // Copied from Partition.cc
  Teuchos::GlobalMPISession mpiSession(&ac, &av);
  LCM::Topology             topology(input_file, output_file);

  //-----------------------------------------------------------------------------------------
  // GET THE 1D BUNDARY FROM THE INPUT MESH USING dijkstra_shortest_paths
  //-----------------------------------------------------------------------------------------
  stk::mesh::BulkData&           bulk_data = topology.get_bulk_data();
  std::vector<stk::mesh::Entity> MeshNodes =
      topology.getEntitiesByRank(bulk_data, 0);

  // Definition of parameters and arrays of the function Graph
  const int TotalNumberNodes =
      MeshNodes.size();  // Total number of nodes of the input the mesh
  std::vector<int> _nodeNames = topology.nodeNames();  // Vector with node names

  // Define edges and weights
  std::vector<stk::mesh::Entity> MeshEdges = topology.getEntitiesByRank(
      bulk_data,
      1);  // Get all the edges of the mesh

  // Initialize Array of edges
  const int ArraySize = MeshEdges.size();
  Edge*     EdgesArray;
  EdgesArray = new Edge[ArraySize];

  // Create an array that holds the distances (or weights) of each of the edges
  double* EdgesWeights;
  EdgesWeights = new double[ArraySize];

  for (unsigned int i = 0; i < MeshEdges.size(); ++i) {
    std::vector<stk::mesh::Entity> EdgeBoundaryNodes(2);
    EdgeBoundaryNodes =
        topology.getDirectlyConnectedEntities((*MeshEdges[i]), 0);
    EdgesArray[i] = Edge(
        (EdgeBoundaryNodes[0]->identifier()) - 1,
        (EdgeBoundaryNodes[1]->identifier()) - 1);
    // Adds the distance of the ith edge to the vector of edge distances
    EdgesWeights[i] = (topology.getDistanceBetweenNodes(
        EdgeBoundaryNodes[0], EdgeBoundaryNodes[1]));
  }

  // Definition of the graph
  graph_t Graph(
      EdgesArray, EdgesArray + ArraySize, EdgesWeights, TotalNumberNodes);

  std::vector<vertex_descriptor> predecessor(num_vertices(
      Graph));  // The vertex descriptor associates a single vertex in the graph
  std::vector<int>               dis(num_vertices(Graph));
  vertex_descriptor              source = vertex(
      3, Graph);  // Source from where the distance and path are calculated
  dijkstra_shortest_paths(
      Graph,
      source,
      predecessor_map(&predecessor[0])
          .distance_map(&dis[0]));  // Compute the shortest path

  delete[] EdgesArray;    // Deallocate memory
  delete[] EdgesWeights;  // Deallocate memory

  //-------------------------------------------------------------------------------
  // Presenting the final output
  //-------------------------------------------------------------------------------
  std::cout << "Display distances to all graph vertices from source"
            << std::endl;
  graph_traits<graph_t>::vertex_iterator VertexIterator, vend;
  for (boost::tie(VertexIterator, vend) = vertices(Graph);
       VertexIterator != vend;
       ++VertexIterator) {
    std::cout << "distance(" << _nodeNames[*VertexIterator]
              << ") = " << dis[*VertexIterator] << std::endl;
  }
  std::cout << std::endl;
  //"predecessor" is the predecessor map obtained from dijkstra_shortest_paths
  std::vector<graph_traits<graph_t>::vertex_descriptor> ShortestPath;

  // When changing the sourceVertex, MAKE SURE TO CHANGE "source" above.
  // Otherwise, an run time error will be displayed
  vertex_descriptor sourceVertex =
      vertex(3, Graph);  // Source from where the distance and path are
                         // calculated (Same as above!)
  vertex_descriptor goalVertex = vertex(4, Graph);

  graph_traits<graph_t>::vertex_descriptor currentVertex = goalVertex;
  while (currentVertex != sourceVertex) {
    ShortestPath.push_back(currentVertex);
    currentVertex = predecessor[currentVertex];
  }
  // ShortestPath contains the shortest path between the given vertices.
  // Vertices are saved starting from end vertex to start vertex. The format of
  // this output is "unsigned long int"
  ShortestPath.push_back(sourceVertex);

  // Change the format of the output from unsigned long int
  std::vector<int>                               FV;
  std::vector<unsigned long int>::const_iterator I_points;
  for (I_points = ShortestPath.begin(); I_points != ShortestPath.end();
       ++I_points) {
    int i = boost::numeric_cast<int>(*I_points);
    FV.push_back(i);
  }

  // This prints the path reversed use reverse_iterator
  std::vector<graph_traits<graph_t>::vertex_descriptor>::iterator Iterator_;
  std::cout << "From end node to start node" << std::endl;
  for (Iterator_ = ShortestPath.begin(); Iterator_ != ShortestPath.end();
       ++Iterator_) {
    std::cout << _nodeNames[*Iterator_] << " ";
  }
  std::cout << std::endl;

  return 0;
}
