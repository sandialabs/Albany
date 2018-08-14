//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_config.h"

// Define only if ALbany is enabled
#if defined(ALBANY_LCM)

#include <stk_mesh/base/FieldBase.hpp>
#include "Topology.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/properties.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/property_map/property_map.hpp>
// typedef std::pair<int, int> Edge;

namespace LCM {

//
// \brief Finds the closest nodes(Entities of rank 0) to each of the
// three points in the input vector
//
std::vector<stk::mesh::Entity>
Topology::getClosestNodes(std::vector<std::vector<double>> points)
{
  std::vector<stk::mesh::Entity> closestNodes;
  stk::mesh::Entity              nodeA;
  stk::mesh::Entity              nodeB;
  stk::mesh::Entity              nodeC;
  std::vector<double>            pointA, pointB, pointC;
  double                         minDA, minDB, minDC;

  std::vector<stk::mesh::Entity> entities_D0 = getEntitiesByRank(
      get_bulk_data(),
      stk::topology::NODE_RANK);  // get all the nodes

  // iterator for the nodes
  std::vector<stk::mesh::Entity>::const_iterator i_entities_d0;

  // Before iterate, it is necessary to have a distance with which it
  // is possible to compare the new distances to.
  nodeA = entities_D0[0];
  minDA = getDistanceNodeAndPoint(entities_D0[0], points[0]);

  nodeB = entities_D0[0];
  minDB = getDistanceNodeAndPoint(entities_D0[0], points[1]);

  nodeC = entities_D0[0];
  minDC = getDistanceNodeAndPoint(entities_D0[0], points[2]);

  // For each of the nodes calculate distance from point1, point 2,
  // point 3 if any distance is less than the min distance to that
  // point update the min distance and the closest node for that point
  for (i_entities_d0 = entities_D0.begin(); i_entities_d0 != entities_D0.end();
       ++i_entities_d0) {
    // adist is the distance between the current node and the first
    // point, point A.
    double aDist = getDistanceNodeAndPoint(*i_entities_d0, points[0]);
    if (aDist < minDA) {
      nodeA = *i_entities_d0;
      minDA = aDist;
    }

    double bDist = getDistanceNodeAndPoint(*i_entities_d0, points[1]);
    if (bDist < minDB) {
      nodeB = *i_entities_d0;
      minDB = bDist;
    }

    double cDist = getDistanceNodeAndPoint(*i_entities_d0, points[2]);
    if (cDist < minDC) {
      nodeC = *i_entities_d0;
      minDC = cDist;
    }
  }
  closestNodes.push_back(nodeA);
  closestNodes.push_back(nodeB);
  closestNodes.push_back(nodeC);
  return closestNodes;
}

//
// \brief Finds the closest nodes(Entities of rank 0) to each
//        of the three points in the input vectorThese nodes
//        lie over the surface of the mesh
//
std::vector<stk::mesh::Entity>
Topology::getClosestNodesOnSurface(std::vector<std::vector<double>> points)
{
  // Obtain all the nodes that lie over the surface
  // Obtain all the faces of the mesh
  std::vector<stk::mesh::Entity> MeshFaces =
      getEntitiesByRank(get_bulk_data(), stk::topology::FACE_RANK);

  // Find the faces (Entities of rank 2) that build the boundary of
  // the given mesh
  std::vector<stk::mesh::Entity>                 BoundaryFaces;
  std::vector<stk::mesh::Entity>::const_iterator I_faces;
  for (I_faces = MeshFaces.begin(); I_faces != MeshFaces.end(); I_faces++) {
    std::vector<stk::mesh::Entity> temp;
    temp = getDirectlyConnectedEntities(*I_faces, stk::topology::ELEMENT_RANK);
    // If the number of boundary entities of rank 3 is 1 then, this is
    // a boundary face
    if (temp.size() == 1) { BoundaryFaces.push_back(*I_faces); }
  }

  // Obtain the Edges that belong to the Boundary Faces delete the
  // repeated edges
  std::vector<stk::mesh::Entity>                 MeshEdges;
  std::vector<stk::mesh::Entity>::const_iterator I_BoundaryFaces;
  std::vector<stk::mesh::Entity>::const_iterator I_Edges;
  for (I_BoundaryFaces = BoundaryFaces.begin();
       I_BoundaryFaces != BoundaryFaces.end();
       I_BoundaryFaces++) {
    std::vector<stk::mesh::Entity> boundaryEdges = getDirectlyConnectedEntities(
        *I_BoundaryFaces, stk::topology::EDGE_RANK);
    for (I_Edges = boundaryEdges.begin(); I_Edges != boundaryEdges.end();
         I_Edges++) {
      if (findEntityInVector(MeshEdges, *I_Edges) == false) {
        MeshEdges.push_back(*I_Edges);
      }
    }
  }

  // Obtain the nodes that lie on the surface
  // This vector contains all the nodes that lie on the surface
  std::vector<stk::mesh::Entity> entities_D0;
  for (unsigned int i = 0; i < MeshEdges.size(); ++i) {
    std::vector<stk::mesh::Entity> EdgeBoundaryNodes;
    EdgeBoundaryNodes =
        getDirectlyConnectedEntities(MeshEdges[i], stk::topology::NODE_RANK);
    for (unsigned int i = 0; i < EdgeBoundaryNodes.size(); i++) {
      if (findEntityInVector(entities_D0, EdgeBoundaryNodes[i]) == false) {
        entities_D0.push_back(EdgeBoundaryNodes[i]);
      }
    }
  }

  std::vector<stk::mesh::Entity> closestNodes;
  stk::mesh::Entity              nodeA;
  stk::mesh::Entity              nodeB;
  stk::mesh::Entity              nodeC;
  std::vector<double>            pointA, pointB, pointC;
  double                         minDA, minDB, minDC;

  // iterator for the nodes
  std::vector<stk::mesh::Entity>::const_iterator i_entities_d0;

  // Before iterate, it is necessary to have a distance with which it
  // is possible to compare the new distances to.
  nodeA = entities_D0[0];
  minDA = getDistanceNodeAndPoint(entities_D0[0], points[0]);

  nodeB = entities_D0[0];
  minDB = getDistanceNodeAndPoint(entities_D0[0], points[1]);

  nodeC = entities_D0[0];
  minDC = getDistanceNodeAndPoint(entities_D0[0], points[2]);

  // For each of the nodes
  // calculate distance from point1, point 2, point 3
  // if any distance is less than the min distance to that point
  // update the min distance and the closest node for that point
  for (i_entities_d0 = entities_D0.begin(); i_entities_d0 != entities_D0.end();
       ++i_entities_d0) {
    // adist is the distance between the current node and the first
    // point, point A.
    double aDist = getDistanceNodeAndPoint(*i_entities_d0, points[0]);
    if (aDist < minDA) {
      nodeA = *i_entities_d0;
      minDA = aDist;
    }

    double bDist = getDistanceNodeAndPoint(*i_entities_d0, points[1]);
    if (bDist < minDB) {
      nodeB = *i_entities_d0;
      minDB = bDist;
    }

    double cDist = getDistanceNodeAndPoint(*i_entities_d0, points[2]);
    if (cDist < minDC) {
      nodeC = *i_entities_d0;
      minDC = cDist;
    }
  }
  closestNodes.push_back(nodeA);
  closestNodes.push_back(nodeB);
  closestNodes.push_back(nodeC);

  return closestNodes;
}

//
// \brief calculates the distance between a node and a point
//
double
Topology::getDistanceNodeAndPoint(
    stk::mesh::Entity   node,
    std::vector<double> point)
{
  // Declare x, y, and z coordinates of the node
  double* entity_coordinates_xyz = getEntityCoordinates(node);
  double  x_Node                 = entity_coordinates_xyz[0];
  double  y_Node                 = entity_coordinates_xyz[1];
  double  z_Node                 = entity_coordinates_xyz[2];

  // Declare x,y,and z coordinates of the point
  double x_point = point[0];
  double y_point = point[1];
  double z_point = point[2];

  // Calculate the distance between point and node in the x, y, and z
  // directions
  double x_dist = x_point - x_Node;
  double y_dist = y_point - y_Node;
  double z_dist = z_point - z_Node;

  // Compute the distance between the point and the node
  // (stk::mesh::Entity of rank 0)
  return sqrt(x_dist * x_dist + y_dist * y_dist + z_dist * z_dist);
}

//
// \brief Returns the coordinates of the points that form a
//  equilateral triangle.  This triangle lies on the plane that
//  intersects the ellipsoid.
//
std::vector<std::vector<double>>
Topology::getCoordinatesOfTriangle(const std::vector<double> normalToPlane)
{
  // vectorA, vectorB, and vectorC are vectors of magnitude R (radius
  // of the circle).  vectors B1, B2, and C1 are component unit vectors
  // used to find B and C xB1, xB, xA, yA, etc. are all eventually
  // components of unit vectors.

  // Compute the coordinates of the resulting circle.  This circle
  // results from the intersection of the plane and the ellipsoid
  std::vector<double> CoordOfMaxAndMin = getCoordinatesOfMaxAndMin();
  double              maxX             = CoordOfMaxAndMin[0];
  double              minX             = CoordOfMaxAndMin[1];
  double              maxY             = CoordOfMaxAndMin[2];
  double              minY             = CoordOfMaxAndMin[3];
  double              maxZ             = CoordOfMaxAndMin[4];
  double              minZ             = CoordOfMaxAndMin[5];

  // Find the center of the cube of nodes
  std::vector<double> coordOfCenter;
  double              xCenter = (maxX + minX) / 2.0;
  double              yCenter = (maxY + minY) / 2.0;
  double              zCenter = (maxZ + minZ) / 2.0;
  coordOfCenter.push_back(xCenter);
  coordOfCenter.push_back(yCenter);
  coordOfCenter.push_back(zCenter);

  // Radius of the circle
  double              radius = maxX - coordOfCenter[0];
  std::vector<double> vectorN;

  // Find a perpendicular vector to the input one

  for (int i = 0; i < 3; i++) {
    vectorN.push_back(normalToPlane[i] - coordOfCenter[i]);
  }

  std::vector<double> vectorA;

  // Throw exception if input vector is 0,0,0 CHANGE THIS EXCEPTION
  // SINCE NORMAL CAN COINCIDE WITH COORDINATES OF THE CENTER
  TEUCHOS_TEST_FOR_EXCEPTION(
      vectorN[0] == 0 && vectorN[1] == 0 && vectorN[0] == 0,
      std::logic_error,
      "The input normal vector was 0,0,0 \n");

  double theta_rand = 2 * (3.14159) * randomNumber(0, 1);
  double x;
  double y;
  double z;
  if (vectorN[2] != 0) {
    x = cos(theta_rand);
    y = sin(theta_rand);
    z = -(vectorN[0] * x + vectorN[1] * y) / vectorN[2];
  } else if (vectorN[0] != 0) {
    z = cos(theta_rand);
    y = sin(theta_rand);
    x = -(vectorN[2] * z + vectorN[1] * y) / vectorN[0];
  } else {
    x = cos(theta_rand);
    z = sin(theta_rand);
    y = -(vectorN[0] * x + vectorN[2] * z) / vectorN[1];
  }

  double L  = sqrt(x * x + y * y + z * z);
  double xA = x / L;
  vectorA.push_back(xA * radius + xCenter);
  double yA = y / L;
  vectorA.push_back(yA * radius + yCenter);
  double zA = z / L;
  vectorA.push_back(zA * radius + zCenter);

  // Find a particular unit vector perpendicular to the previous two
  // (normalToPlane X vectorA) declare the vector
  double xB1;
  double yB1;
  double zB1;

  // Compute the coordinates of the vector B1, which is a component of
  // vectorB
  xB1 = (vectorN[1] * zA - vectorN[2] * yA);
  yB1 = (vectorN[2] * xA - vectorN[0] * zA);
  zB1 = (vectorN[0] * yA - vectorN[1] * xA);

  // Normalize vectorB1
  double magB1 = sqrt(xB1 * xB1 + yB1 * yB1 + zB1 * zB1);
  xB1          = xB1 / magB1;
  yB1          = yB1 / magB1;
  zB1          = zB1 / magB1;

  // Rotate vectorB1 30degrees

  // Find first component of rotated unit vectorB1 (120).
  double xB2;
  double yB2;
  double zB2;

  xB2 = -xA * tan(3.14159 / 6);
  yB2 = -yA * tan(3.14159 / 6);
  zB2 = -zA * tan(3.14159 / 6);

  // Add the two components of the 120 degree vectorB (vectorB1 and
  // vectorB2)
  std::vector<double> vectorB;
  double              xB;
  double              yB;
  double              zB;

  xB = xB1 + xB2;
  yB = yB1 + yB2;
  zB = zB1 + zB2;

  // Normalize vector B
  double magB = sqrt(xB * xB + yB * yB + zB * zB);
  xB          = xB / magB;
  yB          = yB / magB;
  zB          = zB / magB;

  vectorB.push_back(xB * radius + xCenter);
  vectorB.push_back(yB * radius + yCenter);
  vectorB.push_back(zB * radius + zCenter);

  // Find the third vector, vectorC, 120 degrees from vectorA and
  // vectorB;
  std::vector<double> vectorC;
  double              xC;
  double              yC;
  double              zC;

  // The first component will be the negative of the first component
  // for B
  double xC1 = -xB1;
  double yC1 = -yB1;
  double zC1 = -zB1;

  // The second component will be the same as the second component for
  // B(xB2, yB2, zB2) The components of C are just the sums of each of
  // the components of its component vectors
  xC = xC1 + xB2;
  yC = yC1 + yB2;
  zC = zC1 + zB2;

  double magC = magB;
  xC          = xC / magC;
  yC          = yC / magC;
  zC          = zC / magC;

  vectorC.push_back(xC * radius + xCenter);
  vectorC.push_back(yC * radius + yCenter);
  vectorC.push_back(zC * radius + zCenter);

  std::vector<std::vector<double>> VectorOfPoints;
  VectorOfPoints.push_back(vectorA);
  VectorOfPoints.push_back(vectorB);
  VectorOfPoints.push_back(vectorC);

  return VectorOfPoints;
}

//
// \brief Return a random number between two given numbers
//
double
Topology::randomNumber(double valMin, double valMax)
{
  double value = (double)rand() / RAND_MAX;
  return valMin + value * (valMax - valMin);
}

//
// \brief Returns the distance between two entities of rank 0 (nodes)
//
double
Topology::getDistanceBetweenNodes(
    stk::mesh::Entity node1,
    stk::mesh::Entity node2)
{
  // Declares the x,y,and z coordinates for the first node
  double* coordinate1 = getEntityCoordinates(node1);
  double  x1          = coordinate1[0];
  double  y1          = coordinate1[1];
  double  z1          = coordinate1[2];

  // Declares the x,y,and z coordinates for the first node
  double* coordinate2 = getEntityCoordinates(node2);
  double  x2          = coordinate2[0];
  double  y2          = coordinate2[1];
  double  z2          = coordinate2[2];

  // Computes the distance between the two nodes
  double distance =
      sqrt(pow((x1 - x2), 2.0) + pow((y1 - y2), 2.0) + pow((z1 - z2), 2.0));
  return distance;
}

//
// \brief Returns the coordinates of the max and min of x y and z in
// the order max of x, min of x, max of y, min of y, max of z, min of
// z
//
std::vector<double>
Topology::getCoordinatesOfMaxAndMin()
{
  std::vector<stk::mesh::Entity> entities_D0 = getEntitiesByRank(
      get_bulk_data(),
      stk::topology::NODE_RANK);  // get all the nodes

  // iterator for the nodes
  std::vector<stk::mesh::Entity>::const_iterator i_entities_d0;

  // Get the coordinates of the first node
  double* entity_coordinates_xyz = getEntityCoordinates(entities_D0[0]);
  double  x_coordinate           = entity_coordinates_xyz[0];
  double  y_coordinate           = entity_coordinates_xyz[1];
  double  z_coordinate           = entity_coordinates_xyz[2];

  // Declare all the variables for the max and min coordinates and set
  // them equal to the values of the first coordinates
  double maxX = x_coordinate;
  double minX = x_coordinate;
  double maxY = y_coordinate;
  double minY = y_coordinate;
  double maxZ = z_coordinate;
  double minZ = z_coordinate;

  // Declare the vector that has the coordinates of the center
  std::vector<double> coordOfMaxAndMin;

  // Iterate through every node
  for (i_entities_d0 = entities_D0.begin(); i_entities_d0 != entities_D0.end();
       ++i_entities_d0) {
    // Get the coordinates of the ith node
    double* entity_coordinates_xyz = getEntityCoordinates(*i_entities_d0);
    double  x_coordinate           = entity_coordinates_xyz[0];
    double  y_coordinate           = entity_coordinates_xyz[1];
    double  z_coordinate           = entity_coordinates_xyz[2];

    // Compare the x,y, and z coordinates to the max and min for x y
    // and z if value is more extreme than max or min update max or
    // min
    if (x_coordinate > maxX) { maxX = x_coordinate; }
    if (y_coordinate > maxY) { maxY = y_coordinate; }
    if (z_coordinate > maxZ) { maxZ = z_coordinate; }

    if (x_coordinate < minX) { minX = x_coordinate; }
    if (y_coordinate < minY) { minY = y_coordinate; }
    if (z_coordinate < minZ) { minZ = z_coordinate; }
  }
  coordOfMaxAndMin.push_back(maxX);
  coordOfMaxAndMin.push_back(minX);
  coordOfMaxAndMin.push_back(maxY);
  coordOfMaxAndMin.push_back(minY);
  coordOfMaxAndMin.push_back(maxZ);
  coordOfMaxAndMin.push_back(minZ);
  return coordOfMaxAndMin;
}

//
// \brief Returns the edges necessary to compute the shortest path on
//        the outer surface of the mesh
//
std::vector<stk::mesh::Entity>
Topology::meshEdgesShortestPath()
{
  // Obtain all the faces of the mesh
  std::vector<stk::mesh::Entity> MeshFaces =
      getEntitiesByRank(get_bulk_data(), stk::topology::FACE_RANK);

  // Find the faces (Entities of rank 2) that build the boundary of the
  // given mesh
  std::vector<stk::mesh::Entity>                 BoundaryFaces;
  std::vector<stk::mesh::Entity>::const_iterator I_faces;
  for (I_faces = MeshFaces.begin(); I_faces != MeshFaces.end(); I_faces++) {
    std::vector<stk::mesh::Entity> temp;
    temp = getDirectlyConnectedEntities(*I_faces, stk::topology::ELEMENT_RANK);
    // If the number of boundary entities of rank 3 is 1
    // then, this is a boundary face
    if (temp.size() == 1) { BoundaryFaces.push_back(*I_faces); }
  }

  // Obtain the Edges that belong to the Boundary Faces
  // delete the repeated edges
  std::vector<stk::mesh::Entity>                 MeshEdges;
  std::vector<stk::mesh::Entity>::const_iterator I_BoundaryFaces;
  std::vector<stk::mesh::Entity>::const_iterator I_Edges;
  for (I_BoundaryFaces = BoundaryFaces.begin();
       I_BoundaryFaces != BoundaryFaces.end();
       I_BoundaryFaces++) {
    std::vector<stk::mesh::Entity> boundaryEdges = getDirectlyConnectedEntities(
        *I_BoundaryFaces, stk::topology::EDGE_RANK);
    for (I_Edges = boundaryEdges.begin(); I_Edges != boundaryEdges.end();
         I_Edges++) {
      if (findEntityInVector(MeshEdges, *I_Edges) == false) {
        MeshEdges.push_back(*I_Edges);
      }
    }
  }

  return MeshEdges;
}

//
// \brief Returns the shortest path over the boundary faces given
//        three input nodes and the edges that belong to the outer
//        surface
//
std::vector<std::vector<int>>
Topology::shortestpathOnBoundaryFaces(
    const std::vector<stk::mesh::Entity>& nodes,
    const std::vector<stk::mesh::Entity>& MeshEdgesShortestPath)
{
  typedef float                                              Weight;
  typedef boost::property<boost::edge_weight_t, Weight>      WeightProperty;
  typedef boost::property<boost::vertex_name_t, std::string> NameProperty;

  typedef boost::adjacency_list<
      boost::vecS,
      boost::vecS,
      boost::undirectedS,
      NameProperty,
      WeightProperty>
      Graph;

  typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;

  typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
  typedef boost::property_map<Graph, boost::vertex_name_t>::type  NameMap;

  typedef boost::iterator_property_map<Vertex*, IndexMap, Vertex, Vertex&>
      PredecessorMap;

  typedef boost::iterator_property_map<Weight*, IndexMap, Weight, Weight&>
      DistanceMap;

  // Define the input graph
  Graph g;

  // Add the edges weights to the graph
  for (unsigned int i = 0; i < MeshEdgesShortestPath.size(); ++i) {
    std::vector<stk::mesh::Entity> EdgeBoundaryNodes;
    EdgeBoundaryNodes = getDirectlyConnectedEntities(
        MeshEdgesShortestPath[i], stk::topology::NODE_RANK);
    Weight weight(
        getDistanceBetweenNodes(EdgeBoundaryNodes[0], EdgeBoundaryNodes[1]));
    boost::add_edge(
        get_entity_id(EdgeBoundaryNodes[0]) - 1,
        get_entity_id(EdgeBoundaryNodes[1]) - 1,
        weight,
        g);
  }

  // Create predecessors and distances vectors
  // Defined to save parents
  std::vector<Vertex> predecessors(boost::num_vertices(g));
  // Defined to save distances
  std::vector<Weight> distances(boost::num_vertices(g));

  IndexMap       indexMap = boost::get(boost::vertex_index, g);
  PredecessorMap predecessorMap(&predecessors[0], indexMap);
  DistanceMap    distanceMap(&distances[0], indexMap);

  // NOTE: The dijkstra_shortest_paths function returns the nodes
  // corresponding to the shortest path starting from the end node to
  // the start node
  //
  // Compute the  shortest path between nodes[0] and nodes[1]

  // Source from where the distance and path are calculated
  Vertex sourceVertex_0 = get_entity_id(nodes[1]) - 1;
  Vertex goalVertex_0   = get_entity_id(nodes[0]) - 1;
  boost::dijkstra_shortest_paths(
      g,
      sourceVertex_0,
      boost::distance_map(distanceMap).predecessor_map(predecessorMap));

  std::vector<boost::graph_traits<Graph>::vertex_descriptor> ShortestPath_0;

  boost::graph_traits<Graph>::vertex_descriptor currentVertex_0 = goalVertex_0;
  while (currentVertex_0 != sourceVertex_0) {
    ShortestPath_0.push_back(currentVertex_0);
    currentVertex_0 = predecessorMap[currentVertex_0];
  }
  // ShortestPath contains the shortest path between the given
  // vertices.  Vertices are saved starting from end vertex to start
  // vertex. The format of this output is "unsigned long int"
  ShortestPath_0.push_back(sourceVertex_0);

  // Change the format of the output from unsigned long int to int
  std::vector<int>                               FV_0;
  std::vector<unsigned long int>::const_iterator I_points0;
  for (I_points0 = ShortestPath_0.begin(); I_points0 != ShortestPath_0.end();
       ++I_points0) {
    int i = boost::numeric_cast<int>(*I_points0);
    FV_0.push_back(i);
  }
  // Define the vector that holds the shortest path
  std::vector<std::vector<int>> ShortestPathFinal;
  ShortestPathFinal.push_back(FV_0);

  // Compute the  shortest path between nodes[1] and nodes[2]
  // Source from where the distance and path are calculated
  Vertex sourceVertex_1 = get_entity_id(nodes[2]) - 1;
  Vertex goalVertex_1   = get_entity_id(nodes[1]) - 1;
  boost::dijkstra_shortest_paths(
      g,
      sourceVertex_1,
      boost::distance_map(distanceMap).predecessor_map(predecessorMap));

  std::vector<boost::graph_traits<Graph>::vertex_descriptor> ShortestPath_1;

  boost::graph_traits<Graph>::vertex_descriptor currentVertex_1 = goalVertex_1;
  while (currentVertex_1 != sourceVertex_1) {
    ShortestPath_1.push_back(currentVertex_1);
    currentVertex_1 = predecessorMap[currentVertex_1];
  }
  // ShortestPath contains the shortest path between the given
  // vertices.  Vertices are saved starting from end vertex to start
  // vertex. The format of this output is "unsigned long int"
  ShortestPath_1.push_back(sourceVertex_1);

  // Change the format of the output from unsigned long int to int
  std::vector<int>                               FV_1;
  std::vector<unsigned long int>::const_iterator I_points1;
  for (I_points1 = ShortestPath_1.begin(); I_points1 != ShortestPath_1.end();
       ++I_points1) {
    int i = boost::numeric_cast<int>(*I_points1);
    FV_1.push_back(i);
  }
  ShortestPathFinal.push_back(FV_1);

  // Compute the  shortest path between nodes[2] and nodes[0]
  // Source from where the distance and path are calculated
  Vertex sourceVertex_2 = get_entity_id(nodes[0]) - 1;
  Vertex goalVertex_2   = get_entity_id(nodes[2]) - 1;
  boost::dijkstra_shortest_paths(
      g,
      sourceVertex_2,
      boost::distance_map(distanceMap).predecessor_map(predecessorMap));

  std::vector<boost::graph_traits<Graph>::vertex_descriptor> ShortestPath_2;

  boost::graph_traits<Graph>::vertex_descriptor currentVertex_2 = goalVertex_2;
  while (currentVertex_2 != sourceVertex_2) {
    ShortestPath_2.push_back(currentVertex_2);
    currentVertex_2 = predecessorMap[currentVertex_2];
  }
  // ShortestPath contains the shortest path between the given
  // vertices.  Vertices are saved starting from end vertex to start
  // vertex. The format of this output is "unsigned long int"
  ShortestPath_2.push_back(sourceVertex_2);

  // Change the format of the output from unsigned long int to int
  std::vector<int>                               FV_2;
  std::vector<unsigned long int>::const_iterator I_points2;
  for (I_points2 = ShortestPath_2.begin(); I_points2 != ShortestPath_2.end();
       ++I_points2) {
    int i = boost::numeric_cast<int>(*I_points2);
    FV_2.push_back(i);
  }
  ShortestPathFinal.push_back(FV_2);

  std::vector<std::vector<int>> ShortestPathOutput;

  for (int j = 0; j < 3; j++) {
    for (unsigned int k = 0; k < (ShortestPathFinal[j].size() - 1); k++) {
      std::vector<int> temp;
      temp.push_back((ShortestPathFinal[j][k]) + 1);
      temp.push_back((ShortestPathFinal[j][k + 1]) + 1);
      ShortestPathOutput.push_back(temp);
    }
  }
  return ShortestPathOutput;
}

//----------------------------------------------------------------------------
//
// \brief Returns the shortest path between three input nodes
//
//
std::vector<std::vector<int>>
Topology::shortestpath(const std::vector<stk::mesh::Entity>& nodes)
{
  typedef float                                              Weight;
  typedef boost::property<boost::edge_weight_t, Weight>      WeightProperty;
  typedef boost::property<boost::vertex_name_t, std::string> NameProperty;

  typedef boost::adjacency_list<
      boost::vecS,
      boost::vecS,
      boost::undirectedS,
      NameProperty,
      WeightProperty>
      Graph;

  typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;

  typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
  typedef boost::property_map<Graph, boost::vertex_name_t>::type  NameMap;

  typedef boost::iterator_property_map<Vertex*, IndexMap, Vertex, Vertex&>
      PredecessorMap;

  typedef boost::iterator_property_map<Weight*, IndexMap, Weight, Weight&>
      DistanceMap;

  // Define the input graph
  Graph g;

  // Obtain all the faces of the mesh
  std::vector<stk::mesh::Entity> MeshFaces =
      getEntitiesByRank(get_bulk_data(), stk::topology::FACE_RANK);

  // Find the faces (Entities of rank 2) that build the boundary of the
  // given mesh
  std::vector<stk::mesh::Entity>                 BoundaryFaces;
  std::vector<stk::mesh::Entity>::const_iterator I_faces;
  for (I_faces = MeshFaces.begin(); I_faces != MeshFaces.end(); I_faces++) {
    std::vector<stk::mesh::Entity> temp;
    temp = getDirectlyConnectedEntities(*I_faces, stk::topology::ELEMENT_RANK);
    // If the number of boundary entities of rank 3 is 1
    // then, this is a boundary face
    if (temp.size() == 1) { BoundaryFaces.push_back(*I_faces); }
  }

  // Obtain the Edges that belong to the Boundary Faces
  // delete the repeated edges
  std::vector<stk::mesh::Entity>                 MeshEdges;
  std::vector<stk::mesh::Entity>::const_iterator I_BoundaryFaces;
  std::vector<stk::mesh::Entity>::const_iterator I_Edges;
  for (I_BoundaryFaces = BoundaryFaces.begin();
       I_BoundaryFaces != BoundaryFaces.end();
       I_BoundaryFaces++) {
    std::vector<stk::mesh::Entity> boundaryEdges = getDirectlyConnectedEntities(
        *I_BoundaryFaces, stk::topology::EDGE_RANK);
    for (I_Edges = boundaryEdges.begin(); I_Edges != boundaryEdges.end();
         I_Edges++) {
      if (findEntityInVector(MeshEdges, *I_Edges) == false) {
        MeshEdges.push_back(*I_Edges);
      }
    }
  }

  // Add the edges weights to the graph
  for (unsigned int i = 0; i < MeshEdges.size(); ++i) {
    std::vector<stk::mesh::Entity> EdgeBoundaryNodes;
    EdgeBoundaryNodes =
        getDirectlyConnectedEntities(MeshEdges[i], stk::topology::NODE_RANK);
    Weight weight(
        getDistanceBetweenNodes(EdgeBoundaryNodes[0], EdgeBoundaryNodes[1]));
    boost::add_edge(
        get_entity_id(EdgeBoundaryNodes[0]) - 1,
        get_entity_id(EdgeBoundaryNodes[1]) - 1,
        weight,
        g);  //
  }

  // Create predecessors and distances vectors
  // Defined to save parents
  std::vector<Vertex> predecessors(boost::num_vertices(g));
  // Defined to save distances
  std::vector<Weight> distances(boost::num_vertices(g));

  IndexMap       indexMap = boost::get(boost::vertex_index, g);
  PredecessorMap predecessorMap(&predecessors[0], indexMap);
  DistanceMap    distanceMap(&distances[0], indexMap);

  // NOTE: The dijkstra_shortest_paths function returns the nodes
  // corresponding to the shortest path starting from the end node to
  // the start node
  //
  // Compute the  shortest path between nodes[0] and nodes[1]

  // Source from where the distance and path are calculated
  Vertex sourceVertex_0 = get_entity_id(nodes[1]) - 1;
  Vertex goalVertex_0   = get_entity_id(nodes[0]) - 1;
  boost::dijkstra_shortest_paths(
      g,
      sourceVertex_0,
      boost::distance_map(distanceMap).predecessor_map(predecessorMap));

  std::vector<boost::graph_traits<Graph>::vertex_descriptor> ShortestPath_0;

  boost::graph_traits<Graph>::vertex_descriptor currentVertex_0 = goalVertex_0;
  while (currentVertex_0 != sourceVertex_0) {
    ShortestPath_0.push_back(currentVertex_0);
    currentVertex_0 = predecessorMap[currentVertex_0];
  }
  // ShortestPath contains the shortest path between the given
  // vertices.  Vertices are saved starting from end vertex to start
  // vertex. The format of this output is "unsigned long int"
  ShortestPath_0.push_back(sourceVertex_0);

  // Change the format of the output from unsigned long int to int
  std::vector<int>                               FV_0;
  std::vector<unsigned long int>::const_iterator I_points0;
  for (I_points0 = ShortestPath_0.begin(); I_points0 != ShortestPath_0.end();
       ++I_points0) {
    int i = boost::numeric_cast<int>(*I_points0);
    FV_0.push_back(i);
  }
  // Define the vector that holds the shortest path
  std::vector<std::vector<int>> ShortestPathFinal;
  ShortestPathFinal.push_back(FV_0);

  // Compute the  shortest path between nodes[1] and nodes[2]
  // Source from where the distance and path are calculated
  Vertex sourceVertex_1 = get_entity_id(nodes[2]) - 1;
  Vertex goalVertex_1   = get_entity_id(nodes[1]) - 1;
  boost::dijkstra_shortest_paths(
      g,
      sourceVertex_1,
      boost::distance_map(distanceMap).predecessor_map(predecessorMap));

  std::vector<boost::graph_traits<Graph>::vertex_descriptor> ShortestPath_1;

  boost::graph_traits<Graph>::vertex_descriptor currentVertex_1 = goalVertex_1;
  while (currentVertex_1 != sourceVertex_1) {
    ShortestPath_1.push_back(currentVertex_1);
    currentVertex_1 = predecessorMap[currentVertex_1];
  }
  // ShortestPath contains the shortest path between the given
  // vertices.  Vertices are saved starting from end vertex to start
  // vertex. The format of this output is "unsigned long int"
  ShortestPath_1.push_back(sourceVertex_1);

  // Change the format of the output from unsigned long int to int
  std::vector<int>                               FV_1;
  std::vector<unsigned long int>::const_iterator I_points1;
  for (I_points1 = ShortestPath_1.begin(); I_points1 != ShortestPath_1.end();
       ++I_points1) {
    int i = boost::numeric_cast<int>(*I_points1);
    FV_1.push_back(i);
  }
  ShortestPathFinal.push_back(FV_1);

  // Compute the  shortest path between nodes[2] and nodes[0]
  // Source from where the distance and path are calculated
  Vertex sourceVertex_2 = get_entity_id(nodes[0]) - 1;
  Vertex goalVertex_2   = get_entity_id(nodes[2]) - 1;
  boost::dijkstra_shortest_paths(
      g,
      sourceVertex_2,
      boost::distance_map(distanceMap).predecessor_map(predecessorMap));

  std::vector<boost::graph_traits<Graph>::vertex_descriptor> ShortestPath_2;

  boost::graph_traits<Graph>::vertex_descriptor currentVertex_2 = goalVertex_2;
  while (currentVertex_2 != sourceVertex_2) {
    ShortestPath_2.push_back(currentVertex_2);
    currentVertex_2 = predecessorMap[currentVertex_2];
  }
  // ShortestPath contains the shortest path between the given
  // vertices.  Vertices are saved starting from end vertex to start
  // vertex. The format of this output is "unsigned long int"
  ShortestPath_2.push_back(sourceVertex_2);

  // Change the format of the output from unsigned long int to int
  std::vector<int>                               FV_2;
  std::vector<unsigned long int>::const_iterator I_points2;
  for (I_points2 = ShortestPath_2.begin(); I_points2 != ShortestPath_2.end();
       ++I_points2) {
    int i = boost::numeric_cast<int>(*I_points2);
    FV_2.push_back(i);
  }
  ShortestPathFinal.push_back(FV_2);

  std::vector<std::vector<int>> ShortestPathOutput;

  for (int j = 0; j < 3; j++) {
    for (unsigned int k = 0; k < (ShortestPathFinal[j].size() - 1); k++) {
      std::vector<int> temp;
      temp.push_back((ShortestPathFinal[j][k]) + 1);
      temp.push_back((ShortestPathFinal[j][k + 1]) + 1);
      ShortestPathOutput.push_back(temp);
    }
  }
  return ShortestPathOutput;
}

//----------------------------------------------------------------------------
//
// \brief Returns the directions of all the edges of the input mesh
//
std::vector<std::vector<int>>
Topology::edgesDirections()
{
  // Get all of the edges
  std::vector<stk::mesh::Entity> setOfEdges =
      getEntitiesByRank(get_bulk_data(), stk::topology::EDGE_RANK);

  // Create a map that assigns new numbering to the Edges
  std::map<stk::mesh::Entity, int>               edge_map;
  int                                            counter = 0;
  std::vector<stk::mesh::Entity>::const_iterator I_setOfEdges;
  for (I_setOfEdges = setOfEdges.begin(); I_setOfEdges != setOfEdges.end();
       ++I_setOfEdges) {
    edge_map[*I_setOfEdges] = counter;
    counter++;
  }

  // edgesDirec will be the vector of vectors that is returned, it will be Nx2,
  // where N is the number of edges, and each edge has two nodes
  std::vector<std::vector<int>> edgesDirec(setOfEdges.size());
  std::map<stk::mesh::Entity, int>::const_iterator mapIter;

  // Iterate through the map, at each row of edgesDirec save the
  // integers that identify the directions of the edges
  for (mapIter = edge_map.begin(); mapIter != edge_map.end(); ++mapIter) {
    int                            index = mapIter->second;
    std::vector<stk::mesh::Entity> connectedNodes =
        getDirectlyConnectedEntities(mapIter->first, stk::topology::NODE_RANK);
    std::vector<int> tempInt;
    tempInt.push_back(get_entity_id(connectedNodes[0]));
    tempInt.push_back(get_entity_id(connectedNodes[1]));
    edgesDirec[index] = tempInt;
  }

  return edgesDirec;
}

//----------------------------------------------------------------------------
//
// \brief Returns the directions of all the boundary edges of the input mesh
//
std::vector<std::vector<int>>
Topology::edgesDirectionsOuterSurface()
{
  // Obtain all the faces of the mesh
  std::vector<stk::mesh::Entity> MeshFaces =
      getEntitiesByRank(get_bulk_data(), stk::topology::FACE_RANK);

  // Find the faces (Entities of rank 2) that build the boundary of the
  // given mesh
  std::vector<stk::mesh::Entity>                 BoundaryFaces;
  std::vector<stk::mesh::Entity>::const_iterator I_faces;
  for (I_faces = MeshFaces.begin(); I_faces != MeshFaces.end(); I_faces++) {
    std::vector<stk::mesh::Entity> temp;
    temp = getDirectlyConnectedEntities(*I_faces, stk::topology::ELEMENT_RANK);
    // If the number of boundary entities of rank 3 is 1
    // then, this is a boundary face
    if (temp.size() == 1) { BoundaryFaces.push_back(*I_faces); }
  }

  // Obtain the Edges that belong to the Boundary Faces
  // delete the repeated edges
  std::vector<stk::mesh::Entity>                 setOfEdges;
  std::vector<stk::mesh::Entity>::const_iterator I_BoundaryFaces;
  std::vector<stk::mesh::Entity>::const_iterator I_Edges;
  for (I_BoundaryFaces = BoundaryFaces.begin();
       I_BoundaryFaces != BoundaryFaces.end();
       I_BoundaryFaces++) {
    std::vector<stk::mesh::Entity> boundaryEdges = getDirectlyConnectedEntities(
        *I_BoundaryFaces, stk::topology::EDGE_RANK);
    for (I_Edges = boundaryEdges.begin(); I_Edges != boundaryEdges.end();
         I_Edges++) {
      if (findEntityInVector(setOfEdges, *I_Edges) == false) {
        setOfEdges.push_back(*I_Edges);
      }
    }
  }

  // Create a map that assigns new numbering to the Edges
  std::map<stk::mesh::Entity, int>               edge_map;
  int                                            counter = 0;
  std::vector<stk::mesh::Entity>::const_iterator I_setOfEdges;
  for (I_setOfEdges = setOfEdges.begin(); I_setOfEdges != setOfEdges.end();
       ++I_setOfEdges) {
    edge_map[*I_setOfEdges] = counter;
    counter++;
  }

  // edgesDirec will be the vector of vectors that is returned, it will be Nx2,
  // where N is the number of edges, and each edge has two nodes
  std::vector<std::vector<int>> edgesDirec(setOfEdges.size());
  std::map<stk::mesh::Entity, int>::const_iterator mapIter;

  // Iterate through the map, at each row of edgesDirec save the
  // integers that identify the directions of the edges
  for (mapIter = edge_map.begin(); mapIter != edge_map.end(); ++mapIter) {
    int                            index = mapIter->second;
    std::vector<stk::mesh::Entity> connectedNodes =
        getDirectlyConnectedEntities(mapIter->first, stk::topology::NODE_RANK);
    std::vector<int> tempInt;
    tempInt.push_back(get_entity_id(connectedNodes[0]));
    tempInt.push_back(get_entity_id(connectedNodes[1]));
    edgesDirec[index] = tempInt;
  }

  return edgesDirec;
}

//----------------------------------------------------------------------------
//
// \brief Returns the directions of all of the faces of the input mesh
//
std::vector<std::vector<int>>
Topology::facesDirections()
{
  // Get the faces
  std::vector<stk::mesh::Entity> setOfFaces =
      getEntitiesByRank(get_bulk_data(), stk::topology::FACE_RANK);

  // Make a new map, mapping the Entities of Rank 2(faces) to a counter
  std::map<stk::mesh::Entity, int>               face_map;
  int                                            counter = 0;
  std::vector<stk::mesh::Entity>::const_iterator I_setOfFaces;
  for (I_setOfFaces = setOfFaces.begin(); I_setOfFaces != setOfFaces.end();
       ++I_setOfFaces) {
    face_map[*I_setOfFaces] = counter;
    counter++;
  }

  // facesDirec will be the vector of vectors that is returned, it will
  // be Nx4, where N is the number of faces, and each face has three
  // nodes with the first node being repeated for a total of 4
  std::vector<std::vector<int>> facesDirec(setOfFaces.size());

  // Iterate through the map, at each row of facesDirec save the integers that
  // identify the directions of the face
  std::map<stk::mesh::Entity, int>::const_iterator mapIter;
  for (mapIter = face_map.begin(); mapIter != face_map.end(); ++mapIter) {
    int                            index = mapIter->second;
    std::vector<stk::mesh::Entity> edgeBoundaryNodes =
        getBoundaryEntities(mapIter->first, stk::topology::NODE_RANK);
    std::vector<int> temp;
    temp.push_back(get_entity_id(edgeBoundaryNodes[0]));
    temp.push_back(get_entity_id(edgeBoundaryNodes[1]));
    temp.push_back(get_entity_id(edgeBoundaryNodes[2]));
    temp.push_back(get_entity_id(edgeBoundaryNodes[0]));
    facesDirec[index] = temp;
  }

  return facesDirec;
}

//----------------------------------------------------------------------------
//
// \brief Returns a vector with the areas of each of the faces of the input mesh
//
std::vector<double>
Topology::facesAreas()
{
  std::vector<stk::mesh::Entity> setOfFaces =
      getEntitiesByRank(get_bulk_data(), stk::topology::FACE_RANK);

  // Create the map
  std::map<stk::mesh::Entity, int>               face_map;
  int                                            counter = 0;
  std::vector<stk::mesh::Entity>::const_iterator I_setOfFaces;
  for (I_setOfFaces = setOfFaces.begin(); I_setOfFaces != setOfFaces.end();
       ++I_setOfFaces) {
    face_map[*I_setOfFaces] = counter;
    counter++;
  }

  // Initialize facesAreas as a vector of zeros
  std::vector<double> facesAreas;
  for (unsigned int i = 0; i < (setOfFaces.size()); i++) {
    facesAreas.push_back(0);
  }

  // Iterate through the map
  std::map<stk::mesh::Entity, int>::const_iterator mapIter;
  for (mapIter = face_map.begin(); mapIter != face_map.end(); ++mapIter) {
    // Obtain the key from the map
    int index = mapIter->second;

    // Compute the area
    std::vector<stk::mesh::Entity> Nodes =
        getBoundaryEntities(mapIter->first, stk::topology::NODE_RANK);
    double a    = getDistanceBetweenNodes(Nodes[0], Nodes[1]);
    double b    = getDistanceBetweenNodes(Nodes[1], Nodes[2]);
    double c    = getDistanceBetweenNodes(Nodes[2], Nodes[0]);
    double p    = (a + b + c) / 2;
    double Area = sqrt(p * (p - a) * (p - b) * (p - c));

    // Put the area into the array the right index
    facesAreas[index] = Area;
  }

  return facesAreas;
}

//----------------------------------------------------------------------------
//
// \brief Returns the boundary operator of the input mesh.
//        matrix that has nonzeros only
//
std::vector<std::vector<int>>
Topology::boundaryOperator()
{
  std::vector<std::vector<int>>  edgesDirec = edgesDirections();
  std::vector<std::vector<int>>  facesDirec = facesDirections();
  std::vector<stk::mesh::Entity> meshFaces =
      getEntitiesByRank(get_bulk_data(), stk::topology::FACE_RANK);
  std::vector<std::vector<int>> boundaryOp;

  // Iterate through every row of facesDirec
  for (unsigned int i = 0; i < facesDirec.size(); i++) {
    int              mIndex  = 2 * i;
    std::vector<int> faceDir = facesDirec[i];

    // Iterate through the ith row of facesDir
    for (unsigned int j = 0; j < 3; j++) {
      // Iterate through the rows of edgesDirec to find the appropriate edge
      for (unsigned int k = 0; k < edgesDirec.size(); k++) {
        // If the edge is found in the correct direction
        if (facesDirec[i][j] == edgesDirec[k][0] &&
            facesDirec[i][j + 1] == edgesDirec[k][1]) {
          std::vector<int> temp1;
          std::vector<int> temp2;
          temp1.push_back(k + 1);
          temp1.push_back(mIndex + 1);
          temp1.push_back(1);  // original positions started from 1 for solver
          // For testing
          // temp1.push_back(k);temp1.push_back(mIndex);temp1.push_back(1);
          boundaryOp.push_back(temp1);
          temp2.push_back(k + 1);
          temp2.push_back(mIndex + 2);
          temp2.push_back(-1);  // original positions started from 1 for solver
          // For testing
          // temp2.push_back(k);temp2.push_back(mIndex+1);temp2.push_back(-1);
          boundaryOp.push_back(temp2);

        }
        // If the edge is found in the opposite direction
        else if ((facesDirec[i][j + 1] == edgesDirec[k][0] &&
                  facesDirec[i][j] == edgesDirec[k][1])) {
          std::vector<int> temp3;
          std::vector<int> temp4;
          temp3.push_back(k + 1);
          temp3.push_back(mIndex + 1);
          temp3.push_back(-1);  // original positions started from 1 for
          // solver GLPK

          // For testing
          // temp3.push_back(k);temp3.push_back(mIndex);temp3.push_back(-1);
          boundaryOp.push_back(temp3);
          temp4.push_back(k + 1);
          temp4.push_back(mIndex + 2);
          temp4.push_back(1);  // original positions started from 1 for solver

          // For testing
          // temp4.push_back(k);temp4.push_back(mIndex+1);temp4.push_back(1);
          boundaryOp.push_back(temp4);
        }
      }
    }
  }

  return boundaryOp;
}

//
// \brief returns the boundary operator along with the faces areas
//        to create the columns of an mps file
//
std::vector<std::vector<double>>
Topology::outputForMpsFile()
{
  // Obtain the areas of all the faces of the mesh
  std::vector<double> FacesAreas = facesAreas();

  // Define the boundary operator
  std::vector<std::vector<int>>  edgesDirec = edgesDirections();
  std::vector<std::vector<int>>  facesDirec = facesDirections();
  std::vector<stk::mesh::Entity> meshFaces =
      getEntitiesByRank(get_bulk_data(), stk::topology::FACE_RANK);
  std::vector<std::vector<double>> matrixForMpsFile;

  // Iterate through every row of facesDirec
  for (unsigned int i = 0; i < facesDirec.size(); i++) {
    int              mIndex  = 2 * i;
    std::vector<int> faceDir = facesDirec[i];

    // Add the area of the face of analysis
    std::vector<double> temp;
    temp.push_back(FacesAreas[i]);
    matrixForMpsFile.push_back(temp);

    // Iterate through the ith row of facesDir
    for (unsigned int j = 0; j < 3; j++) {
      // Iterate through the rows of edgesDirec to find the appropriate edge
      for (unsigned int k = 0; k < edgesDirec.size(); k++) {
        // If the edge is found in the correct direction
        if (facesDirec[i][j] == edgesDirec[k][0] &&
            facesDirec[i][j + 1] == edgesDirec[k][1]) {
          std::vector<double> temp1;
          // The column position is saved first and then the
          // corresponding column And finally the number
          temp1.push_back(mIndex + 1);
          temp1.push_back(k + 1);  // original positions started from 1
          // for solver
          temp1.push_back(1);
          temp1.push_back(mIndex + 2);
          temp1.push_back(k + 1);  // original positions started from 1
          // for solver
          temp1.push_back(-1);
          matrixForMpsFile.push_back(temp1);
        }
        // If the edge is found in the opposite direction
        else if ((facesDirec[i][j + 1] == edgesDirec[k][0] &&
                  facesDirec[i][j] == edgesDirec[k][1])) {
          std::vector<double> temp2;
          temp2.push_back(mIndex + 1);
          temp2.push_back(k + 1);  // original positions started from 1
          // for solver GLPK
          temp2.push_back(-1);
          temp2.push_back(mIndex + 2);
          temp2.push_back(k + 1);  // original positions started from 1
          // for solver
          temp2.push_back(1);
          matrixForMpsFile.push_back(temp2);
        }
      }
    }
  }

  return matrixForMpsFile;
}

//
// \brief Returns the 1-D boundary required to compute the minimum
//        surface of the input mesh. The input to this function is a
//        shortest path (composed by egdes) between three nodes
//
std::vector<std::vector<int>>
Topology::boundaryVector(std::vector<std::vector<int>>& shortPath)
{
  std::vector<std::vector<int>> edgesDirec = edgesDirections();

  // Define the Matrix that will hold the edges that build the 1D
  // boundary and their respective position
  std::vector<std::vector<int>> rVector;

  // Iterate through all the edges of edgesDirections
  for (unsigned int i = 0; i < edgesDirec.size(); i++) {
    // temp is the value that will go in the vector at the index i
    int temp = 0;
    // count is to ensure that a given edge is not in the shortest path
    // more than once
    int count = 0;

    // Iterate through all the edges of the shortest path
    for (unsigned int j = 0; j < shortPath.size(); j++) {
      // Check if the edge from edges directions matches the ith edge
      // from shortestpath
      if (edgesDirec[i][0] == shortPath[j][0] &&
          edgesDirec[i][1] == shortPath[j][1]) {
        temp = 1;
        count++;
      }
      // Check if the edge from edges directions matches the reverse of
      // the ith edge from shortestpath
      else if (
          edgesDirec[i][1] == shortPath[j][0] &&
          edgesDirec[i][0] == shortPath[j][1]) {
        temp = -1;
        count++;
      }
    }
    // Handles the case in which an edge is in the shortest path more
    // than once
    if (count < 2 && temp != 0) {
      std::vector<int> temp1;
      temp1.push_back(i + 1);  // indexing starts from 1 as required
      // by the Solver
      temp1.push_back(temp);
      rVector.push_back(temp1);
    }
  }
  return rVector;
}

//
// \brief Returns the 1-D boundary required to compute the minimum
//        surface of the input mesh boundary faces. The input to this
//        function is a shortest path (composed by edges) between
//        three nodes
//
std::vector<std::vector<int>>
Topology::boundaryVectorOuterSurface(std::vector<std::vector<int>>& shortPath)
{
  std::vector<std::vector<int>> edgesDirec = edgesDirectionsOuterSurface();

  // Define the Matrix that will hold the edges that build the 1D
  // boundary and their respective position
  std::vector<std::vector<int>> rVector;

  // Iterate through all the edges of edgesDirections
  for (unsigned int i = 0; i < edgesDirec.size(); i++) {
    // temp is the value that will go in the vector at the index i
    int temp = 0;
    // count is to ensure that a given edge is not in the shortest path
    // more than once
    int count = 0;

    // Iterate through all the edges of the shortest path
    for (unsigned int j = 0; j < shortPath.size(); j++) {
      // Check if the edge from edges directions matches the ith edge
      // from shortestpath
      if (edgesDirec[i][0] == shortPath[j][0] &&
          edgesDirec[i][1] == shortPath[j][1]) {
        temp = 1;
        count++;
      }
      // Check if the edge from edges directions matches the reverse of
      // the ith edge from shortestpath
      else if (
          edgesDirec[i][1] == shortPath[j][0] &&
          edgesDirec[i][0] == shortPath[j][1]) {
        temp = -1;
        count++;
      }
    }
    // Handles the case in which an edge is in the shortest path more
    // than once.
    if (count < 2 && temp != 0) {
      std::vector<int> temp1;
      temp1.push_back(i + 1);  // indexing starts from 1 as required
      // by the Solver
      temp1.push_back(temp);
      rVector.push_back(temp1);
    }
  }
  return rVector;
}

//
// \brief Returns the corresponding entities of rank 2 that build the
//        minimum surface.  It takes as an input the resulting vector
//        taken from the solution of the linear programming solver
//
std::vector<stk::mesh::Entity>
Topology::minimumSurfaceFaces(std::vector<int> VectorFromLPSolver)
{
  // Obtain the faces
  std::vector<stk::mesh::Entity> setOfFaces =
      getEntitiesByRank(get_bulk_data(), stk::topology::FACE_RANK);

  // Define the map with the entities and their identifiers
  std::map<int, stk::mesh::Entity>               face_map;
  int                                            counter = 0;
  std::vector<stk::mesh::Entity>::const_iterator I_setOfFaces;
  for (I_setOfFaces = setOfFaces.begin(); I_setOfFaces != setOfFaces.end();
       ++I_setOfFaces) {
    face_map[counter] = *I_setOfFaces;
    counter++;
  }

  // Use the input vector to obtain the corresponding entities
  std::vector<stk::mesh::Entity> MinSurfaceEntities;
  int                            count = 0;
  for (unsigned int i = 0; i < VectorFromLPSolver.size(); i += 2) {
    if (VectorFromLPSolver[i] != 0) {
      MinSurfaceEntities.push_back(face_map.find(count)->second);
    }
    if (VectorFromLPSolver[i + 1] != 0) {
      MinSurfaceEntities.push_back(face_map.find(count)->second);
    }
    count++;
  }

  return MinSurfaceEntities;
}

//----------------------------------------------------------------------------
// \brief Returns the number of times an entity is repeated in a vector
//
int
Topology::numberOfRepetitions(
    std::vector<stk::mesh::Entity>& entities,
    stk::mesh::Entity               entity)
{
  std::vector<stk::mesh::Entity>::iterator iterator_entities;
  int                                      count = 0;
  for (iterator_entities = entities.begin();
       iterator_entities != entities.end();
       ++iterator_entities) {
    if (*iterator_entities == entity) { count++; }
  }
  return count;
}

//----------------------------------------------------------------------------
//
// \brief Returns the coordinates of an input node.
//        The input is the identifier of a node
//
std::vector<double>
Topology::findCoordinates(unsigned int nodeIdentifier)
{
  std::vector<stk::mesh::Entity> MeshNodes = getEntitiesByRank(
      get_bulk_data(),
      stk::topology::NODE_RANK);  // Get all the nodes
  // of the mesh
  std::vector<stk::mesh::Entity>::const_iterator Ientities_D0;

  std::vector<double> coordinates_;
  for (Ientities_D0 = MeshNodes.begin(); Ientities_D0 != MeshNodes.end();
       Ientities_D0++) {
    if (get_entity_id(*Ientities_D0) == nodeIdentifier) {
      double* coordinate = getEntityCoordinates(*Ientities_D0);
      double  x          = coordinate[0];
      double  y          = coordinate[1];
      double  z          = coordinate[2];

      coordinates_.push_back(x);
      coordinates_.push_back(y);
      coordinates_.push_back(z);

      break;
    }
  }

  return coordinates_;
}

}  // namespace LCM

#endif  // #if defined (ALBANY_LCM)
