//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Define only if ALbany is enabled
#if defined (ALBANY_LCM)

#include <stk_mesh/base/FieldData.hpp>
#include "Topology.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <cmath>


namespace LCM {

//
// \brief Finds the closest nodes(Entities of rank 0) to each of the three points in the input vector.
//
std::vector<Entity*> Topology::getClosestNodes(std::vector<std::vector<double> > points)
{
	std::vector<Entity*> closestNodes;
	Entity* nodeA;
	Entity* nodeB;
	Entity* nodeC;
	std::vector<double> pointA, pointB, pointC;
	double minDA, minDB, minDC;

	std::vector<Entity*> entities_D0 = getEntitiesByRank(*(getBulkData()), 0);//get all the nodes
	std::vector<Entity*>::const_iterator i_entities_d0;//iterator for the nodes

	//Before iterate, it is necessary to have a distance with which it is possible to compare the new distances to.
	nodeA = entities_D0[0];
	minDA = getDistanceNodeAndPoint(entities_D0[0], points[0]);

	nodeB = entities_D0[0];
	minDB = getDistanceNodeAndPoint(entities_D0[0], points[1]);

	nodeC = entities_D0[0];
	minDC = getDistanceNodeAndPoint(entities_D0[0], points[2]);

	//For each of the nodes
	//calculate distance from point1, point 2, point 3
	//if any distance is less than the min distance to that point
	//update the min distance and the closest node for that point
	for (i_entities_d0 = entities_D0.begin();i_entities_d0 != entities_D0.end(); ++i_entities_d0)
	{
		//adist is the distance between the current node and the first point, point A.
		double aDist = getDistanceNodeAndPoint(*i_entities_d0, points[0]);
		if (aDist<minDA)
		{
			nodeA = *i_entities_d0;
			minDA = aDist;
		}

		double bDist = getDistanceNodeAndPoint(*i_entities_d0, points[1]);
		if (bDist<minDB)
		{
			nodeB = *i_entities_d0;
			minDB = bDist;
		}

		double cDist = getDistanceNodeAndPoint(*i_entities_d0, points[2]);
		if (cDist<minDC)
		{
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
double Topology::getDistanceNodeAndPoint(Entity* node, std::vector<double> point)
{


	double * entity_coordinates_xyz = getPointerOfCoordinates(node);
	double x_Node = entity_coordinates_xyz[0];
	double y_Node = entity_coordinates_xyz[1];
	double z_Node = entity_coordinates_xyz[2];

	double x_point = point[0];
	double y_point = point[1];
	double z_point = point[2];

	double x_dist = x_point - x_Node;
	double y_dist = y_point - y_Node;
	double z_dist = z_point - z_Node;

	//Compute the distance between the point and the node (Entity of rank 0)
	return sqrt(x_dist*x_dist + y_dist*y_dist + z_dist*z_dist);
}

//
// \brief Returns the coordinates of the points that form a equilateral triangle.
// This triangle lies on the plane that intersects the ellipsoid.
//
std::vector<std::vector<double> > Topology::getCoordinatesOfTriangle(const std::vector<double> normalToPlane)
{
	//Compute the coordinates of the resulting circle.
	//This circle results from the intersection of the plane and the ellipsoid
	std::vector<double> CoordOfMaxAndMin = getCoordinatesOfMaxAndMin();
	double maxX = CoordOfMaxAndMin[0];
	double minX = CoordOfMaxAndMin[1];
	double maxY = CoordOfMaxAndMin[2];
	double minY = CoordOfMaxAndMin[3];
	double maxZ = CoordOfMaxAndMin[4];
	double minZ = CoordOfMaxAndMin[5];

	//Find the center of the cube of nodes
	std::vector<double> coordOfCenter;
	coordOfCenter.push_back((maxX + minX)/2.0);
	coordOfCenter.push_back((maxY + minY)/2.0);
	coordOfCenter.push_back((maxZ + minZ)/2.0);

	//Radius of the circle
	double radius = maxX - coordOfCenter[0];

	//Find a perpendicular vector to the input one
	std::vector<double> vectorA;
	double x = 1;
	double y = 1;
	double z = -(normalToPlane[0]*x + normalToPlane[1]*y)/normalToPlane[2];
	double L = sqrt(x*x + y*y +z*z);
	double xA = x*radius/L ;
	vectorA.push_back(xA);
	double yA = y*radius/L ;
	vectorA.push_back(yA);
	double za = z*radius/L;
	vectorA.push_back(za);

    //Find a particular vector perpendicular to the previous two (normalToPlane X vectorA)




	std::vector<std::vector<double> > VV;
    return VV;


}

//
// \brief Returns the distance between two entities of rank 0 (nodes)
//
double Topology::getDistanceBetweenNodes(Entity * node1, Entity * node2)
{
	double * coordinate1 = getPointerOfCoordinates(node1);
	double x1 = coordinate1[0];
	double y1 = coordinate1[1];
	double z1 = coordinate1[2];
	double * coordinate2 = getPointerOfCoordinates(node2);
	double x2 = coordinate2[0];
	double y2 = coordinate2[1];
	double z2 = coordinate2[2];

	double distance = sqrt(pow((x1-x2),2) + pow((y1-y2),2) + pow((z1-z2),2));
	return distance;
}

//
// \brief Returns the coordinates of the max and min of x y and z
// in the order max of, min of x, max of y, min of y, max of z, min of z
//
std::vector<double> Topology::getCoordinatesOfMaxAndMin()
{
	std::vector<Entity*> entities_D0 = getEntitiesByRank(*(getBulkData()), 0);//get all the nodes
	std::vector<Entity*>::const_iterator i_entities_d0;//iterator for the nodes

	//Get the coordinates of the first node
	double * entity_coordinates_xyz = getPointerOfCoordinates(entities_D0[0]);
	double x_coordinate = entity_coordinates_xyz[0];
	double y_coordinate = entity_coordinates_xyz[1];
	double z_coordinate = entity_coordinates_xyz[2];

	//Declare all the variables for the max and min coordinates
	//and set them equal to the values of the first coordinates
	double maxX = x_coordinate;
	double minX = x_coordinate;
	double maxY = y_coordinate;
	double minY = y_coordinate;
	double maxZ = z_coordinate;
	double minZ = z_coordinate;

	//Declare the vector that has the coordinates of the center
	std::vector<double> coordOfMaxAndMin;

	//Iterate through every node
	for(i_entities_d0 = entities_D0.begin(); i_entities_d0 != entities_D0.end();++i_entities_d0)
	{
		//Get the coordinates of the ith node
		double * entity_coordinates_xyz = getPointerOfCoordinates(*i_entities_d0);
		double x_coordinate = entity_coordinates_xyz[0];
		double y_coordinate = entity_coordinates_xyz[1];
		double z_coordinate = entity_coordinates_xyz[2];

		//Compare the x,y, and z coordinates to the max and min for x y and z
		//if value is more extreme than max or min update max or min
		if (x_coordinate > maxX)
		{
			maxX = x_coordinate;
		}
		if (y_coordinate > maxY)
		{
			maxY = y_coordinate;
		}
		if (z_coordinate > maxZ)
		{
			maxZ = z_coordinate;
		}

		if (x_coordinate < minX)
		{
			minX = x_coordinate;
		}
		if (y_coordinate < minY)
		{
			minY = y_coordinate;
		}
		if (z_coordinate < minZ)
		{
			minZ = z_coordinate;
		}
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
// \brief It returns a vector of four vectors,
// each containing the nodes of the exterior boundary.
// The vectors are in order, -X, +X, -Y, +Y
//
std::vector<std::vector<Entity*> > Topology::NodesOnPlane()
{
	//This vector returns the boundary nodes required to
	// get the 1d boundary of the input mesh
	// xmin,xmax,ymin,ymax
	std::vector<std::vector<Entity*> > boundary_Node_Vectors(4);

    //Reads the nodes of the input mesh
	std::vector<Entity*> entities_D0 = getEntitiesByRank(*(getBulkData()), 0);
	std::vector<Entity*>::const_iterator i_entities_d0;

	//Creates vectors that hold the entities with the same value for coordinate x as the maximum and minimum
	std::vector<Entity*> entities_minX;
	std::vector<Entity*> entities_maxX;

	//Initializes the variables that hold the min and max for x
	double * entity_coordinates_xyz = getPointerOfCoordinates(entities_D0[0]);
	double x_coordinate = entity_coordinates_xyz[0];
	double maxX = x_coordinate;
	double minX = x_coordinate;

	//Repeat last two steps with y instead of x
	std::vector<Entity*> entities_minY;
	std::vector<Entity*> entities_maxY;
	double y_coordinate = entity_coordinates_xyz[1];
	double maxY = y_coordinate;
	double minY = y_coordinate;

	//Iterates through every node
	for(i_entities_d0 = entities_D0.begin(); i_entities_d0 != entities_D0.end();++i_entities_d0)
	{
		//Gets the x coordinate of the ith entity
		double * entity_coordinates_xyz = getPointerOfCoordinates(*i_entities_d0);
		double x_coordinate = entity_coordinates_xyz[0];

		//Compares to the max X value
		//if greater than the current max{
		//clear the vector holding the max nodes
		//make the vector just the current entity
		//set the max X value to the current X value}
		if (x_coordinate > maxX)
		{
			entities_maxX.clear();
			entities_maxX.push_back(*i_entities_d0);
			maxX = x_coordinate;
		}

		//If equal to the current max value{
		//add this entity to the vector holding the max entities
		else if (x_coordinate == maxX)
		{
			entities_maxX.push_back(*i_entities_d0);
		}

		//Compares to the min X value
		//if less than the current min{
		//clear the vector holding the min nodes
		//make the vector just the current entity
		//set the min X value to the current X value}
		if (x_coordinate < minX)
		{
			entities_minX.clear();
			entities_minX.push_back(*i_entities_d0);
			minX = x_coordinate;
		}

		//If equal to the current minimum value{
		//add this entity to the vector holding the minimum entities
		else if (x_coordinate == minX)
		{
			entities_minX.push_back(*i_entities_d0);
		}

		//Repeat for y faces
		double y_coordinate = entity_coordinates_xyz[1];
		if (y_coordinate > maxY)
		{
			entities_maxY.clear();
			entities_maxY.push_back(*i_entities_d0);
			maxY = y_coordinate;
		}
		else if (y_coordinate == maxY)
		{
			entities_maxY.push_back(*i_entities_d0);
		}


		if (y_coordinate < minY)
		{
			entities_minY.clear();
			entities_minY.push_back(*i_entities_d0);
			minY = y_coordinate;
		}
		else if (y_coordinate == minY)
		{
			entities_minY.push_back(*i_entities_d0);
		}
	}

	//Putting the four vectors containing nodes of the
	//exterior faces into the vector of vectors that
	//will be returned
	boundary_Node_Vectors[0]= entities_minX;
	boundary_Node_Vectors[1]= entities_maxX;
	boundary_Node_Vectors[2]= entities_minY;
	boundary_Node_Vectors[3]= entities_maxY;

	return boundary_Node_Vectors;
}

//
// \brief Returns the names of all the nodes of the input mesh
//
std::vector<int> Topology::nodeNames(){

	//Get all the nodes of the input mesh
	std::vector<Entity*> entities_D0 = getEntitiesByRank(
			*(getBulkData()), 0);

	//Get the identifier of each node and save them all in a vector
	std::vector<int> vectorWithNames;
	std::vector<Entity*>::iterator IteratorEntities;
	for ( IteratorEntities = entities_D0.begin();
			IteratorEntities != entities_D0.end();++IteratorEntities){
		//The mesh entities in LCM  are identified with numbers from 1 to n
		//The Dijkstra function from boost reads number identifiers from 0 to n
		int name = ((*IteratorEntities)->identifier())-1; //Number identifiers start from zero
		vectorWithNames.push_back(name);
	}

	return vectorWithNames;
}

}//namespace LCM

#endif // #if defined (ALBANY_LCM)

