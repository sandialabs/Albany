/*
 * Projection.hpp
 *
 *  Created on: Jun 29, 2012
 *      Author: jrthune
 */

#ifndef PROJECTION_HPP_
#define PROJECTION_HPP_

#include "Albany_Utils.hpp"

namespace LCM{

/* Class to manage the projection of a variable from the Gaussian quadrature points to the element nodes.
 *   variable may be: scalar, vector, or tensor in 1D, 2D, or 3D.
 */

class Projection {
public:
	// Default constructor
	Projection();

	// Constructor
	Projection(std::string& variableName, int& rank, int& components, int& numDim);

	// Returns whether a projection is required
	bool isProjected(){return isProjected_;};

	// Return the rank of the variable to be projected
	int getProjectedRank(){return rank_;};

	// Return the number of components to be projected
	int getProjectedComponents();

	// Return the name of the variable to be projected
	std::string getProjectionName(){return variableName_;};

private:


	// Determines whether a projection is required
	bool isProjected_;

	// The name of the projected variable
	std::string variableName_;

	// Rank of the projected variable (e.g. 0=scalar, 1=vector, 2=tensor)
	int rank_;

	// Number of components in the variable
	int components_;

	// Spatial dimensions of the system
	int numDim_;

	// Determine the rank of the variable to be projected
	void getRank();

}; // Class Projection

} // Namespace LCM

#endif /* PROJECTION_HPP_ */
