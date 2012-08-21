/*
 * Projection.cpp
 *
 *  Created on: Jun 29, 2012
 *      Author: jrthune
 */

#include "Projection.hpp"

namespace LCM{

Projection::Projection():
		isProjected_(false),
		rank_(0),
		numDim_(0)
{
	return;
}

Projection::Projection(std::string& variableName, int& rank, int& components, int& numDim):
		isProjected_(true),
		rank_(rank),
		components_(components),
		numDim_(numDim),
		variableName_(variableName)
{
	if(variableName.empty())
	{
		isProjected_=false;
	}
	else{
		Projection::getRank();
	}
	return;
}

// Return the number of components to be projected
int Projection::getProjectedComponents()
{

	/* The number of components is not necessarily determined solely by the variable rank and spatial dimension
	 * of the problem. For now, assume that the number of components is passed to class from the input file
    int projectedComp;

	if (rank_!=0){
		projectedComp = rank_*numDim_;
	}
	else
		projectedComp = 1;

	return projectedComp;*/
	return components_;
}

void Projection::getRank()
{
	// Assume the projected variable is a scalar for now - TODO: change once you get it working for vectors and tensors
	//rank_ = 0;
}

} // namspace LCM


