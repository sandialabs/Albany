//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Projection.hpp"

namespace LCM {

Projection::Projection() :
    is_projected_(false),
    rank_(0),
    number_components_(0),
    number_dimensions_(0)
{
  return;
}

Projection::Projection(
    std::string & field_name,
    int & rank,
    int & number_components,
    int & number_dimensions) :
    is_projected_(true),
    rank_(rank),
    number_components_(number_components),
    number_dimensions_(number_dimensions),
    field_name_(field_name)
{
  if (field_name.empty()) {
    is_projected_ = false;
  }
  return;
}

// Return the number of components to be projected
int Projection::getProjectedComponents()
{

  //
  // The number of components is not necessarily determined solely
  // by the variable rank and spatial dimension of the problem.
  // For now, assume that the number of components is passed to the class
  // from the input file
  //

  /*
   int projectedComp;

   if (rank_!=0){
   projectedComp = rank_*numDim_;
   }
   else
   projectedComp = 1;

   return projectedComp;
   */

  return number_components_;
}

} // namspace LCM

