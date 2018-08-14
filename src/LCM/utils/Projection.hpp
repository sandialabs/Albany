//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(Projection_hpp)
#define Projection_hpp

#include "Albany_Utils.hpp"

namespace LCM {

//
// Class to manage the projection of a variable from quadrature points
// to the element nodes.
// variable may be: scalar, vector, or tensor in 1D, 2D, or 3D.
//

class Projection
{
 public:
  // Default constructor
  Projection();

  // Constructor
  Projection(
      std::string const& field_name,
      int const          rank,
      int const          number_components,
      int const          number_dimensions);

  // Returns whether a projection is required
  bool
  isProjected()
  {
    return is_projected_;
  }

  // Return the rank of the variable to be projected
  int
  getProjectedRank()
  {
    return rank_;
  }

  // Return the number of components to be projected
  int
  getProjectedComponents();

  // Return the name of the variable to be projected
  std::string
  getProjectionName()
  {
    return field_name_;
  }

 private:
  // Determines whether a projection is required
  bool is_projected_;

  // The name of the projected variable
  std::string field_name_;

  // Rank of the projected variable (e.g. 0=scalar, 1=vector, 2=tensor)
  int rank_;

  // Number of components in the variable
  int number_components_;

  // Spatial dimensions of the system
  int number_dimensions_;
};
// Class Projection

}  // Namespace LCM

#endif  // Projection_hpp
