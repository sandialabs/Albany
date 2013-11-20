//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_CUBATURE_HPP
#define AERAS_CUBATURE_HPP

#include "Intrepid_CubaturePolylib.hpp"

namespace Aeras {

/** \class Aeras::Cubature
    \brief Defines the base class for cubature (integration) rules in Aeras.
*/
template<class Scalar>
class Cubature : public Intrepid::Cubature<Scalar> {
  private:
  const Intrepid::CubaturePolylib<Scalar> polylib_;

  public:

  typedef Intrepid::FieldContainer<Scalar> ArrayPoint;
  typedef ArrayPoint                       ArrayWeight;

  Cubature(const int degree) : polylib_(degree, Intrepid::PL_GAUSS_LOBATTO) {}

  virtual ~Cubature() {}

  /** \brief Returns cubature points and weights
             (return arrays must be pre-sized/pre-allocated).

      \param cubPoints       [out]     - Array containing the cubature points.
      \param cubWeights      [out]     - Array of corresponding cubature weights.
  */
  virtual void getCubature(ArrayPoint  & cubPoints,
                           ArrayWeight & cubWeights) const {

    const int num_points_1d = polylib_.getNumPoints();
    const int numCubPoints  = num_points_1d*num_points_1d;
    const int cellDim       = 2;                

    TEUCHOS_TEST_FOR_EXCEPTION( ( ( (int)cubPoints.size() < numCubPoints*cellDim ) || ( (int)cubWeights.size() < numCubPoints ) ),
                       std::out_of_range,
                       ">>> ERROR (Cubature): Insufficient space allocated for cubature points or weights.");

    ArrayPoint cubPoints_1d  (num_points_1d,1);
    ArrayPoint cubWeights_1d (num_points_1d);
    polylib_.getCubature(cubPoints_1d, cubWeights_1d);

    for (unsigned i=0,k=0; i<num_points_1d; ++i) {
      for (unsigned j=0; j<num_points_1d; ++j,++k) {
        cubPoints (k,0) = cubPoints_1d(i);
        cubPoints (k,1) = cubPoints_1d(j);
        cubWeights(k)   = cubWeights_1d(i)*cubWeights_1d(j);
      }
    }
  }

  /** \brief Returns the number of cubature points.
  */
  virtual int getNumPoints() const { return polylib_.getNumPoints()*polylib_.getNumPoints();}


  /** \brief Returns dimension of the integration domain.
  */
  virtual int getDimension() const { return 2; }


  /** \brief Returns algebraic accuracy (e.g. max. degree of polynomial
             that is integrated exactly). For tensor-product or sparse
             rules, algebraic accuracy for each coordinate direction
             is returned.

             Since the size of the return argument need not be known
             ahead of time, other return options are possible, depending
             on the type of the cubature rule. 
  */
  virtual void getAccuracy(std::vector<int> & accuracy) const { 
    polylib_.getAccuracy(accuracy);
    accuracy.assign(2, accuracy[0]);
  }

}; // class Cubature 

}

#endif
