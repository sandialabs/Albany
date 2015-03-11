//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef Integrator_HPP
#define Integrator_HPP

#include "Intrepid_FieldContainer.hpp"
#include <Intrepid_MiniTensor.h>
#include <vector>
#include <functional>

namespace ATO {

/** \brief Conformal integration class

    This class provides basic support for conformal integration.

*/
template<typename T>
class Integrator
{

 public:
  Integrator(Teuchos::RCP<shards::CellTopology> celltype);
  virtual ~Integrator(){};

  template<typename C>
  void getMeasure(T& measure, 
                 const Intrepid::FieldContainer<T>& coordCon, 
                 const Intrepid::FieldContainer<T>& topoVals,
                 T zeroVal,
                 C comparison);

  typedef std::greater_equal<T> Positive;
  typedef std::less_equal<T> Negative;

// this needs c++11 to compile or -std=c++0x
//  template<typename C = std::greater<typename std::iterator_traits<T>::value_type> >
//  void getMeasure(T& measure, 
//                 const Intrepid::FieldContainer<T>& coordCon, 
//                 const Intrepid::FieldContainer<T>& topoVals,
//                 T zeroVal,
//                 C c = C());

  void getCubature(std::vector<std::vector<T> >& refPoints, std::vector<T>& weights, 
                   const Intrepid::FieldContainer<T>& coordCon, 
                   const Intrepid::FieldContainer<T>& topoVals, T zeroVal);

 private:

  typedef Intrepid::Vector<T,3> Vector3D;
  typedef Intrepid::Vector<int,3> Tri;
  typedef Intrepid::Vector<int,4> Tet;

  Teuchos::RCP<shards::CellTopology> cellTopology;

  T getTriMeasure(const std::vector< Vector3D >& points,
                  const Tri& tri);

  template<typename C>
  void getSurfaceTris(std::vector< Vector3D >& points,
                      std::vector< Tri >& tris,
                      const CellTopologyData& cellData,
                      const Intrepid::FieldContainer<T>& coordCon,
                      const Intrepid::FieldContainer<T>& topoVals, 
                      T zeroVal, C comparison);

  void addCubature(std::vector<std::vector<T> >& refPoints, std::vector<T>& weights, 
                   const Vector3D& c0,
                   const Vector3D& c1,
                   const Vector3D& c2);
  void addCubature(std::vector<std::vector<T> >& refPoints, std::vector<T>& weights, 
                   const Vector3D& c0,
                   const Vector3D& c1,
                   const Vector3D& c2,
                   const Vector3D& c3);
};

}

#include "ATO_Integrator_Def.hpp"
#endif
