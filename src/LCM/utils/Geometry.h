///
/// \file Geometry.h
/// Geometry utilities for LCM. Declarations.
/// \author Alejandro Mota
///
#if !defined(LCM_Geometry_h)
#define LCM_Geometry_h

#include <vector>
#include "Tensor.h"

namespace LCM {

  ///
  /// Length of a segment
  ///
  template<typename ScalarT>
  ScalarT
  length(Vector<ScalarT> const & p0, Vector<ScalarT> const & p1);

  ///
  /// Area of a triangle
  ///
  template<typename ScalarT>
  ScalarT
  area(Vector<ScalarT> const & p0, Vector<ScalarT> const & p1,
      Vector<ScalarT> const & p2);

  ///
  /// Area of a quadrilateral, assummed planar. If not planar, returns
  /// the sum of the areas of the two triangles p0,p1,p2 and p0,p2,p3
  ///
  template<typename ScalarT>
  ScalarT
  area(Vector<ScalarT> const & p0, Vector<ScalarT> const & p1,
      Vector<ScalarT> const & p2, Vector<ScalarT> const & p3);

  ///
  /// Volume of tetrahedron
  ///
  template<typename ScalarT>
  ScalarT
  volume(Vector<ScalarT> const & p0, Vector<ScalarT> const & p1,
      Vector<ScalarT> const & p2, Vector<ScalarT> const & p3);

  ///
  /// Volume of pyramid of quadrilateral base
  /// Base is assumed planar
  /// Base is p0,p1,p2,p3
  /// Apex is p4
  ///
  template<typename ScalarT>
  ScalarT
  volume(Vector<ScalarT> const & p0, Vector<ScalarT> const & p1,
      Vector<ScalarT> const & p2, Vector<ScalarT> const & p3,
      Vector<ScalarT> const & p4);

  ///
  /// Volume of hexahedron
  /// Assumption: all faces are planar
  /// Decompose into 3 pyramids
  ///
  template<typename ScalarT>
  ScalarT
  volume(Vector<ScalarT> const & p0, Vector<ScalarT> const & p1,
      Vector<ScalarT> const & p2, Vector<ScalarT> const & p3,
      Vector<ScalarT> const & p4, Vector<ScalarT> const & p5,
      Vector<ScalarT> const & p6, Vector<ScalarT> const & p7);

  ///
  /// Centroids of segment, triangle, tetrahedron, quadrilateral
  /// and hexahedron.
  /// For these we can just take the average of the vertices
  ///
  template<typename ScalarT>
  ScalarT
  centroid(std::vector<Vector<ScalarT> > const & points);

} // namespace LCM

#include "Geometry.i.cc"
#include "Geometry.t.cc"

#endif // LCM_Geometry_h
