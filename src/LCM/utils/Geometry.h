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
  template<typename T>
  T
  length(Vector<T, 3> const & p0, Vector<T, 3> const & p1);

  ///
  /// Area of a triangle
  ///
  template<typename T>
  T
  area(Vector<T, 3> const & p0, Vector<T, 3> const & p1,
      Vector<T, 3> const & p2);

  ///
  /// Area of a quadrilateral, assummed planar. If not planar, returns
  /// the sum of the areas of the two triangles p0,p1,p2 and p0,p2,p3
  ///
  template<typename T>
  T
  area(Vector<T, 3> const & p0, Vector<T, 3> const & p1,
      Vector<T, 3> const & p2, Vector<T, 3> const & p3);

  ///
  /// Volume of tetrahedron
  ///
  template<typename T>
  T
  volume(Vector<T, 3> const & p0, Vector<T, 3> const & p1,
      Vector<T, 3> const & p2, Vector<T, 3> const & p3);

  ///
  /// Volume of pyramid of quadrilateral base
  /// Base is assumed planar
  /// Base is p0,p1,p2,p3
  /// Apex is p4
  ///
  template<typename T>
  T
  volume(Vector<T, 3> const & p0, Vector<T, 3> const & p1,
      Vector<T, 3> const & p2, Vector<T, 3> const & p3,
      Vector<T, 3> const & p4);

  ///
  /// Volume of hexahedron
  /// Assumption: all faces are planar
  /// Decompose into 3 pyramids
  ///
  template<typename T>
  T
  volume(Vector<T, 3> const & p0, Vector<T, 3> const & p1,
      Vector<T, 3> const & p2, Vector<T, 3> const & p3,
      Vector<T, 3> const & p4, Vector<T, 3> const & p5,
      Vector<T, 3> const & p6, Vector<T, 3> const & p7);

  ///
  /// Centroids of segment, triangle, tetrahedron, quadrilateral
  /// and hexahedron.
  /// For these we can just take the average of the vertices
  ///
  template<typename T>
  Vector<T, 3>
  centroid(std::vector<Vector<T, 3> > const & points);

  ///
  /// The surface normal of a face
  /// Assumption: face is planar
  /// Input: 3 independent nodes on the face
  /// Output: normal vector
  ///
  template<typename T>
  T
  faceNormal(Vector<T,3> const & p0,
          Vector<T,3> const & p1,
          Vector<T,3> const & p2);

} // namespace LCM

#include "Geometry.i.cc"
#include "Geometry.t.cc"

#endif // LCM_Geometry_h
