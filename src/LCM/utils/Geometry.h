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
  length(Vector<T> const & p0, Vector<T> const & p1);

  ///
  /// Area of a triangle
  ///
  template<typename T>
  T
  area(Vector<T> const & p0, Vector<T> const & p1,
      Vector<T> const & p2);

  ///
  /// Area of a quadrilateral, assummed planar. If not planar, returns
  /// the sum of the areas of the two triangles p0,p1,p2 and p0,p2,p3
  ///
  template<typename T>
  T
  area(Vector<T> const & p0, Vector<T> const & p1,
      Vector<T> const & p2, Vector<T> const & p3);

  ///
  /// Volume of tetrahedron
  ///
  template<typename T>
  T
  volume(Vector<T> const & p0, Vector<T> const & p1,
      Vector<T> const & p2, Vector<T> const & p3);

  ///
  /// Volume of pyramid of quadrilateral base
  /// Base is assumed planar
  /// Base is p0,p1,p2,p3
  /// Apex is p4
  ///
  template<typename T>
  T
  volume(Vector<T> const & p0, Vector<T> const & p1,
      Vector<T> const & p2, Vector<T> const & p3,
      Vector<T> const & p4);

  ///
  /// Volume of hexahedron
  /// Assumption: all faces are planar
  /// Decompose into 3 pyramids
  ///
  template<typename T>
  T
  volume(Vector<T> const & p0, Vector<T> const & p1,
      Vector<T> const & p2, Vector<T> const & p3,
      Vector<T> const & p4, Vector<T> const & p5,
      Vector<T> const & p6, Vector<T> const & p7);

  ///
  /// Centroids of segment, triangle, tetrahedron, quadrilateral
  /// and hexahedron.
  /// For these we can just take the average of the vertices
  ///
  template<typename T>
  Vector<T>
  centroid(std::vector<Vector<T> > const & points);

  ///
  /// The surface normal of a face
  /// Input: 3 independent nodes on the face
  /// Output: normal vector
  ///
  template<typename T>
  Vector<T>
  normal(Vector<T> const & p0,
          Vector<T> const & p1,
          Vector<T> const & p2);

  ///
  /// Given 3 points p0, p1, p2 that define a plane
  /// determine if point p is in the same side of the normal
  /// to the plane as defined by the right hand rule.
  ///
  template<typename T>
  bool
  in_normal_side(
      Vector<T> const & p,
      Vector<T> const & p0,
      Vector<T> const & p1,
      Vector<T> const & p2);

} // namespace LCM

#include "Geometry.i.cc"
#include "Geometry.t.cc"

#endif // LCM_Geometry_h
