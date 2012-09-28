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

  ///
  /// Given two iterators to a container of points,
  /// find the associated bounding box.
  /// \param start, end: define sequence of points
  /// \return vectors that define the bounding box
  ///
  template<typename T, typename I>
  std::pair< Vector<T>, Vector<T> >
  bounding_box(I start, I end);

  ///
  /// Determine if a given point is inside a bounding box.
  /// \param p the point
  /// \param min, max points defining the box
  /// \return whether the point is inside
  ///
  template<typename T>
  bool
  in_box(
      Vector<T> const & p,
      Vector<T> const & min,
      Vector<T> const & max);

  ///
  /// Generate random point inside bounding box
  /// \param min, max the bounding box
  /// \return p point inside box
  ///
  template<typename T>
  Vector<T>
  random_in_box(
      Vector<T> const & min,
      Vector<T> const & max);

  ///
  /// Given 4 points p0, p1, p2, p3 that define a tetrahedron
  /// determine if point p is inside it.
  ///
  template<typename T>
  bool
  in_tetrahedron(
      Vector<T> const & p,
      Vector<T> const & p0,
      Vector<T> const & p1,
      Vector<T> const & p2,
      Vector<T> const & p3);

  ///
  /// Given 8 points that define a hexahedron
  /// determine if point p is inside it.
  /// Assumption: faces are planar
  ///
  template<typename T>
  bool
  in_hexahedron(
      Vector<T> const & p,
      Vector<T> const & p0,
      Vector<T> const & p1,
      Vector<T> const & p2,
      Vector<T> const & p3,
      Vector<T> const & p4,
      Vector<T> const & p5,
      Vector<T> const & p6,
      Vector<T> const & p7);

  ///
  /// Closest point
  /// \param p the point
  /// \param n vector of points to test
  /// \return index to closest point
  ///
  template<typename T>
  typename std::vector< Vector<T> >::size_type
  closest_point(Vector<T> const & p, std::vector< Vector<T> > const & n);


} // namespace LCM

#include "Geometry.i.cc"
#include "Geometry.t.cc"

#endif // LCM_Geometry_h
