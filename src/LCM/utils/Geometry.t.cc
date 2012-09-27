///
/// \file Geometry.t.cc
/// Geometry utilities for LCM. Templates.
/// \author Alejandro Mota
///
#if !defined(LCM_Geometry_t_cc)
#define LCM_Geometry_t_cc


namespace LCM {

  //
  // Length of a segment
  //
  template<typename T>
  T
  length(Vector<T> const & p0, Vector<T> const & p1)
  {
    Vector<T> v = p1 - p0;
    return norm(v);
  }

  //
  // Area of a triangle
  //
  template<typename T>
  T
  area(Vector<T> const & p0, Vector<T> const & p1,
      Vector<T> const & p2)
  {
    Vector<T> u = p1 - p0;
    Vector<T> v = p2 - p0;
    T a = 0.5 * norm(cross(u,v));
    return a;
  }

  //
  // Area of a quadrilateral, assummed planar. If not planar, returns
  // the sum of the areas of the two triangles p0,p1,p2 and p0,p2,p3
  //
  template<typename T>
  T
  area(Vector<T> const & p0, Vector<T> const & p1,
      Vector<T> const & p2, Vector<T> const & p3)
  {
    return area(p0, p1, p2) + area(p0, p2, p3);
  }

  //
  // Volume of tetrahedron
  //
  template<typename T>
  T
  volume(Vector<T> const & p0, Vector<T> const & p1,
      Vector<T> const & p2, Vector<T> const & p3)
  {
    // Area of base triangle
    T A = area(p0, p1, p2);

    // Height
    Vector<T> u = p1 - p0;
    Vector<T> v = p2 - p0;
    Vector<T> n = cross(u, v);
    n = n / norm(n);
    Vector<T> w = p3 - p0;
    T h = fabs(dot(w, n));

    // Volume
    T V = A * h / 3.0;
    return V;
  }

  //
  // Volume of pyramid of quadrilateral base
  // Base is assumed planar
  // Base is p0,p1,p2,p3
  // Apex is p4
  //
  template<typename T>
  T
  volume(Vector<T> const & p0, Vector<T> const & p1,
      Vector<T> const & p2, Vector<T> const & p3,
      Vector<T> const & p4)
  {
    // Area of base quadrilateral
    T A = area(p0, p1, p2, p3);

    // Height
    Vector<T> u = p1 - p0;
    Vector<T> v = p2 - p0;
    Vector<T> n = cross(u, v);
    n = n / norm(n);
    Vector<T> w = p4 - p0;
    T h = fabs(dot(w, n));

    // Volume
    T V = A * h / 3.0;
    return V;
  }

  //
  // Volume of hexahedron
  // Assumption: all faces are planar
  // Decompose into 3 pyramids
  //
  template<typename T>
  T
  volume(Vector<T> const & p0, Vector<T> const & p1,
      Vector<T> const & p2, Vector<T> const & p3,
      Vector<T> const & p4, Vector<T> const & p5,
      Vector<T> const & p6, Vector<T> const & p7)
  {
    // 1st pyramid
    T V1 = volume(p4, p7, p6, p5, p0);

    // 2nd pyramid
    T V2 = volume(p3, p2, p6, p7, p0);

    // 3rd pyramid
    T V3 = volume(p1, p5, p6, p2, p0);

    return V1 + V2 + V3;
  }

  //
  // Centroids of segment, triangle, tetrahedron, quadrilateral
  // and hexahedron
  // For these we can just take the average of the vertices
  //
  template<typename T>
  Vector<T>
  centroid(std::vector<Vector<T> > const & points)
  {
    Vector<T> C(points[0].get_dimension());
    C.clear();
    typedef typename std::vector<Vector<T> >::size_type sizeT;
    sizeT n = points.size();

    for (sizeT i = 0; i < n; ++i) {
      C += points[i];
    }
    return C / static_cast<double>(n);
  }

  ///
  /// The surface normal of a face
  /// Input: 3 independent nodes on the face
  /// Output: unit normal vector
  ///
  template<typename T>
  Vector<T>
  normal(Vector<T> const & p0,
          Vector<T> const & p1,
          Vector<T> const & p2)
  {
      // Construct 2 independent vectors
      Vector<T> v0 = p1 - p0;
      Vector<T> v1 = p2 - p0;

      Vector<T> n = LCM::cross(v0,v1);
      n = n/LCM::norm(n);
      return n;
  }

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
      Vector<T> const & p2)
  {
    Vector<T> v0 = p1 - p0;
    Vector<T> v1 = p2 - p0;

    Vector<T> n = cross(v0, v1);
    Vector<T> v = p - p0;

    T s = dot(v, n);

    if (s <= 0.0) return false;

    return true;
  }

} // namespace LCM

#endif // LCM_Geometry_t_cc
