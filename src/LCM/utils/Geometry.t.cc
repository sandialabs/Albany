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
  length(Vector<T, 3> const & p0, Vector<T, 3> const & p1)
  {
    Vector<T, 3> v = p1 - p0;
    return norm(v);
  }

  //
  // Area of a triangle
  //
  template<typename T>
  T
  area(Vector<T, 3> const & p0, Vector<T, 3> const & p1,
      Vector<T, 3> const & p2)
  {
    Vector<T, 3> u = p1 - p0;
    Vector<T, 3> v = p2 - p0;
    T a = 0.5 * norm(cross(u,v));
    return a;
  }

  //
  // Area of a quadrilateral, assummed planar. If not planar, returns
  // the sum of the areas of the two triangles p0,p1,p2 and p0,p2,p3
  //
  template<typename T>
  T
  area(Vector<T, 3> const & p0, Vector<T, 3> const & p1,
      Vector<T, 3> const & p2, Vector<T, 3> const & p3)
  {
    return area(p0, p1, p2) + area(p0, p2, p3);
  }

  //
  // Volume of tetrahedron
  //
  template<typename T>
  T
  volume(Vector<T, 3> const & p0, Vector<T, 3> const & p1,
      Vector<T, 3> const & p2, Vector<T, 3> const & p3)
  {
    // Area of base triangle
    T A = area(p0, p1, p2);

    // Height
    Vector<T, 3> u = p1 - p0;
    Vector<T, 3> v = p2 - p0;
    Vector<T, 3> n = cross(u, v);
    n = n / norm(n);
    Vector<T, 3> w = p3 - p0;
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
  volume(Vector<T, 3> const & p0, Vector<T, 3> const & p1,
      Vector<T, 3> const & p2, Vector<T, 3> const & p3,
      Vector<T, 3> const & p4)
  {
    // Area of base quadrilateral
    T A = area(p0, p1, p2, p3);

    // Height
    Vector<T, 3> u = p1 - p0;
    Vector<T, 3> v = p2 - p0;
    Vector<T, 3> n = cross(u, v);
    n = n / norm(n);
    Vector<T, 3> w = p4 - p0;
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
  volume(Vector<T, 3> const & p0, Vector<T, 3> const & p1,
      Vector<T, 3> const & p2, Vector<T, 3> const & p3,
      Vector<T, 3> const & p4, Vector<T, 3> const & p5,
      Vector<T, 3> const & p6, Vector<T, 3> const & p7)
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
  Vector<T, 3>
  centroid(std::vector<Vector<T, 3> > const & points)
  {
    Vector<T, 3> C(0.0, 0.0, 0.0);
    typedef typename std::vector<Vector<T, 3> >::size_type sizeT;
    sizeT n = points.size();

    for (sizeT i = 0; i < n; ++i) {
      C += points[i];
    }
    return C / static_cast<double>(n);
  }

  ///
  /// The surface normal of a face
  /// Assumption: face is planar
  /// Input: 3 independent nodes on the face
  /// Output: normal vector
  ///
  template<typename T>
  Vector<T,3>
  faceNormal(Vector<T,3> const & p0,
          Vector<T,3> const & p1,
          Vector<T,3> const & p2)
  {
      // Construct 2 independent vectors
      Vector<T,3> v0 = p1 - p0;
      Vector<T,3> v1 = p2 - p0;

      Vector<T,3> n(0.0, 0.0, 0.0);
      n = LCM::cross(v0,v1);
      return n;
  }

} // namespace LCM

#endif // LCM_Geometry_t_cc
