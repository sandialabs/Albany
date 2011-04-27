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
  template<typename ScalarT>
  ScalarT
  length(Vector<ScalarT> const & p0, Vector<ScalarT> const & p1)
  {
    Vector<ScalarT> v = p1 - p0;
    return norm(v);
  }

  //
  // Area of a triangle
  //
  template<typename ScalarT>
  ScalarT
  area(Vector<ScalarT> const & p0, Vector<ScalarT> const & p1,
      Vector<ScalarT> const & p2)
  {
    Vector<ScalarT> u = p1 - p0;
    Vector<ScalarT> v = p2 - p0;
    ScalarT a = 0.5 * norm(cross(u,v));
    return a;
  }

  //
  // Area of a quadrilateral, assummed planar. If not planar, returns
  // the sum of the areas of the two triangles p0,p1,p2 and p0,p2,p3
  //
  template<typename ScalarT>
  ScalarT
  area(Vector<ScalarT> const & p0, Vector<ScalarT> const & p1,
      Vector<ScalarT> const & p2, Vector<ScalarT> const & p3)
  {
    return area(p0, p1, p2) + area(p0, p2, p3);
  }

  //
  // Volume of tetrahedron
  //
  template<typename ScalarT>
  ScalarT
  volume(Vector<ScalarT> const & p0, Vector<ScalarT> const & p1,
      Vector<ScalarT> const & p2, Vector<ScalarT> const & p3)
  {
    // Area of base triangle
    ScalarT A = area(p0, p1, p2);

    // Height
    Vector<ScalarT> u = p1 - p0;
    Vector<ScalarT> v = p2 - p0;
    Vector<ScalarT> n = cross(u, v);
    n = n / norm(n);
    Vector<ScalarT> w = p3 - p0;
    ScalarT h = fabs(dot(w, n));

    // Volume
    ScalarT V = A * h / 3.0;
    return V;
  }

  //
  // Volume of pyramid of quadrilateral base
  // Base is assumed planar
  // Base is p0,p1,p2,p3
  // Apex is p4
  //
  template<typename ScalarT>
  ScalarT
  volume(Vector<ScalarT> const & p0, Vector<ScalarT> const & p1,
      Vector<ScalarT> const & p2, Vector<ScalarT> const & p3,
      Vector<ScalarT> const & p4)
  {
    // Area of base quadrilateral
    ScalarT A = area(p0, p1, p2, p3);

    // Height
    Vector<ScalarT> u = p1 - p0;
    Vector<ScalarT> v = p2 - p0;
    Vector<ScalarT> n = cross(u, v);
    n = n / norm(n);
    Vector<ScalarT> w = p4 - p0;
    ScalarT h = fabs(dot(w, n));

    // Volume
    ScalarT V = A * h / 3.0;
    return V;
  }

  //
  // Volume of hexahedron
  // Assumption: all faces are planar
  // Decompose into 3 pyramids
  //
  template<typename ScalarT>
  ScalarT
  volume(Vector<ScalarT> const & p0, Vector<ScalarT> const & p1,
      Vector<ScalarT> const & p2, Vector<ScalarT> const & p3,
      Vector<ScalarT> const & p4, Vector<ScalarT> const & p5,
      Vector<ScalarT> const & p6, Vector<ScalarT> const & p7)
  {
    // 1st pyramid
    ScalarT V1 = volume(p4, p7, p6, p5, p0);

    // 2nd pyramid
    ScalarT V2 = volume(p3, p2, p6, p7, p0);

    // 3rd pyramid
    ScalarT V3 = volume(p1, p5, p6, p2, p0);

    return V1 + V2 + V3;
  }

  //
  // Centroids of segment, triangle, tetrahedron, quadrilateral
  // and hexahedron
  // For these we can just take the average of the vertices
  //
  template<typename ScalarT>
  Vector<ScalarT>
  centroid(std::vector<Vector<ScalarT> > const & points)
  {
    Vector<ScalarT> C(0.0, 0.0, 0.0);
    typedef typename std::vector<Vector<ScalarT> >::size_type sizeT;
    sizeT n = points.size();

    for (sizeT i = 0; i < n; ++i) {
      C += points[i];
    }
    return C / static_cast<double>(n);
  }

} // namespace LCM

#endif // LCM_Geometry_t_cc
