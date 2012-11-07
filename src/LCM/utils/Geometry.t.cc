//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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
  // Area of a quadrilateral, assumed planar. If not planar, returns
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
    return C / static_cast<T>(n);
  }

  //
  // The surface normal of a face
  // Input: 3 independent nodes on the face
  // Output: unit normal vector
  //
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

  //
  // Given 3 points p0, p1, p2 that define a plane
  // determine if point p is in the same side of the normal
  // to the plane as defined by the right hand rule.
  //
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

    if (s < 0.0) return false;

    return true;
  }

  //
  // Given two iterators to a container of points,
  // find the associated bounding box.
  // \param start, end: define sequence of points
  // \return vectors that define the bounding box
  //
  template<typename T, typename I>
  std::pair< Vector<T>, Vector<T> >
  bounding_box(I start, I end)
  {
    I
    it = start;

    LCM::Vector<T>
    min = (*it);

    LCM::Vector<T>
    max = min;

    const Index
    N = min.get_dimension();

    ++it;

    for (; it != end; ++it) {

      LCM::Vector<T> const &
      point = (*it);

      for (Index i = 0; i < N; ++i) {
        const T s = point(i);
        if (s < min(i)) min(i) = s;
        if (s > max(i)) max(i) = s;
      }

    }

    return std::make_pair(min, max);
  }

  //
  // Determine if a given point is inside a bounding box.
  // \param p the point
  // \param min, max points defining the box
  // \return whether the point is inside
  //
  template<typename T>
  bool
  in_box(
      Vector<T> const & p,
      Vector<T> const & min,
      Vector<T> const & max)
  {
    const Index
    N = p.get_dimension();

    assert(min.get_dimension() == N);
    assert(max.get_dimension() == N);

    for (Index i = 0; i < N; ++i) {
      T const & s = p(i);
      if (s < min(i)) return false;
      if (s > max(i)) return false;
    }

    return true;
  }

  //
  // Generate random point inside bounding box
  // \param min, max the bounding box
  // \return p point inside box
  //
  template<typename T>
  Vector<T>
  random_in_box(Vector<T> const & min, Vector<T> const & max)
  {
    const Index
    N = min.get_dimension();

    assert(max.get_dimension() == N);

    Vector<T> p(N);

    for (Index i = 0; i < N; ++i) {
      p(i) = (max(i) - min(i)) * T(std::rand())/T(RAND_MAX) + min(i);
    }

    return p;
  }

  //
  // Given 4 points p0, p1, p2, p3 that define a tetrahedron
  // determine if point p is inside it.
  //
  template<typename T>
  bool
  in_tetrahedron(
      Vector<T> const & p,
      Vector<T> const & p0,
      Vector<T> const & p1,
      Vector<T> const & p2,
      Vector<T> const & p3)
  {
    if (in_normal_side(p, p0, p1, p2) == false) {

      return false;

    } else if (in_normal_side(p, p0, p3, p1) == false) {

      return false;

    } else if (in_normal_side(p, p1, p3, p2) == false) {

      return false;

    } else if (in_normal_side(p, p2, p3, p0) == false) {

      return false;

    }

    return true;
  }

  //
  // Given 8 points that define a hexahedron
  // determine if point p is inside it.
  // Assumption: faces are planar
  //
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
      Vector<T> const & p7)
  {
    if (in_normal_side(p, p0, p1, p2) == false) {

      return false;

    } else if (in_normal_side(p, p0, p4, p5) == false) {

      return false;

    } else if (in_normal_side(p, p1, p5, p6) == false) {

      return false;

    } else if (in_normal_side(p, p2, p6, p7) == false) {

      return false;

    } else if (in_normal_side(p, p3, p7, p4) == false) {

      return false;

    } else if (in_normal_side(p, p4, p7, p6) == false) {

      return false;

    }

    return true;
  }

  //
  // Closest point
  // \param p the point
  // \param n vector of points to test
  // \return index to closest point
  //
  template<typename T>
  typename std::vector< Vector<T> >::size_type
  closest_point(Vector<T> const & p, std::vector< Vector<T> > const & n)
  {
    assert(n.size() > 0);

    typename std::vector< Vector<T> >::size_type
    index = 0;

    T
    min = norm(p - n[0]);

    for (typename std::vector< Vector<T> >::size_type i = 1;
        i < n.size();
        ++i) {

      const T
      s = norm(p - n[i]);

      if (s < min) {
        min = s;
        index = i;
      }

    }

    return index;
  }

  // Median of a sequence defined by random
  // access iterators. Undefined for empty set.
  // \param begin, end Iterators that define the sequence
  // \return median of sequence
  //
  template<typename T, typename Iterator>
  T
  median(Iterator begin, Iterator end)
  {
    // Firewall
    if (begin == end) {
      std::cerr << "ERROR: Median undefined for empty set." << std::endl;
      exit(1);
    }

    Index
    size = end - begin;

    T
    median;

    Index
    mid_index = size / 2;

    Iterator
    mid_iterator = begin + mid_index;

    std::nth_element(begin, mid_iterator, end);

    if (size % 2 == 0) {

      // Even number of elements
      T
      b = *mid_iterator;

      Iterator
      previous = mid_iterator - 1;

      T
      a = *previous;

      median = (a + b) / 2.0;

    } else {

      // Odd number of elements
      median = *mid_iterator;

    }

    return median;

  }

} // namespace LCM

#endif // LCM_Geometry_t_cc
