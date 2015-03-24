//$ mex -largeArrayDims CXXFLAGS="\$CXXFLAGS -fopenmp -Wall -O3"
//$ -I. LDFLAGS="\$LDFLAGS -fopenmp -llapack -lmwblas" vtkmex.cpp

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>

typedef int Int;
typedef double Real;

// -----------------------------------------------------------------------------
// Topology.

template<typename T> class ArrayList {
public:
  ArrayList (const Int n) : n_(n) {}
  ArrayList (const Int n, const Int m) : n_(n), v_(m*n) {}
  const Int n () const { return n_; }
  void resize (const Int m) { v_.resize(m*n_); }
  void push_back (const Int* a) { v_.insert(v_.end(), a, a + n_); }
  T* operator() (Int i) { return &v_[0] + n_*i; }
  T* operator[] (Int i) { return &v_[0] + n_*i; }
  const T* operator() (Int i) const { return &v_[0] + n_*i; }
  const T* operator[] (Int i) const { return &v_[0] + n_*i; }
  size_t size () const { return v_.size() / n_; }
  bool empty () const { return size() == 0; }
private:
  Int n_;
  std::vector<T> v_;
};
typedef ArrayList<Int> IntArrayList;

struct Tets : public IntArrayList {
  Tets () : IntArrayList(4) {}
  Tets (const Int m) : IntArrayList(4, m) {}
};
struct Tris : public IntArrayList {
  Tris () : IntArrayList(3) {}
  Tris (const Int m) : IntArrayList(3, m) {}
};
struct Edges : public IntArrayList {
  Edges () : IntArrayList(2) {}
  Edges (const Int m) : IntArrayList(2, m) {}
};

// The signature is given by the sorted indices.
template<Int _n> class Signature {
public:
  Int poly_[_n], ps_[_n];
public:
  Signature (const Int tri[_n]) {
    for (Int i = 0; i < _n; ++i) {
      poly_[i] = tri[i];
      ps_[i] = tri[i];
    }
    std::sort(ps_, ps_ + _n);
  }
  const Int* poly () const { return poly_; }
  bool operator< (const Signature& ts) const {
    for (Int i = 0; i < _n; ++i) {
      if (ps_[i] < ts.ps_[i]) return true;
      else if (ps_[i] > ts.ps_[i]) return false;
    }
    return false;
  }
  bool operator== (const Signature& ts) const {
    for (Int i = 0; i < _n; ++i)
      if (ps_[i] != ts.ps_[i]) return false;
    return true;
  }
  bool operator!= (const Signature& ts) const { return ! operator==(ts); }
};
typedef Signature<2> EdgeSignature;
typedef Signature<3> TriSignature;

inline std::vector<EdgeSignature> get_all_edgesigs (const Tris& tris) {
  const Int idxs[3][2] = {{0,1}, {1,2}, {2,0}};
  std::vector<EdgeSignature> sts;
  for (size_t itri = 0, k = 0; itri < tris.size(); ++itri) {
    const Int* tri = tris(itri);
    for (Int i = 0; i < 3; ++i, ++k) {
      Int edge[2];
      for (Int j = 0; j < 2; ++j)
        edge[j] = tri[idxs[i][j]];
      sts.push_back(EdgeSignature(edge));
    }
  }
  return sts;
}

inline std::vector<TriSignature> get_all_trisigs (const Tets& tets) {
  const Int idxs[4][3] = {{1,0,2}, {3,0,1}, {3,1,2}, {3,2,0}};
  std::vector<TriSignature> tss;
  for (size_t itet = 0, k = 0; itet < tets.size(); ++itet) {
    const Int* tet = tets(itet);
    for (Int i = 0; i < 4; ++i, ++k) {
      Int tri[3];
      for (Int j = 0; j < 3; ++j)
        tri[j] = tet[idxs[i][j]];
      tss.push_back(TriSignature(tri));
    }
  }
  return tss;
}

Tris get_skin (const Tets& tets) {
  // Find all triangles and get their signatures.
  std::vector<TriSignature> tss = get_all_trisigs(tets);
  // Sort by signature.
  std::sort(tss.begin(), tss.end());
  // If there are two identical sigs in a row, the triangle is not on the skin.
  Tris stris;
  for (size_t i = 0; i < tss.size(); )
    if (i == tss.size() - 1 || tss[i] != tss[i+1]) {
      stris.push_back(tss[i].poly());
      ++i;
    } else i += 2;
  return stris;
}

Edges get_unique_edges (const IntArrayList& polys, const IntArrayList& pis) {
  std::vector<EdgeSignature> ess;
  for (size_t i = 0; i < polys.size(); ++i)
    for (Int j = 0; j < pis.n(); ++j) {
      Int edge[2];
      edge[0] = polys(i)[pis(0)[j]-1];
      edge[1] = polys(i)[pis(1)[j]-1];
      ess.push_back(EdgeSignature(edge));
    }

  std::sort(ess.begin(), ess.end());

  Edges edges;
  for (size_t i = 0; i < ess.size(); ) {
    edges.push_back(ess[i].poly());
    i += (i == ess.size() - 1 || ess[i] != ess[i+1]) ? 1 : 2;
  }
  return edges;
}

// -----------------------------------------------------------------------------
// NLA.

// LAPACK and BLAS declarations.
typedef long long blas_int;
extern "C" void dgeqrf_(
  blas_int* m, blas_int* n, double* A, blas_int* lda, double* tau,
  double* work, blas_int* lwork, blas_int* info);
extern "C" void dorgqr_(
  blas_int* m, blas_int* n, blas_int* k, double* A, blas_int* lda, double* tau,
  double* work, blas_int* lwork, blas_int* info);
// Wrapper declarations.
template<typename T> void geqrf(
  blas_int m, blas_int n, T* A, blas_int lda, T* tau, T* work,
  blas_int lwork, blas_int& info);
template<typename T> void orgqr(
  blas_int m, blas_int n, blas_int k, T* A, blas_int lda, T* tau, T* work,
  blas_int lwork, blas_int& info);
// Specializations.
template<> inline void geqrf<double> (
  blas_int m, blas_int n, double* A, blas_int lda, double* tau, double* work,
  blas_int lwork, blas_int& info)
{ dgeqrf_(&m, &n, A, &lda, tau, work, &lwork, &info); }
template<> inline void orgqr<double> (
  blas_int m, blas_int n, blas_int k, double* A, blas_int lda, double* tau,
  double* work, blas_int lwork, blas_int& info)
{ dorgqr_(&m, &n, &k, A, &lda, tau, work, &lwork, &info); }

// -----------------------------------------------------------------------------
// Geometry.

typedef ArrayList<Real> RealArrayList;
typedef RealArrayList Vertices;

template<Int D> struct Dimensional { enum { dim = D }; };
template<Int D> struct Point : public Dimensional<D> {
  Real d[D];
  Point (const Real* data=0)
    { if (data) for (Int i = 0; i < D; ++i) d[i] = data[i]; }
};
template<Int D> struct Vector : public Point<D> {
  Vector (const Real* data=0) : Point<D>(data) {}
};
template<Int D> struct LineSegment : public Dimensional<D> {
  Point<D> p[2];
  LineSegment(const Point<D>& p1, const Point<D>& p2);
};
template<Int D> struct Tri : public Dimensional<D> {
  Point<D> p[3];
  Tri(const Point<D>& p1, const Point<D>& p2, const Point<D>& p3);
};
// A plane is given by { x : dot(n, x) = c }. The signed distance function is
//     signed_dist(x, plane) = dot(n, x) - c.
template<Int D> struct Plane : public Dimensional<D> { Vector<D> n; Real c; };

template<typename T> inline char sign (const T& v)
{ if (v >= 0) return 1; else return -1; }

// x *= a
template<Int D> inline void scale (Real x[D], const Real a)
{ for (Int i = 0; i < D; ++i) x[i] *= a; }
template<Int D> inline void scale (Point<D>& x, const Real a)
{ scale<D>(x.d, a); }

// z = y - x
template<Int D> inline void
subtract (const Real y[D], const Real x[D], Real z[D])
{ for (Int i = 0; i < D; ++i) z[i] = y[i] - x[i]; }
template<typename PointT1, typename PointT2> inline void
subtract (const PointT1& y, const PointT1& x, PointT2& z)
{ subtract<PointT1::dim>(y.d, x.d, z.d); }

// y = a x + b y
template<Int D> inline void
axpby (const Real a, const Real x[D], const Real b, Real y[D])
{ for (Int i = 0; i < D; ++i) y[i] = a*x[i] + b*y[i]; }
template<Int D> inline void
axpby (const Real a, const Point<D>& x, const Real b, Point<D>& y)
{ return axpby<D>(a, x.d, b, y.d); }

template<Int D> inline Real dot (const Real x[D], const Real y[D]) {
  Real a = 0;
  for (Int i = 0; i < D; ++i) a += x[i]*y[i];
  return a;
}
template<Int D> inline Real dot (const Point<D>& x, const Point<D>& y)
{ return dot<D>(x.d, y.d); }

template<Int D> inline Real norm2 (const Real x[D])
{ return dot<D>(x, x); }
template<Int D> inline Real norm2 (const Point<D>& v)
{ return dot<D>(v.d, v.d); }

// c = a x b
inline void cross (const Real a[3], const Real b[3], Real c[3]) {
  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0];
}
inline void cross (const Point<3>& a, const Point<3>& b, Point<3>& c)
{ cross(a.d, b.d, c.d); }

// v /= norm(v, 2)
template<Int D> inline void normalize (Point<D>& v)
{ scale(v, 1/std::sqrt(norm2(v))); }

inline void init_Point (Point<3>& p, const Real a, const Real b, const Real c)
{ p.d[0] = a; p.d[1] = b; p.d[2] = c; }
template<Int D> inline void init_Point (Point<D>& p, const Real* d)
{ for (Int i = 0; i < D; ++i) p.d[i] = d[i]; }

template<Int D> inline LineSegment<D>::
LineSegment (const Point<D>& p1, const Point<D>& p2) { p[0] = p1; p[1] = p2; }

template<Int D> inline Tri<D>::
Tri (const Point<D>& p1, const Point<D>& p2, const Point<D>& p3)
{ p[0] = p1; p[1] = p2; p[2] = p3; }

template<Int D> inline void
init_Vector (const Point<D>& from, const Point<D>& to, Vector<D>& v)
{ subtract(to, from, v); }

template<Int D> inline void
init_Plane (const Vector<D>& v, const Point<D>& p_on_plane, Plane<D>& p) {
  p.n = v;
  normalize(p.n);
  p.c = dot(p_on_plane, p.n);
}

template<Int D> inline Real dist2 (const Point<D>& x, const Point<D>& y) {
  Real a = 0;
  for (Int i = 0; i < D; ++i) {
    const Real d = x.d[i] - y.d[i];
    a += d*d;
  }
  return a;
}

Real dist2 (const Point<3>& p, const LineSegment<3>& ls) {
  // Solution to
  //     alpha* = arg min_alpha f(alpha),
  // where
  //     v = b - a
  //     r = a + alpha v
  //     f(alpha) = 1/2 norm2(r(alpha) - p).
  Vector<3> v1, v2;
  // v1 = p - a.
  init_Vector(ls.p[0], p, v1);
  // v2 = b - a.
  init_Vector(ls.p[0], ls.p[1], v2);
  const Real alpha_star = dot(v1, v2) / norm2(v2);
  if (alpha_star <= 0) return dist2(p, ls.p[0]);
  else if (alpha_star >= 1) return dist2(p, ls.p[1]);
  else {
    // v2 = a + alpha* (b - a), which is the projection of p onto the line
    // segment.
    axpby(1, ls.p[0], alpha_star, v2);
    // v1 = p - v2.
    init_Vector(v2, p, v1);
    return norm2(v1);
  }
}
inline Real dist2 (const LineSegment<3>& ls, const Point<3>& p)
{ return dist2(p, ls); }

Real signed_dist (const Point<3>& p, const Plane<3>& plane)
{ return dot(plane.n, p) - plane.c; }
inline Real signed_dist (const Plane<3>& plane, const Point<3>& p)
{ return signed_dist(p, plane); }

template<Int D> class TriWithData;

template<> class TriWithData<3> {
  Point<3> p_[3];    // Vertices.
  Plane<3> plane_;   // Plane of the triangle.
  Real Q_[6], R_[4]; // QR factorization for projecting.
  Point<3> center_;  // Center ...
  Real radius_;      // ... and radius of a bounding sphere.
public:
  TriWithData () {}
  TriWithData (const Tri<3>& t) { init(t); }
  void init (const Tri<3>& t) {
    // Vertices.
    for (Int i = 0; i < 3; ++i) p_[i] = t.p[i];
    // Plane.
    Vector<3> e1, e2, v;
    init_Vector(p_[0], p_[1], e1);
    init_Vector(p_[0], p_[2], e2);
    cross(e1, e2, v);
    init_Plane(v, p_[0], plane_);
    { // QR factorization.
      subtract<3>(p_[1].d, p_[0].d, Q_);
      subtract<3>(p_[2].d, p_[0].d, Q_ + 3);
      const blas_int lwork = 6;
      Real work[lwork], tau[6];
      blas_int info;
      // Factorize.
      geqrf(3, 2, Q_, 3, tau, work, lwork, info);
      // Get R.
      R_[0] = Q_[0]; R_[1] = 0; R_[2] = Q_[3]; R_[3] = Q_[4];
      // Get Q.
      orgqr(3, 2, 2, Q_, 3, tau, work, lwork, info);
    }
    // Sphere.
    center_.d[0] = center_.d[1] = center_.d[2] = 0;
    for (Int i = 0; i < 3; ++i) axpby(1, p_[i], 1, center_);
    scale(center_, 1./3);
    radius_ = 0;
    for (Int i = 0; i < 3; ++i) {
      const Real d2 = dist2(center_, p_[i]);
      if (d2 > radius_) radius_ = d2;
    }
    radius_ = std::sqrt(radius_);
  }
  const Point<3>& vertex (const Int i) const { return p_[i]; }
  const Plane<3>& plane () const { return plane_; }
  bool is_projected_in (const Point<3>& p) const {
    Vector<3> v;
    init_Vector(p_[0], p, v);
    // x = Q' v.
    Real x[3];
    x[0] = dot<3>(Q_, v);
    x[1] = dot<3>(Q_ + 3, v);
    // x = R \ x.
    x[1] /= R_[3];
    x[0] = (x[0] - R_[2]*x[1]) / R_[0];
    // Complete the barycentric coordinates of the projected point.
    x[2] = 1 - x[0] - x[1];
#define in01(u) (0 <= (u) && (u) <= 1)
    return in01(x[0]) && in01(x[1]) && in01(x[2]);
#undef in01
  }
  Real bv_dist (const Point<3>& p) const {
    Vector<3> v;
    init_Vector(center_, p, v);
    return std::max(0.0, std::sqrt(norm2(v)) - radius_);
  }
};

inline Real signed_dist (const Point<3>& p, const TriWithData<3>& t) {
  //todo Could reduce number of ops.
  const Real sd = signed_dist(p, t.plane());
  if (t.is_projected_in(p))
    return sd;
  else {
    Real ls_dist2;
    for (Int i = 0; i < 3; ++i) {
      const Real
        lsd2i = dist2(p, LineSegment<3>(t.vertex(i), t.vertex((i+1) % 3)));
      if (i == 0 || lsd2i < ls_dist2) ls_dist2 = lsd2i;
    }
    return sign(sd)*std::sqrt(ls_dist2);
  }
}
inline Real signed_dist (const TriWithData<3>& t, const Point<3>& p)
{ return signed_dist(p, t); }

// For each point, find the smallest distance to the set of triangles (probably
// representing a skin) and the nearest triangle.
//   A typical application is to find this distance between two skins, where ps
// is the vertices of one skin and tris is the tris of the other. One can then
// compute a skin-distance norm.
//   Eventually I'd like to use a bounding volume hierarchy, at least with
// spheres, to do this calculation. For now I brute force it.
void dist (const Vertices& ps, const Tris& tris, const Vertices& xs,
           Real* d, Int* tri_idxs) {
  if (tris.empty()) return;

  std::vector< TriWithData<3> > tris_wd(tris.size());
  for (size_t i = 0; i < tris_wd.size(); ++i)
    tris_wd[i].init(Tri<3>(xs[tris(i)[0]], xs[tris(i)[1]], xs[tris(i)[2]]));

  for (size_t ip = 0; ip < ps.size(); ++ip) {
    const Point<3>& p(ps(ip));
    double min_dist = signed_dist(p, tris_wd[0]);
    Int min_idx = 0;
    for (size_t it = 1; it < tris_wd.size(); ++it) {
      // I'm not using a tree yet, but at least do a quick check.
      const Real bv_dist = tris_wd[it].bv_dist(p);
      if (bv_dist >= std::abs(min_dist)) continue;
      const Real sd = signed_dist(p, tris_wd[it]);
      if (std::abs(sd) < std::abs(min_dist)) {
        min_dist = sd;
        min_idx = it;
      }
    }
    d[ip] = min_dist;
    tri_idxs[ip] = min_idx;
  }
}

// -----------------------------------------------------------------------------
// Mex.

#include "mexutil.hpp"

template<typename T> void
convert (const mexutil::ConstDenseMexMat& mp, ArrayList<T>& p, const T add=0) {
  p.resize(mp.m);
  for (size_t i = 0; i < mp.m; ++i)
    for (int j = 0; j < p.n(); ++j)
      p(i)[j] = static_cast<T>(mp.a[mp.m*j + i]) + add;
}

template<typename T> mexutil::DenseMexMat
convert (const ArrayList<T>& p, const double add=0) {
  mexutil::DenseMexMat mp(p.size(), p.n());
  for (size_t i = 0; i < mp.m; ++i)
    for (int j = 0; j < p.n(); ++j)
      mp[mp.m*j + i] = static_cast<double>(p(i)[j]) + add;
  return mp;
}

void mexFunction (int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  std::string cmd = mexutil::init_mex(nrhs, prhs);
  if (cmd == "get_skin") {
    if (nlhs != 1 || nrhs != 1)
      mexErrMsgTxt("skin_tris = get_skin(tets)");
    const mexutil::ConstDenseMexMat mtets(prhs[0]);
    reqorexit(mtets.n == 4);
    Tets tets; convert(mtets, tets);
    const Tris stris = get_skin(tets);
    mexutil::DenseMexMat mstris = convert(stris);
    plhs[0] = mstris.ma;
  } else if (cmd == "get_unique_edges") {
    if (nlhs != 1 || nrhs != 2)
      mexErrMsgTxt("lc = get_unique_edges(c, is)");
    const mexutil::ConstDenseMexMat mc(prhs[0]), mis(prhs[1]);
    reqorexit(mis.m == 2);
    IntArrayList polys(mc.n); convert(mc, polys);
    IntArrayList pis(mis.n); convert(mis, pis);
    Edges edges = get_unique_edges(polys, pis);
    mexutil::DenseMexMat medges = convert(edges);
    plhs[0] = medges.ma;
  } else if (cmd == "signed_dist_ps_tris") {
    if (nlhs != 2 || nrhs != 3)
      mexErrMsgTxt("[sd idx] = signed_dist_ps_tris(points, tris, verts)");
    const mexutil::ConstDenseMexMat
      mps(prhs[0]), mtris(prhs[1]), mverts(prhs[2]);
    reqorexit(mps.n == 3); reqorexit(mtris.n == 3); reqorexit(mverts.n == 3);
    Vertices ps(3); convert(mps, ps);
    Tris tris(mtris.m); convert(mtris, tris, -1);
    Vertices verts(3); convert(mverts, verts);
    mexutil::DenseMexMat sd(mps.m, 1);
    plhs[0] = sd.ma;
    std::vector<Int> tri_idxs(mps.m);
    dist(ps, tris, verts, sd.a, &tri_idxs[0]);
    mexutil::DenseMexMat idxs(mps.m, 1);
    plhs[1] = idxs.ma;
    for (size_t i = 0; i < mps.m; ++i)
      idxs.a[i] = static_cast<double>(tri_idxs[i]) + 1;
  } else if (cmd == "dist2_ps_ls") {
    if (nlhs != 1 || nrhs != 2)
      mexErrMsgTxt("dist2 = dist2_ps_ls(points, line_segment)");
    const mexutil::ConstDenseMexMat mps(prhs[0]), mls(prhs[1]);
    reqorexit(mps.n == 3);
    reqorexit(mls.m == 2 && mls.n == 3);
    Vertices vps(3); convert(mps, vps);
    Vertices vls(3); convert(mls, vls);
    mexutil::DenseMexMat d2(mps.m, 1);
    plhs[0] = d2.ma;
    LineSegment<3> ls(Point<3>(vls(0)), Point<3>(vls(1)));
    for (size_t i = 0; i < mps.m; ++i) {
      Point<3> p(vps(i));
      d2.a[i] = dist2(p, ls);
    }
  } else if (cmd == "signed_dist_ps_tri") {
    if (nlhs != 1 || nrhs != 2)
      mexErrMsgTxt("dist2 = signed_dist_ps_tri(points, tri_verts)");
    const mexutil::ConstDenseMexMat mps(prhs[0]), mtri(prhs[1]);
    reqorexit(mps.n == 3);
    reqorexit(mtri.m == 3 && mtri.n == 3);
    Vertices vps(3); convert(mps, vps);
    Vertices vtri(3); convert(mtri, vtri);
    mexutil::DenseMexMat d(mps.m, 1);
    plhs[0] = d.ma;
    Tri<3> tri(Point<3>(vtri(0)), Point<3>(vtri(1)), Point<3>(vtri(2)));
    for (size_t i = 0; i < mps.m; ++i) {
      Point<3> p(vps(i));
      d.a[i] = signed_dist(p, tri);
    }
  } else {
    mexErrMsgTxt((std::string("Invalid function: ") + cmd).c_str());
  }
}
