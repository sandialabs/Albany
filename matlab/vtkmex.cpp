//$ mex -largeArrayDims CXXFLAGS="\$CXXFLAGS -fopenmp -Wall -O3"
//$ -I. LDFLAGS="\$LDFLAGS -fopenmp -llapack -lmwblas" vtkmex.cpp

#include <vector>
#include <iostream>
#include <algorithm>

// -----------------------------------------------------------------------------
// Topology.

class IntArrayList {
public:
  IntArrayList (const int n) : n_(n) {}
  IntArrayList (const int n, const int m) : n_(n), v_(m*n) {}
  const int n () const { return n_; }
  void resize (const int m) { v_.resize(m*n_); }
  void push_back (const int* a) { v_.insert(v_.end(), a, a + n_); }
  int* operator() (int i) { return &v_[0] + n_*i; }
  int* operator[] (int i) { return &v_[0] + n_*i; }
  const int* operator() (int i) const { return &v_[0] + n_*i; }
  size_t size () const { return v_.size() / n_; }
private:
  int n_;
  std::vector<int> v_;
};
struct Tets : public IntArrayList {
  Tets () : IntArrayList(4) {}
  Tets (const int m) : IntArrayList(4, m) {}
};
struct Tris : public IntArrayList {
  Tris () : IntArrayList(3) {}
  Tris (const int m) : IntArrayList(3, m) {}
};
struct Edges : public IntArrayList {
  Edges () : IntArrayList(2) {}
  Edges (const int m) : IntArrayList(2, m) {}
};

// The signature is given by the sorted indices.
template<int _n> class Signature {
public:
  int poly_[_n], ps_[_n];
public:
  Signature (const int tri[_n]) {
    for (int i = 0; i < _n; ++i) {
      poly_[i] = tri[i];
      ps_[i] = tri[i];
    }
    std::sort(ps_, ps_ + _n);
  }
  const int* poly () const { return poly_; }
  bool operator< (const Signature& ts) const {
    for (int i = 0; i < _n; ++i) {
      if (ps_[i] < ts.ps_[i]) return true;
      else if (ps_[i] > ts.ps_[i]) return false;
    }
    return false;
  }
  bool operator== (const Signature& ts) const {
    for (int i = 0; i < _n; ++i)
      if (ps_[i] != ts.ps_[i]) return false;
    return true;
  }
  bool operator!= (const Signature& ts) const { return ! operator==(ts); }
};
typedef Signature<2> EdgeSignature;
typedef Signature<3> TriSignature;

inline std::vector<EdgeSignature> get_all_edgesigs (const Tris& tris) {
  const int idxs[3][2] = {{0,1}, {1,2}, {2,0}};
  std::vector<EdgeSignature> sts;
  for (size_t itri = 0, k = 0; itri < tris.size(); ++itri) {
    const int* tri = tris(itri);
    for (int i = 0; i < 3; ++i, ++k) {
      int edge[2];
      for (int j = 0; j < 2; ++j)
        edge[j] = tri[idxs[i][j]];
      sts.push_back(EdgeSignature(edge));
    }
  }
  return sts;
}

inline std::vector<TriSignature> get_all_trisigs (const Tets& tets) {
  const int idxs[4][3] = {{0,1,2}, {0,1,3}, {1,2,3}, {2,0,3}};
  std::vector<TriSignature> tss;
  for (size_t itet = 0, k = 0; itet < tets.size(); ++itet) {
    const int* tet = tets(itet);
    for (int i = 0; i < 4; ++i, ++k) {
      int tri[3];
      for (int j = 0; j < 3; ++j)
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
    for (int j = 0; j < pis.n(); ++j) {
      int edge[2];
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
// Geometry.

template<int _d> struct Dimensional { enum { dim = _d }; };
template<int _d> struct Point : public Dimensional<_d> { double d[_d]; };
template<int _d> struct Vector : public Dimensional<_d> { double d[_d]; };
template<int _d> struct LineSeg : public Dimensional<_d> { Point<_d> p[2]; };
template<int _d> struct Tri : public Dimensional<_d> { Point<_d> p[3]; };
template<int _d> struct Plane : public Dimensional<_d> { Vector<_d> n; };

template<int _d> inline void scale (double x[_d], const double a)
{ for (int i = 0; i < _d; ++i) x[i] *= a; }

template<int _d> inline void
subtract (const double x[_d], const double y[_d], double z[_d])
{ for (int i = 0; i < _d; ++i) z[i] = x[i] - y[i]; }
template<typename PointT1, typename PointT2> inline void
subtract (const PointT1& x, const PointT1& y, PointT2& z)
{ subtract(x.d, y.d, z.d); }

template<int _d> inline double dot (const double x[_d], const double y[_d]) {
  double a = 0;
  for (int i = 0; i < _d; ++i) a += x[i]*y[i];
  return a;
}
template<int _d> inline double dot (const Vector<_d>& x, const Vector<_d>& y)
{ return dot(x.d, y.d); }
template<int _d> inline double norm2 (const double x[_d])
{ return dot<_d>(x, x); }

// c = a x b
inline void cross (const double a[3], const double b[3], double c[3]) {
  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0];
}
inline void cross (const Vector<3>& a, const Vector<3>& b, Vector<3>& c)
{ cross(a.d, b.d, c.d); }

template<int _d> inline void normalize (Vector<_d>& v)
{ scale(v, 1/sqrt(norm2(v))); }

template<int _d> inline double dist2 (const Point<_d>& x, const Point<_d>& y) {
  double a = 0;
  for (int i = 0; i < _d; ++i) {
    const double d = x.d[i] - y.d[i];
    a += d*d;
  }
  return a;
}

double dist2 (const Point<3>& p, const LineSeg<3>& ls) {

}
inline double dist2 (const LineSeg<3>& ls, const Point<3>& p)
{ return dist2(p, ls); }

double dist2 (const Point<3>& p, const Tri<3>& t) {

}
inline double dist2 (const Tri<3>& t, const Point<3>& p) { return dist2(p, t); }

double dist2 (const Point<3>& p, const Plane<3>& plane) {

}
inline double dist2 (const Plane<3>& plane, const Point<3>& p)
{ return dist2(p, plane); }

template<int _d> inline void init_Plane (const Vector<_d>& v, Plane<_d>& p)
{ p.n = v; normalize(p.n); }

template<int _d> inline void
init_Vector (const Point<_d>& from, const Point<_d>& to, Vector<_d>& v)
{ subtract(to, from, v.d); }

// -----------------------------------------------------------------------------
// Mex.

#include "mexutil.hpp"
using namespace mexutil;

void convert (const ConstDenseMexMat& mp, IntArrayList& p) {
  p.resize(mp.m);
  for (size_t i = 0; i < mp.m; ++i)
    for (int j = 0; j < p.n(); ++j)
      p(i)[j] = mp.a[mp.m*j + i];
}

DenseMexMat convert (const IntArrayList& p) {
  DenseMexMat mp(p.size(), p.n());
  for (size_t i = 0; i < mp.m; ++i)
    for (int j = 0; j < p.n(); ++j)
      mp[mp.m*j + i] = p(i)[j];
  return mp;
}

void mexFunction (int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  string cmd = init_mex(inout(nrhs), inout(prhs));
  if (cmd == "get_skin") {
    if (nlhs != 1 || nrhs != 1)
      mexErrMsgTxt("skin_tris = get_skin(tets)");
    const ConstDenseMexMat mtets(prhs[0]);
    reqorexit(mtets.n == 4);
    Tets tets; convert(mtets, tets);
    const Tris stris = get_skin(tets);
    DenseMexMat mstris = convert(stris);
    plhs[0] = mstris.ma;
  } else if (cmd == "get_unique_edges") {
    if (nlhs != 1 || nrhs != 2)
      mexErrMsgTxt("lc = get_lines_for_shape(c, is)");
    const ConstDenseMexMat mc(prhs[0]), mis(prhs[1]);
    reqorexit(mis.m == 2);
    IntArrayList polys(mc.n); convert(mc, polys);
    IntArrayList pis(mis.n); convert(mis, pis);
    Edges edges = get_unique_edges(polys, pis);
    DenseMexMat medges = convert(edges);
    plhs[0] = medges.ma;
  } else {
    mexErrMsgTxt((string("Invalid function: ") + cmd).c_str());
  }
}
