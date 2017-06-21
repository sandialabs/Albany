// The usual mex file utils.
// Original. 15 Sep 2014. AMB ambradl@sandia.gov

#ifndef INCLUDE_MEXUTIL
#define INCLUDE_MEXUTIL

#include <string.h>
#include <math.h>
#include <mex.h>
#include <omp.h>
#include <string>
#include <vector>
#include <exception>
#include <limits>

#define reqorexit(a)                                            \
  if (!(a)) {                                                   \
    fprintf(stdout, "%s (%d): " #a "\n", __FILE__, __LINE__);   \
    mexErrMsgTxt("");                                           \
  } while (0)

namespace mexutil {
using namespace std;

template<typename T> inline T& inout (T& a) { return a; }
template<typename T> inline T& out (T& a) { return a; }

class Exception {
public:
  Exception (const string& msg = "e") : msg(msg) {}
  virtual ~Exception () {}
  const string& get_msg () const { return msg; }
private:
  string msg;
};

struct BaseMexMat {
  const size_t m, n;
  BaseMexMat (const mxArray* ma)
    : m(ma ? mxGetM(ma) : 0), n(ma ? mxGetN(ma) : 0) {}
  BaseMexMat (size_t m, size_t n) : m(m), n(n) {}
  virtual ~BaseMexMat () {}
  size_t numel () const { return m*n; }
};
struct ConstDenseMexMat : public BaseMexMat {
  const double* a;
  ConstDenseMexMat (const mxArray* ma)
    : BaseMexMat(ma), a(ma ? mxGetPr(ma) : NULL) {}
  const double& operator[] (const size_t i) const { return a[i]; }
};
struct DenseMexMatRef : public BaseMexMat {
  double* a;
  mxArray* ma;
  DenseMexMatRef (mxArray* ma)
    : BaseMexMat(ma), a(ma ? mxGetPr(ma) : NULL), ma(ma) {}
  const double& operator[] (const size_t i) const { return a[i]; }
  double& operator[] (const size_t i) { return a[i]; }
};
struct DenseMexMat : public DenseMexMatRef {
  DenseMexMat (size_t m, size_t n)
    : DenseMexMatRef(mxCreateDoubleMatrix(m, n, mxREAL)) {}
  void free () {
    mxDestroyArray(ma);
    ma = 0; a = 0;
  }
};

typedef long long int blas_int;
extern "C" {
  void dgemv_(
    const char*, const blas_int*, const blas_int*, const double*, const double*,
    const blas_int*, const double*, const blas_int*, const double*, double*,
    const blas_int*);
  double ddot_(
    const blas_int*, const double*, const blas_int*, const double*,
    const blas_int*);
  void daxpy_(
    const blas_int*, double*, const double*, const blas_int*, double*,
    const blas_int*);
}

inline void
gemv (const double* A, const blas_int m, const blas_int n,
      const double* x, const double alpha, const double beta, double* y) {
  const blas_int inc = 1;
  const char trans = 'n';
  dgemv_(&trans, &m, &n, &alpha, A, &m, x, &inc, &beta, y, &inc);
}
inline double dot (
  blas_int n, const double* x, blas_int incx, const double* y, blas_int incy)
{ return ddot_(&n, x, &incx, y, &incy); }
inline void axpy (
  blas_int n, double a, const double* x, blas_int incx, double* y,
  blas_int incy)
{ daxpy_(&n, &a, x, &incx, y, &incy); }

// No error checking.
inline double dot (const vector<double>& x, const vector<double>& y) {
  return dot(x.size(), &x[0], 1, &y[0], 1);
}
inline vector<double>& axpy (
  const double a, const vector<double>& x, vector<double>& y)
{
  axpy(x.size(), a, &x[0], 1, &y[0], 1);
  return y;
}

inline int numel (const mxArray* ma) { return mxGetNumberOfElements(ma); }

inline bool get_stringv (const mxArray* ms, vector<char>& s) {
  int strlen = mxGetNumberOfElements(ms) + 1;
  s.resize(strlen);
  if (mxGetString(ms, &s[0], strlen) != 0) return false;
  return true;
}
inline bool get_string (const mxArray* ms, string& s) {
  vector<char> vs;
  if (!get_stringv(ms, vs)) return false;
  s = string(&vs[0]);
  return true;
}

inline double get_scalar (const mxArray* ma) {
  ConstDenseMexMat a(ma);
  reqorexit(a.m == 1 && a.n == 1);
  return a.a[0];
}

inline bool are_same_array (const mxArray* ma, const mxArray* mb) {
  return mxGetPr(ma) == mxGetPr(mb);
}

string init_mex (int& nrhs, const mxArray**& prhs) {
  if (nrhs == 0) mexErrMsgTxt("Missing cmd.");
  string cmd;
  if (!get_string(prhs[0], cmd)) mexErrMsgTxt("First arg must be function.");
  --nrhs;
  ++prhs;
  return cmd;
}
} // mexutil

#endif
