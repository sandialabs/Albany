//
// First cut of LCM small tensor utilities.
//
#if !defined(LCM_Tensor_h)
#define LCM_Tensor_h

#include <iostream>

namespace LCM {

  typedef unsigned int Index;
  const Index MaxDim = 3;

  //
  // Vector
  //
  template<typename ScalarT>
  class Vector
  {
  public:
    Vector();
    Vector(const ScalarT s);
    Vector(const ScalarT s0, const ScalarT s1, const ScalarT s2);
    Vector(Vector const & v);
    ~Vector();

    const ScalarT & operator()(const Index i) const;
    ScalarT & operator()(const Index i);

    Vector<ScalarT> & operator=  (Vector<ScalarT> const & v);
    Vector<ScalarT> & operator+= (Vector<ScalarT> const & v);
    Vector<ScalarT> & operator-= (Vector<ScalarT> const & v);

    void clear();

  private:
    ScalarT e[MaxDim];
  };

  //
  // Second order tensor
  //
  template<typename ScalarT>
  class Tensor
  {
  public:
    Tensor();
    Tensor(const ScalarT s);
    Tensor(
        const ScalarT s00, const ScalarT s01, const ScalarT s02,
        const ScalarT s10, const ScalarT s11, const ScalarT s12,
        const ScalarT s20, const ScalarT s21, const ScalarT s22);
    Tensor(const Tensor & A);
    ~Tensor();

    const ScalarT & operator()(const Index i, const Index j) const;
    ScalarT & operator()(const Index i, const Index j);

    Tensor<ScalarT> & operator=  (Tensor<ScalarT> const & A);
    Tensor<ScalarT> & operator+= (Tensor<ScalarT> const & A);
    Tensor<ScalarT> & operator-= (Tensor<ScalarT> const & A);

    void clear();

  private:
    ScalarT e[MaxDim][MaxDim];
  };

  //
  // Third order tensor
  //
  template<typename ScalarT>
  class Tensor3
  {
  public:
    Tensor3();
    Tensor3(const ScalarT s);
    Tensor3(const Tensor3 & A);
    ~Tensor3();

    const ScalarT & operator()(
        const Index i, const Index j, const Index k) const;

    ScalarT & operator()(const Index i, const Index j, const Index k);

    Tensor3<ScalarT> & operator=  (Tensor3<ScalarT> const & A);
    Tensor3<ScalarT> & operator+= (Tensor3<ScalarT> const & A);
    Tensor3<ScalarT> & operator-= (Tensor3<ScalarT> const & A);

    void clear();

  private:
    ScalarT e[MaxDim][MaxDim][MaxDim];
  };

  //
  // Fourth order tensor
  //
  template<typename ScalarT>
  class Tensor4
  {
  public:
    Tensor4();
    Tensor4(const ScalarT s);
    Tensor4(const Tensor4 & A);
    ~Tensor4();

    const ScalarT & operator()(
        const Index i, const Index j, const Index k, const Index l) const;

    ScalarT & operator()(
        const Index i, const Index j,
        const Index k, const Index l);

    Tensor4<ScalarT> & operator=  (Tensor4<ScalarT> const & A);
    Tensor4<ScalarT> & operator+= (Tensor4<ScalarT> const & A);
    Tensor4<ScalarT> & operator-= (Tensor4<ScalarT> const & A);

    void clear();

  private:
    ScalarT e[MaxDim][MaxDim][MaxDim][MaxDim];
  };

  //
  // Prototypes for utilities
  //
  template<typename ScalarT>
  Vector<ScalarT>
  operator+(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  template<typename ScalarT>
  Vector<ScalarT>
  operator-(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  template<typename ScalarT>
  ScalarT
  operator*(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  template<typename ScalarT>
  bool
  operator==(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  template<typename ScalarT>
  bool
  operator!=(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  template<typename ScalarT>
  Vector<ScalarT>
  operator*(const ScalarT s, Vector<ScalarT> const & u);

  template<typename ScalarT>
  Vector<ScalarT>
  operator*(Vector<ScalarT> const & u, const ScalarT s);

  template<typename ScalarT>
  ScalarT
  dot(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  template<typename ScalarT>
  Vector<ScalarT>
  dot(Tensor<ScalarT> const & A, Vector<ScalarT> const & u);

  template<typename ScalarT>
  Vector<ScalarT>
  cross(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  template<typename ScalarT>
  Vector<ScalarT>
  dot(Vector<ScalarT> const & u, Tensor<ScalarT> const & A);

  template<typename ScalarT>
  ScalarT
  norm(Vector<ScalarT> const & u);

  template<typename ScalarT>
  ScalarT
  norm_1(Vector<ScalarT> const & u);

  template<typename ScalarT>
  ScalarT
  norm_infinity(Vector<ScalarT> const & u);

  template<typename ScalarT>
  Tensor<ScalarT>
  operator+(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  template<typename ScalarT>
  Tensor<ScalarT>
  operator-(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  template<typename ScalarT>
  Tensor<ScalarT>
  operator*(const ScalarT s, Tensor<ScalarT> const & A);

  template<typename ScalarT>
  Tensor<ScalarT>
  operator*(Tensor<ScalarT> const & A, const ScalarT s);

  template<typename ScalarT>
  Vector<ScalarT>
  operator*(Tensor<ScalarT> const & A, Vector<ScalarT> const & v);

  template<typename ScalarT>
  Vector<ScalarT>
  operator*(Vector<ScalarT> const & v, Tensor<ScalarT> const & A);

  template<typename ScalarT>
  Tensor<ScalarT>
  operator*(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  template<typename ScalarT>
  bool
  operator==(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  template<typename ScalarT>
  bool
  operator!=(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  template<typename ScalarT>
  Tensor<ScalarT>
  dot(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  template<typename ScalarT>
  ScalarT
  dotdot(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  template<typename ScalarT>
  Tensor<ScalarT>
  dotdot(Tensor4<ScalarT> const & A, Tensor<ScalarT> const & B);

  template<typename ScalarT>
  Tensor<ScalarT>
  dotdot(Tensor<ScalarT> const & B, Tensor4<ScalarT> const & A);

  template<typename ScalarT>
  Tensor4<ScalarT>
  tensor(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  template<typename ScalarT>
  Tensor4<ScalarT>
  odot(Tensor<ScalarT> const & A, Tensor<ScalarT> const & B);

  template<typename ScalarT>
  ScalarT
  norm(Tensor<ScalarT> const & A);

  template<typename ScalarT>
  ScalarT
  norm_1(Tensor<ScalarT> const & A);

  template<typename ScalarT>
  ScalarT
  norm_infinity(Tensor<ScalarT> const & A);

  template<typename ScalarT>
  Tensor<ScalarT>
  dyad(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  template<typename ScalarT>
  Tensor<ScalarT>
  tensor(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  template<typename ScalarT>
  Tensor<ScalarT>
  bun(Vector<ScalarT> const & u, Vector<ScalarT> const & v);

  template<typename ScalarT>
  const Tensor<ScalarT>
  identity();

  template<typename ScalarT>
  const Tensor<ScalarT>
  eye();

  template<typename ScalarT>
  Tensor<ScalarT>
  transpose(Tensor<ScalarT> const & A);

  template<typename ScalarT>
  Tensor<ScalarT>
  symm(Tensor<ScalarT> const & A);

  template<typename ScalarT>
  Tensor<ScalarT>
  skew(Tensor<ScalarT> const & A);

  template<typename ScalarT>
  Tensor<ScalarT>
  skew(Vector<ScalarT> const & u);

  template<typename ScalarT>
  Tensor<ScalarT>
  inverse(Tensor<ScalarT> const & A);

  template<typename ScalarT>
  ScalarT
  det(Tensor<ScalarT> const & A);

  template<typename ScalarT>
  ScalarT
  trace(Tensor<ScalarT> const & A);

  template<typename ScalarT>
  ScalarT
  I1(Tensor<ScalarT> const & A);

  template<typename ScalarT>
  ScalarT
  I2(Tensor<ScalarT> const & A);

  template<typename ScalarT>
  ScalarT
  I3(Tensor<ScalarT> const & A);

  template<typename ScalarT>
  Tensor<ScalarT>
  exp(Tensor<ScalarT> const & A);

  template<typename ScalarT>
  Tensor<ScalarT>
  log(Tensor<ScalarT> const & A);

  template<typename ScalarT>
  const Tensor4<ScalarT>
  identity_1();

  template<typename ScalarT>
  const Tensor4<ScalarT>
  identity_2();

  template<typename ScalarT>
  const Tensor4<ScalarT>
  identity_3();

  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Vector<ScalarT> & u);

  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Vector<ScalarT> const & u);

  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Tensor<ScalarT> & A);

  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Tensor<ScalarT> const & A);

  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Tensor3<ScalarT> & A);

  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Tensor3<ScalarT> const & A);

  template<typename ScalarT>
  std::istream &
  operator>>(std::istream & is, Tensor4<ScalarT> & A);

  template<typename ScalarT>
  std::ostream &
  operator<<(std::ostream & os, Tensor4<ScalarT> const & A);

  template<typename ScalarT>
  Tensor3<ScalarT>
  operator+(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B);

  template<typename ScalarT>
  Tensor3<ScalarT>
  operator-(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B);

  template<typename ScalarT>
  bool
  operator==(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B);

  template<typename ScalarT>
  bool
  operator!=(Tensor3<ScalarT> const & A, Tensor3<ScalarT> const & B);

  template<typename ScalarT>
  Tensor4<ScalarT>
  operator+(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B);

  template<typename ScalarT>
  Tensor4<ScalarT>
  operator-(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B);

  template<typename ScalarT>
  bool
  operator==(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B);

  template<typename ScalarT>
  bool
  operator!=(Tensor4<ScalarT> const & A, Tensor4<ScalarT> const & B);

} // namespace LCM

#include "Tensor.i.cc"
#include "Tensor.t.cc"

#endif //LCM_Tensor_h
