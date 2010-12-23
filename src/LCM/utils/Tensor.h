//
// First cut of LCM small tensor utilities.
//
#if !defined(LCM_Tensor_h)
#define LCM_Tensor_h

#include <iostream>

#include "Intrepid_FieldContainer.hpp"

namespace LCM {

  typedef unsigned int Index;
  const Index MaxDim = 3;

  //
  // Vector
  //
  template<typename Scalar>
  class Vector
  {
  public:
    Vector();
    Vector(const Scalar s);
    Vector(const Scalar s0, const Scalar s1, const Scalar s2);
    Vector(const Vector & v);
    ~Vector();

    const Scalar & operator()(const Index i) const;
    Scalar & operator()(const Index i);

    Vector<Scalar> & operator=  (const Vector<Scalar> & v);
    Vector<Scalar> & operator+= (const Vector<Scalar> & v);
    Vector<Scalar> & operator-= (const Vector<Scalar> & v);

    void clear();

  private:
    Scalar e[MaxDim];
  };

  //
  // Second order tensor
  //
  template<typename Scalar>
  class Tensor
  {
  public:
    Tensor();
    Tensor(const Scalar s);
    Tensor(
        const Scalar s00, const Scalar s01, const Scalar s02,
        const Scalar s10, const Scalar s11, const Scalar s12,
        const Scalar s20, const Scalar s21, const Scalar s22);
    Tensor(const Tensor & A);
    ~Tensor();

    const Scalar & operator()(const Index i, const Index j) const;
    Scalar & operator()(const Index i, const Index j);

    Tensor<Scalar> & operator=  (const Tensor<Scalar> & A);
    Tensor<Scalar> & operator+= (const Tensor<Scalar> & A);
    Tensor<Scalar> & operator-= (const Tensor<Scalar> & A);

    void clear();

  private:
    Scalar e[MaxDim][MaxDim];
  };

  //
  // Third order tensor
  //
  template<typename Scalar>
  class Tensor3
  {
  public:
    Tensor3();
    Tensor3(const Scalar s);
    Tensor3(const Tensor3 & A);
    ~Tensor3();

    const Scalar & operator()(
        const Index i, const Index j, const Index k) const;

    Scalar & operator()(const Index i, const Index j, const Index k);

    Tensor3<Scalar> & operator=  (const Tensor3<Scalar> & A);
    Tensor3<Scalar> & operator+= (const Tensor3<Scalar> & A);
    Tensor3<Scalar> & operator-= (const Tensor3<Scalar> & A);

    void clear();

  private:
    Scalar e[MaxDim][MaxDim][MaxDim];
  };

  //
  // Fourth order tensor
  //
  template<typename Scalar>
  class Tensor4
  {
  public:
    Tensor4();
    Tensor4(const Scalar s);
    Tensor4(const Tensor4 & A);
    ~Tensor4();

    const Scalar & operator()(
        const Index i, const Index j, const Index k, const Index l) const;

    Scalar & operator()(
        const Index i, const Index j,
        const Index k, const Index l);

    Tensor4<Scalar> & operator=  (const Tensor4<Scalar> & A);
    Tensor4<Scalar> & operator+= (const Tensor4<Scalar> & A);
    Tensor4<Scalar> & operator-= (const Tensor4<Scalar> & A);

    void clear();

  private:
    Scalar e[MaxDim][MaxDim][MaxDim][MaxDim];
  };

  //
  // Prototypes for utilities
  //
  template<typename Scalar>
  Vector<Scalar>
  operator+(const Vector<Scalar> & u, const Vector<Scalar> & v);

  template<typename Scalar>
  Vector<Scalar>
  operator-(const Vector<Scalar> & u, const Vector<Scalar> & v);

  template<typename Scalar>
  Scalar
  operator*(const Vector<Scalar> & u, const Vector<Scalar> & v);

  template<typename Scalar>
  Vector<Scalar>
  operator*(const Scalar s, const Vector<Scalar> & u);

  template<typename Scalar>
  Vector<Scalar>
  operator*(const Vector<Scalar> & u, const Scalar s);

  template<typename Scalar>
  Scalar
  dot(const Vector<Scalar> & u, const Vector<Scalar> & v);

  template<typename Scalar>
  Vector<Scalar>
  dot(const Tensor<Scalar> & A, const Vector<Scalar> & u);

  template<typename Scalar>
  Vector<Scalar>
  cross(const Vector<Scalar> & u, const Vector<Scalar> & v);

  template<typename Scalar>
  Vector<Scalar>
  dot(const Vector<Scalar> & u, const Tensor<Scalar> & A);

  template<typename Scalar>
  Scalar
  norm(const Vector<Scalar> & u);

  template<typename Scalar>
  Scalar
  norm_1(const Vector<Scalar> & u);

  template<typename Scalar>
  Scalar
  norm_infinity(const Vector<Scalar> & u);

  template<typename Scalar>
  Tensor<Scalar>
  operator+(const Tensor<Scalar> & A, const Tensor<Scalar> & B);

  template<typename Scalar>
  Tensor<Scalar>
  operator-(const Tensor<Scalar> & A, const Tensor<Scalar> & B);

  template<typename Scalar>
  Tensor<Scalar>
  operator*(const Scalar s, const Tensor<Scalar> & A);

  template<typename Scalar>
  Tensor<Scalar>
  operator*(const Tensor<Scalar> & A, const Scalar s);

  template<typename Scalar>
  Vector<Scalar>
  operator*(const Tensor<Scalar> & A, const Vector<Scalar> & v);

  template<typename Scalar>
  Vector<Scalar>
  operator*(const Vector<Scalar> & v, const Tensor<Scalar> & A);

  template<typename Scalar>
  Tensor<Scalar>
  operator*(const Tensor<Scalar> & A, const Tensor<Scalar> & B);

  template<typename Scalar>
  Tensor<Scalar>
  dot(const Tensor<Scalar> & A, const Tensor<Scalar> & B);

  template<typename Scalar>
  Scalar
  dotdot(const Tensor<Scalar> & A, const Tensor<Scalar> & B);

  template<typename Scalar>
  Tensor<Scalar>
  dotdot(const Tensor4<Scalar> & A, const Tensor<Scalar> & B);

  template<typename Scalar>
  Tensor<Scalar>
  dotdot(const Tensor<Scalar> & B, const Tensor4<Scalar> & A);

  template<typename Scalar>
  Tensor4<Scalar>
  tensor(const Tensor<Scalar> & A, const Tensor<Scalar> & B);

  template<typename Scalar>
  Tensor4<Scalar>
  odot(const Tensor<Scalar> & A, const Tensor<Scalar> & B);

  template<typename Scalar>
  Scalar
  norm(const Tensor<Scalar> & A);

  template<typename Scalar>
  Scalar
  norm_1(const Tensor<Scalar> & A);

  template<typename Scalar>
  Scalar
  norm_infinity(const Tensor<Scalar> & A);

  template<typename Scalar>
  Tensor<Scalar>
  dyad(const Vector<Scalar> & u, const Vector<Scalar> & v);

  template<typename Scalar>
  Tensor<Scalar>
  tensor(const Vector<Scalar> & u, const Vector<Scalar> & v);

  template<typename Scalar>
  Tensor<Scalar>
  bun(const Vector<Scalar> & u, const Vector<Scalar> & v);

  template<typename Scalar>
  const Tensor<Scalar>
  eye();

  template<typename Scalar>
  const Tensor<Scalar>
  identity();

  template<typename Scalar>
  Tensor<Scalar>
  transpose(const Tensor<Scalar> & A);

  template<typename Scalar>
  Tensor<Scalar>
  symm(const Tensor<Scalar> & A);

  template<typename Scalar>
  Tensor<Scalar>
  skew(const Tensor<Scalar> & A);

  template<typename Scalar>
  Tensor<Scalar>
  skew(const Vector<Scalar> & u);

  template<typename Scalar>
  Tensor<Scalar>
  inverse(const Tensor<Scalar> & A);

  template<typename Scalar>
  Scalar
  det(const Tensor<Scalar> & A);

  template<typename Scalar>
  Scalar
  trace(const Tensor<Scalar> & A);

  template<typename Scalar>
  Scalar
  I1(const Tensor<Scalar> & A);

  template<typename Scalar>
  Scalar
  I2(const Tensor<Scalar> & A);

  template<typename Scalar>
  Scalar
  I3(const Tensor<Scalar> & A);

  template<typename Scalar>
  Tensor<Scalar>
  exp(const Tensor<Scalar> & A);

  template<typename Scalar>
  Tensor<Scalar>
  log(const Tensor<Scalar> & A);

  template<typename Scalar>
  const Tensor4<Scalar>
  identity_1();

  template<typename Scalar>
  const Tensor4<Scalar>
  identity_2();

  template<typename Scalar>
  const Tensor4<Scalar>
  identity_3();

  template<typename Scalar>
  std::istream &
  operator>>(std::istream & is, Vector<Scalar> & u);

  template<typename Scalar>
  std::ostream &
  operator<<(std::ostream & os, const Vector<Scalar> & u);

  template<typename Scalar>
  std::istream &
  operator>>(std::istream & is, Tensor<Scalar> & A);

  template<typename Scalar>
  std::ostream &
  operator<<(std::ostream & os, const Tensor<Scalar> & A);

  template<typename Scalar>
  std::istream &
  operator>>(std::istream & is, Tensor3<Scalar> & A);

  template<typename Scalar>
  std::ostream &
  operator<<(std::ostream & os, const Tensor3<Scalar> & A);

  template<typename Scalar>
  std::istream &
  operator>>(std::istream & is, Tensor4<Scalar> & A);

  template<typename Scalar>
  std::ostream &
  operator<<(std::ostream & os, const Tensor4<Scalar> & A);

} // namespace LCM

#include "Tensor.i.cc"

#endif //LCM_Tensor_h
