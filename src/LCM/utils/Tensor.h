//
// First cut of LCM small tensor utilities.
//
#if !defined(LCM_Tensor_h)
#define LCM_Tensor_h

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
    Vector(const Scalar & s);
    Vector(const Scalar & s0, const Scalar & s1, const Scalar & s2);
    Vector(const Scalar s[MaxDim]);
    Vector(const Vector & v);
    ~Vector();

    const Scalar & operator[](const Index i) const;
    Scalar & operator[](const Index i);

    const Scalar & operator()(const Index i) const;
    Scalar & operator()(const Index i);

    Vector<Scalar> & operator=  (const Vector<Scalar> & v);
    Vector<Scalar>   operator+  (const Vector<Scalar> & v) const;
    Vector<Scalar>   operator-  (const Vector<Scalar> & v) const;
    Vector<Scalar> & operator+= (const Vector<Scalar> & v);
    Vector<Scalar> & operator-= (const Vector<Scalar> & v);

    Scalar operator*(const Vector<Scalar> & v) const;
    Vector<Scalar> operator*(const Scalar & s) const;

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
    Tensor(const Scalar & s);
    Tensor(const Scalar & s00, const Scalar & s01, const Scalar & s02,
        const Scalar & s10, const Scalar & s11, const Scalar & s12,
        const Scalar & s20, const Scalar & s21, const Scalar & s22);
    Tensor(const Scalar s[MaxDim][MaxDim]);
    Tensor(const Tensor & A);
    ~Tensor();

    const Scalar & operator()(const Index i, const Index j) const;
    Scalar & operator()(const Index i, const Index j);

    Tensor<Scalar> & operator=  (const Tensor<Scalar> & A);
    Tensor<Scalar>   operator+  (const Tensor<Scalar> & A) const;
    Tensor<Scalar>   operator-  (const Tensor<Scalar> & A) const;
    Tensor<Scalar> & operator+= (const Tensor<Scalar> & A);
    Tensor<Scalar> & operator-= (const Tensor<Scalar> & A);

    Tensor<Scalar>   operator*  (const Scalar & s) const;
    Tensor<Scalar>   operator*  (const Tensor<Scalar> & A) const;
    Vector<Scalar>   operator*  (const Vector<Scalar> & v) const;

    void clear();

  private:
    Scalar e[MaxDim][MaxDim];
  };

  //
  // Prototypes for utilities
  //
  template<typename Scalar>
  Scalar dot(const Vector<Scalar> & u, const Vector<Scalar> & v);

  template<typename Scalar>
  Vector<Scalar> dot(const Tensor<Scalar> & A, const Vector<Scalar> & u);

  template<typename Scalar>
  Vector<Scalar> cross(const Vector<Scalar> & u, const Vector<Scalar> & v);

  template<typename Scalar>
  Vector<Scalar> dot(const Vector<Scalar> & u, const Tensor<Scalar> & A);

  template<typename Scalar>
  Scalar norm(const Vector<Scalar> & v);

  template<typename Scalar>
  Tensor<Scalar> dot(const Tensor<Scalar> & A, const Tensor<Scalar> & B);

  template<typename Scalar>
  Scalar dotdot(const Tensor<Scalar> & A, const Tensor<Scalar> & B);

  template<typename Scalar>
  Scalar norm(const Tensor<Scalar> & A);

  template<typename Scalar>
  Tensor<Scalar> dyad(const Vector<Scalar> & u, const Vector<Scalar> & v);

  template<typename Scalar>
  Tensor<Scalar> bun(const Vector<Scalar> & u, const Vector<Scalar> & v);

  template<typename Scalar>
  Tensor<Scalar> eye();

} // namespace LCM

#include "Tensor.i.cc"

#endif //LCM_Tensor_h
