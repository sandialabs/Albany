#ifndef QCAD_MATHVECTOR_HPP
#define QCAD_MATHVECTOR_HPP

#include <cstdlib>
#include <vector>
#include <iostream>
#include <cmath>

namespace QCAD {

  // Helper class: a vector with math operators
  class mathVector
  {
  public:
    mathVector();
    mathVector(int n);
    mathVector(const mathVector& copy);
    ~mathVector();

    void resize(std::size_t n);
    void fill(double d);
    void fill(const double* vec);
    double dot(const mathVector& v2) const;
    double distanceTo(const mathVector& v2) const;
    double distanceTo(const double* p) const;

    double norm() const;
    double norm2() const;
    void normalize();

    double* data();
    const double* data() const;
    std::size_t size() const;

    mathVector& operator=(const mathVector& rhs);

    mathVector operator+(const mathVector& v2) const;
    mathVector operator-(const mathVector& v2) const;
    mathVector operator*(double scale) const;

    mathVector& operator+=(const mathVector& v2);
    mathVector& operator-=(const mathVector& v2);
    mathVector& operator*=(double scale);
    mathVector& operator/=(double scale);

    double& operator[](int i);
    const double& operator[](int i) const;

  private:
    int dim_;
    std::vector<double> data_;
  };

  std::ostream& operator<<(std::ostream& os, const mathVector& mv);

  bool ptInPolygon(const std::vector<QCAD::mathVector>& polygon, const QCAD::mathVector& pt);
  bool ptInPolygon(const std::vector<QCAD::mathVector>& polygon, const double* pt);
}

#endif
