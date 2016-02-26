//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "MOR_SingularValuesHelpers.hpp"

#include <algorithm>
#include <numeric>
#include <iterator>
#include <functional>
#include <cmath>

namespace MOR {

namespace Detail {

struct square : public std::unary_function<double, double> {
  double operator()(double x) const { return x * x; }
};

class relative_magnitude_from_square : public std::unary_function<double, double> {
public:
  explicit relative_magnitude_from_square(double x2_ref) :
    x2_ref_(x2_ref)
  {}

  double operator()(double x2) const { return std::sqrt(x2 / x2_ref_); }

private:
  double x2_ref_;
};

} // namespace Detail

Teuchos::Array<double> computeDiscardedEnergyFractions(Teuchos::ArrayView<const double> singularValues)
{
  Teuchos::Array<double> result;

  if (singularValues.begin() != singularValues.end()) {
    result.reserve(singularValues.size());
    std::transform(
        singularValues.begin(), singularValues.end(),
        std::back_inserter(result),
        Detail::square());

    std::partial_sum(result.rbegin(), result.rend(), result.rbegin());

    std::transform(result.begin() + 1, result.end(),
        result.begin(),
        Detail::relative_magnitude_from_square(result.front()));
    result.back() = 0.0;
  }

  return result;
}

} // namespace MOR
