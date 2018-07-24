//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_SIMPLE_OPERATION_HPP
#define LANDICE_SIMPLE_OPERATION_HPP 1

#include <cmath>
#include <algorithm>

#include "Albany_DataTypes.hpp"

namespace LandIce
{

namespace UnaryOps
{

template<typename ScalarT>
struct Scale
{
  void setup (const Teuchos::ParameterList& p) { factor = p.get<double>("Scaling Factor"); }

  ScalarT operator() (const ScalarT& x) const {
    return factor*x;
  }

private:
  double factor;
};

template<typename ScalarT>
struct Log
{
  void setup (const Teuchos::ParameterList& p) { a = p.isParameter("Factor") ? p.get<double>("Factor") : 0.0; }
  ScalarT operator() (const ScalarT& x) const {
    return std::log(a*x);
  }
private:
  double a;
};

template<typename ScalarT>
struct Exp
{
  void setup (const Teuchos::ParameterList& p) { tau = p.isParameter("Tau") ? p.get<double>("Tau") : 0.0; }
  ScalarT operator() (const ScalarT& x) const {
    return std::exp(tau*x);
  }

private:
  double tau;
};

template<typename ScalarT>
struct LowPass
{
  void setup (const Teuchos::ParameterList& p) { threshold_up = p.get<double>("Upper Threshold"); }
  ScalarT operator() (const ScalarT& x) const {
    return std::min(x,threshold_up);
  }

private:
  double threshold_up;
};

template<typename ScalarT>
struct HighPass
{
  void setup (const Teuchos::ParameterList& p) { threshold_lo = p.get<double>("Lower Threshold"); }
  ScalarT operator() (const ScalarT& x) const {
    return std::max(x,threshold_lo);
  }

private:
  double threshold_lo;
};

template<typename ScalarT>
struct BandPass
{
  void setup (const Teuchos::ParameterList& p) { threshold_lo = p.get<double>("Lower Threshold");
                                                 threshold_up = p.get<double>("Upper Threshold"); }
  ScalarT operator() (const ScalarT& x) const {
    return std::max(std::min(x,threshold_up),threshold_lo);
  }

private:
  double threshold_lo;
  double threshold_up;
};

} // namespace UnaryOps

namespace BinaryOps
{

template<typename ScalarT>
struct Scale
{
  void setup (const Teuchos::ParameterList& /*p*/) {}
  ScalarT operator() (const ScalarT& x, const ScalarT& factor) const {
    return factor*x;
  }
};

template<typename ScalarT>
struct Log
{
  void setup (const Teuchos::ParameterList& /*p*/) {}
  ScalarT operator() (const ScalarT& x, const ScalarT& a) const {
    return std::log(a*x);
  }
};

template<typename ScalarT>
struct Exp
{
  void setup (const Teuchos::ParameterList& /*p*/) {}
  ScalarT operator() (const ScalarT& x, const ScalarT& tau) const {
    return std::exp(tau*x);
  }
};

template<typename ScalarT>
struct LowPass
{
  void setup (const Teuchos::ParameterList& /*p*/) {}
  ScalarT operator() (const ScalarT& x, const ScalarT& threshold_up) const {
    return std::min(x,threshold_up);
  }
};

template<typename ScalarT>
struct HighPass
{
  void setup (const Teuchos::ParameterList& /*p*/) {}
  ScalarT operator() (const ScalarT& x, const ScalarT& threshold_lo) const {
    return std::max(x,threshold_lo);
  }
};

template<typename ScalarT>
struct BandPassFixedUpper
{
  void setup (const Teuchos::ParameterList& p) { threshold_up = p.get<double>("Upper Threshold"); }
  ScalarT operator() (const ScalarT& x, const ScalarT& threshold_lo) const {
    return std::max(std::min(x,threshold_up),threshold_lo);
    return std::max(x,threshold_lo);
  }
private:
  double threshold_up;
};

template<typename ScalarT>
struct BandPassFixedLower
{
  void setup (const Teuchos::ParameterList& p) { threshold_lo = p.get<double>("Lower Threshold"); }
  ScalarT operator() (const ScalarT& x, const ScalarT& threshold_up) const {
    return std::max(std::min(x,threshold_up),threshold_lo);
  }
private:
  double threshold_lo;
};

} // namespace BinaryOps

namespace TernaryOps
{

template<typename ScalarT>
struct BandPass
{
  ScalarT operator() (const ScalarT& x, const ScalarT& threshold_lo, const ScalarT& threshold_up) const {
    return std::max(std::min(x,threshold_up),threshold_lo);
  }
};

} // namespace TernaryOps

} // namespace LandIce

#endif // LANDICE_SIMPLE_OPERATION_HPP
