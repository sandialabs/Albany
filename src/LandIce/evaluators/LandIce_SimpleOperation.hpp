//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_SIMPLE_OPERATION_HPP
#define LANDICE_SIMPLE_OPERATION_HPP 1

#include <cmath>
#include <algorithm>

#include "Albany_SacadoTypes.hpp"

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
  void setup (const Teuchos::ParameterList& p) { tau = p.isParameter("Tau") ? p.get<double>("Tau") : 1.0; }
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

template<typename Scalar1T, typename Scalar2T>
struct Scale
{
  using ResultT = typename Albany::StrongestScalarType<Scalar1T,Scalar2T>::type;
  void setup (const Teuchos::ParameterList& /*p*/) {}
  ResultT operator() (const Scalar1T& x, const Scalar2T& factor) const {
    return factor*x;
  }
};

template<typename Scalar1T, typename Scalar2T>
struct Sum
{
  using ResultT = typename Albany::StrongestScalarType<Scalar1T,Scalar2T>::type;
  void setup (const Teuchos::ParameterList& p) {
    beta = alpha = 1.0;
    if (p.isParameter("Alpha")) {
      alpha = p.get<double>("Alpha");
    }
    if (p.isParameter("Beta")) {
      beta = p.get<double>("Beta");
    }
  }
  ResultT operator() (const Scalar1T& x, const Scalar2T& y) const {
    return alpha*x+beta*y;
  }

  double alpha;
  double beta;
};

template<typename Scalar1T, typename Scalar2T>
struct Prod
{
  using ResultT = typename Albany::StrongestScalarType<Scalar1T,Scalar2T>::type;
  void setup (const Teuchos::ParameterList& /*p*/) {}
  ResultT operator() (const Scalar1T& x, const Scalar2T& y) const {
    return x*y;
  }
};

template<typename Scalar1T, typename Scalar2T>
struct Log
{
  using ResultT = typename Albany::StrongestScalarType<Scalar1T,Scalar2T>::type;
  void setup (const Teuchos::ParameterList& /*p*/) {}
  ResultT operator() (const Scalar1T& x, const Scalar2T& a) const {
    return std::log(a*x);
  }
};

template<typename Scalar1T, typename Scalar2T>
struct Exp
{
  using ResultT = typename Albany::StrongestScalarType<Scalar1T,Scalar2T>::type;
  void setup (const Teuchos::ParameterList& /*p*/) {}
  ResultT operator() (const Scalar1T& x, const Scalar2T& tau) const {
    return std::exp(tau*x);
  }
};

template<typename Scalar1T, typename Scalar2T>
struct LowPass
{
  using ResultT = typename Albany::StrongestScalarType<Scalar1T,Scalar2T>::type;
  void setup (const Teuchos::ParameterList& /*p*/) {}
  ResultT operator() (const Scalar1T& x, const Scalar2T& threshold_up) const {
    return std::min(x,threshold_up);
  }
};

template<typename Scalar1T, typename Scalar2T>
struct HighPass
{
  using ResultT = typename Albany::StrongestScalarType<Scalar1T,Scalar2T>::type;
  void setup (const Teuchos::ParameterList& /*p*/) {}
  ResultT operator() (const Scalar1T& x, const Scalar2T& threshold_lo) const {
    return std::max(x,threshold_lo);
  }
};

template<typename Scalar1T, typename Scalar2T>
struct BandPassFixedUpper
{
  using ResultT = typename Albany::StrongestScalarType<Scalar1T,Scalar2T>::type;
  void setup (const Teuchos::ParameterList& p) { threshold_up = p.get<double>("Upper Threshold"); }
  ResultT operator() (const Scalar1T& x, const Scalar2T& threshold_lo) const {
    return std::max(std::min(x,threshold_up),threshold_lo);
    return std::max(x,threshold_lo);
  }
private:
  double threshold_up;
};

template<typename Scalar1T, typename Scalar2T>
struct BandPassFixedLower
{
  using ResultT = typename Albany::StrongestScalarType<Scalar1T,Scalar2T>::type;
  void setup (const Teuchos::ParameterList& p) { threshold_lo = p.get<double>("Lower Threshold"); }
  ResultT operator() (const Scalar1T& x, const Scalar2T& threshold_up) const {
    return std::max(std::min(x,threshold_up),threshold_lo);
  }
private:
  double threshold_lo;
};

} // namespace BinaryOps

namespace TernaryOps
{

template<typename Scalar1T, typename Scalar2T>
struct BandPass
{
  using ResultT = typename Albany::StrongestScalarType<Scalar1T,Scalar2T>::type;
  ResultT operator() (const Scalar1T& x, const Scalar2T& threshold_lo, const Scalar2T& threshold_up) const {
    return std::max(std::min(x,threshold_up),threshold_lo);
  }
};

} // namespace TernaryOps

} // namespace LandIce

#endif // LANDICE_SIMPLE_OPERATION_HPP
