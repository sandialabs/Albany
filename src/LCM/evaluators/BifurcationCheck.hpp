//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_BifurcationCheck_hpp)
#define LCM_BifurcationCheck_hpp

#include <iostream>

#include <MiniTensor.h>
#include "Albany_Layouts.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Sacado.hpp"

namespace LCM {
/// \brief BifurcationCheck Evaluator
///
///  This evaluator checks whether a material point has become
///  unstable
///
template <typename EvalT, typename Traits>
class BifurcationCheck : public PHX::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  ///
  /// Constructor
  ///
  BifurcationCheck(
      const Teuchos::ParameterList&        p,
      const Teuchos::RCP<Albany::Layouts>& dl);

  ///
  /// Phalanx method to allocate space
  ///
  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  ///
  /// Implementation of physics
  ///
  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT                              ScalarT;
  typedef typename EvalT::MeshScalarT                          MeshScalarT;
  typedef typename Sacado::mpl::apply<FadType, ScalarT>::type  DFadType;
  typedef typename Sacado::mpl::apply<FadType, DFadType>::type D2FadType;

  //! Input: Parametrization type
  std::string parametrization_type_;

  //! Input: Parametrization sweep interval
  double parametrization_interval_;

  //! Input: material tangent
  PHX::MDField<const ScalarT, Cell, QuadPoint, Dim, Dim, Dim, Dim> tangent_;

  //! Output: ellipticity indicator
  PHX::MDField<ScalarT, Cell, QuadPoint> ellipticity_flag_;

  //! Output: instability direction
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim> direction_;

  //! Output: minimum of acoustic tensor
  PHX::MDField<ScalarT, Cell, QuadPoint> min_detA_;

  //! number of integration points
  int num_pts_;

  //! number of spatial dimensions
  int num_dims_;

  ///
  /// Spherical parametrization sweep
  ///
  ScalarT spherical_sweep(
      minitensor::Tensor4<ScalarT, 3> const& tangent,
      minitensor::Vector<ScalarT, 2>&        arg_minimum,
      minitensor::Vector<ScalarT, 3>&        direction,
      double const&                          interval);

  ///
  /// Stereographic parametrization sweep
  ///
  ScalarT stereographic_sweep(
      minitensor::Tensor4<ScalarT, 3> const& tangent,
      minitensor::Vector<ScalarT, 2>&        arg_minimum,
      minitensor::Vector<ScalarT, 3>&        direction,
      double const&                          interval);

  ///
  /// Projective parametrization sweep
  ///
  ScalarT projective_sweep(
      minitensor::Tensor4<ScalarT, 3> const& tangent,
      minitensor::Vector<ScalarT, 3>&        arg_minimum,
      minitensor::Vector<ScalarT, 3>&        direction,
      double const&                          interval);

  ///
  /// Tangent parametrization sweep
  ///
  ScalarT tangent_sweep(
      minitensor::Tensor4<ScalarT, 3> const& tangent,
      minitensor::Vector<ScalarT, 2>&        arg_minimum,
      minitensor::Vector<ScalarT, 3>&        direction,
      double const&                          interval);

  ///
  /// Cartesian parametrization sweep
  ///
  ScalarT cartesian_sweep(
      minitensor::Tensor4<ScalarT, 3> const& tangent,
      minitensor::Vector<ScalarT, 2>&        arg_minimum,
      int                                    surface_index,
      minitensor::Vector<ScalarT, 3>&        direction,
      double const&                          interval);

  ///
  /// Newton-Raphson method to find exact min DetA and direction
  ///
  void spherical_newton_raphson(
      minitensor::Tensor4<ScalarT, 3> const& tangent,
      minitensor::Vector<ScalarT, 2>&        parameters,
      minitensor::Vector<ScalarT, 3>&        direction,
      ScalarT&                               min_detA);

  void stereographic_newton_raphson(
      minitensor::Tensor4<ScalarT, 3> const& tangent,
      minitensor::Vector<ScalarT, 2>&        parameters,
      minitensor::Vector<ScalarT, 3>&        direction,
      ScalarT&                               min_detA);

  void projective_newton_raphson(
      minitensor::Tensor4<ScalarT, 3> const& tangent,
      minitensor::Vector<ScalarT, 3>&        parameters,
      minitensor::Vector<ScalarT, 3>&        direction,
      ScalarT&                               min_detA);

  void tangent_newton_raphson(
      minitensor::Tensor4<ScalarT, 3> const& tangent,
      minitensor::Vector<ScalarT, 2>&        parameters,
      minitensor::Vector<ScalarT, 3>&        direction,
      ScalarT&                               min_detA);

  void cartesian_newton_raphson(
      minitensor::Tensor4<ScalarT, 3> const& tangent,
      minitensor::Vector<ScalarT, 2>&        parameters,
      int                                    surface_index,
      minitensor::Vector<ScalarT, 3>&        direction,
      ScalarT&                               min_detA);

  ///
  /// PSO method
  ///
  ScalarT stereographic_pso(
      minitensor::Tensor4<ScalarT, 3> const& tangent,
      minitensor::Vector<ScalarT, 2>&        arg_minimum,
      minitensor::Vector<ScalarT, 3>&        direction);

  ///
  /// Get normal
  ///
  minitensor::Vector<D2FadType, 3> spherical_get_normal(
      minitensor::Vector<D2FadType, 2>& parameters);

  minitensor::Vector<D2FadType, 3> stereographic_get_normal(
      minitensor::Vector<D2FadType, 2>& parameters);

  minitensor::Vector<D2FadType, 3> projective_get_normal(
      minitensor::Vector<D2FadType, 3>& parameters);

  minitensor::Vector<D2FadType, 3> tangent_get_normal(
      minitensor::Vector<D2FadType, 2>& parameters);

  minitensor::Vector<D2FadType, 3> cartesian_get_normal1(
      minitensor::Vector<D2FadType, 2>& parameters);

  minitensor::Vector<D2FadType, 3> cartesian_get_normal2(
      minitensor::Vector<D2FadType, 2>& parameters);

  minitensor::Vector<D2FadType, 3> cartesian_get_normal3(
      minitensor::Vector<D2FadType, 2>& parameters);
};

}  // namespace LCM
#endif
