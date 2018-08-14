//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <random>
#include <typeinfo>

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "LocalNonlinearSolver.hpp"
#include "MiniTensor.h"

namespace LCM {

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
BifurcationCheck<EvalT, Traits>::BifurcationCheck(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : parametrization_type_(p.get<std::string>("Parametrization Type Name")),
      parametrization_interval_(p.get<double>("Parametrization Interval Name")),
      tangent_(p.get<std::string>("Material Tangent Name"), dl->qp_tensor4),
      ellipticity_flag_(
          p.get<std::string>("Ellipticity Flag Name"),
          dl->qp_scalar),
      direction_(
          p.get<std::string>("Bifurcation Direction Name"),
          dl->qp_vector),
      min_detA_(p.get<std::string>("Min detA Name"), dl->qp_scalar)
{
  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_tensor->dimensions(dims);
  num_pts_  = dims[1];
  num_dims_ = dims[2];

  this->addDependentField(tangent_);
  this->addEvaluatedField(ellipticity_flag_);
  this->addEvaluatedField(direction_);
  this->addEvaluatedField(min_detA_);

  this->setName("BifurcationCheck" + PHX::typeAsString<EvalT>());
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
BifurcationCheck<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(tangent_, fm);
  this->utils.setFieldData(ellipticity_flag_, fm);
  this->utils.setFieldData(direction_, fm);
  this->utils.setFieldData(min_detA_, fm);
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
BifurcationCheck<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  minitensor::Vector<ScalarT, 3>  direction(1.0, 0.0, 0.0);
  minitensor::Tensor4<ScalarT, 3> tangent;
  bool                            ellipticity_flag(false);
  ScalarT                         min_detA(1.0);

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      tangent.fill(tangent_, cell, pt, 0, 0, 0, 0);
      ellipticity_flag_(cell, pt) = 0;

      double interval = parametrization_interval_;

      if (parametrization_type_ == "Oliver") {
        boost::tie(ellipticity_flag, direction) =
            minitensor::check_strong_ellipticity(tangent);
        min_detA = minitensor::det(
            minitensor::dot2(direction, minitensor::dot(tangent, direction)));
      } else if (parametrization_type_ == "PSO") {
        minitensor::Vector<ScalarT, 2> arg_minimum;

        min_detA = stereographic_pso(tangent, arg_minimum, direction);

      } else if (parametrization_type_ == "Spherical") {
        minitensor::Vector<ScalarT, 2> arg_minimum;

        min_detA = spherical_sweep(tangent, arg_minimum, direction, interval);
        spherical_newton_raphson(tangent, arg_minimum, direction, min_detA);

      } else if (parametrization_type_ == "Stereographic") {
        minitensor::Vector<ScalarT, 2> arg_minimum;

        min_detA =
            stereographic_sweep(tangent, arg_minimum, direction, interval);
        stereographic_newton_raphson(tangent, arg_minimum, direction, min_detA);

      } else if (parametrization_type_ == "Projective") {
        minitensor::Vector<ScalarT, 3> arg_minimum;

        min_detA = projective_sweep(tangent, arg_minimum, direction, interval);
        projective_newton_raphson(tangent, arg_minimum, direction, min_detA);

      } else if (parametrization_type_ == "Tangent") {
        minitensor::Vector<ScalarT, 2> arg_minimum;

        min_detA = tangent_sweep(tangent, arg_minimum, direction, interval);
        tangent_newton_raphson(tangent, arg_minimum, direction, min_detA);

      } else if (parametrization_type_ == "Cartesian") {
        minitensor::Vector<ScalarT, 2> arg_minimum1;
        minitensor::Vector<ScalarT, 2> arg_minimum2;
        minitensor::Vector<ScalarT, 2> arg_minimum3;
        minitensor::Vector<ScalarT, 3> direction1(1.0, 0.0, 0.0);
        minitensor::Vector<ScalarT, 3> direction2(0.0, 1.0, 0.0);
        minitensor::Vector<ScalarT, 3> direction3(0.0, 0.0, 1.0);

        ScalarT min_detA1 =
            cartesian_sweep(tangent, arg_minimum1, 1, direction1, interval);

        ScalarT min_detA2 =
            cartesian_sweep(tangent, arg_minimum2, 2, direction2, interval);

        ScalarT min_detA3 =
            cartesian_sweep(tangent, arg_minimum3, 3, direction3, interval);

        if (min_detA1 <= min_detA2 && min_detA1 <= min_detA3) {
          cartesian_newton_raphson(
              tangent, arg_minimum1, 1, direction1, min_detA1);

          min_detA  = min_detA1;
          direction = direction1;

        } else if (min_detA2 <= min_detA1 && min_detA2 <= min_detA3) {
          cartesian_newton_raphson(
              tangent, arg_minimum2, 2, direction2, min_detA2);

          min_detA  = min_detA2;
          direction = direction2;

        } else if (min_detA3 <= min_detA1 && min_detA3 <= min_detA2) {
          cartesian_newton_raphson(
              tangent, arg_minimum3, 3, direction3, min_detA3);

          min_detA  = min_detA3;
          direction = direction3;
        }

      } else {
        minitensor::Vector<ScalarT, 2> arg_minimum;

        min_detA = spherical_sweep(tangent, arg_minimum, direction, interval);
        spherical_newton_raphson(tangent, arg_minimum, direction, min_detA);
      }

      ellipticity_flag = true;
      if (min_detA <= 0.0) ellipticity_flag = false;

      ellipticity_flag_(cell, pt) = ellipticity_flag;
      min_detA_(cell, pt)         = min_detA;

      // std::cout << "\n" << min_detA << " @ " << direction << std::endl;

      for (int i(0); i < num_dims_; ++i) {
        direction_(cell, pt, i) = direction(i);
      }
    }
  }
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
typename EvalT::ScalarT BifurcationCheck<EvalT, Traits>::spherical_sweep(
    minitensor::Tensor4<ScalarT, 3> const& tangent,
    minitensor::Vector<ScalarT, 2>&        arg_minimum,
    minitensor::Vector<ScalarT, 3>&        direction,
    double const&                          interval)
{
  minitensor::Index const p_number = floor(1.0 / interval);

  ScalarT const domain_min = 0;

  ScalarT const domain_max = std::acos(-1.0);

  ScalarT const p_mean = (domain_max + domain_min) / 2.0;

  ScalarT const p_span = domain_max - domain_min;

  ScalarT const p_min = p_mean - p_span / 2.0 * interval * p_number;
  // p_min = domain_min;

  ScalarT const p_max = p_mean + p_span / 2.0 * interval * p_number;
  // p_max = domain_min + p_span * interval * p_number;

  // Initialize parameters
  ScalarT const phi_min = p_min;

  ScalarT const phi_max = p_max;

  ScalarT const theta_min = p_min;

  ScalarT const theta_max = p_max;

  minitensor::Index const phi_num_points = p_number * 2 + 1;
  // phi_num_points = p_number + 1;

  minitensor::Index const theta_num_points = p_number * 2 + 1;
  // theta_num_points = p_number + 1;

  minitensor::Vector<ScalarT, 2> const sphere_min(phi_min, theta_min);

  minitensor::Vector<ScalarT, 2> const sphere_max(phi_max, theta_max);

  minitensor::Vector<minitensor::Index, 2> const sphere_num_points(
      phi_num_points, theta_num_points);

  // Build the parametric grid with the specified parameters.
  minitensor::ParametricGrid<ScalarT, 2> sphere_grid(
      sphere_min, sphere_max, sphere_num_points);

  // Build a spherical parametrization for this elasticity.
  minitensor::SphericalParametrization<ScalarT, 3> sphere_param(tangent);

  // Traverse the grid with the parametrization.
  sphere_grid.traverse(sphere_param);

  // Query the parametrization for the minimum and maximum found on the grid.
  // std::cout << "\n*** SPHERICAL PARAMETRIZATION ***\n";
  // std::cout << "Interval: " << parametrization_interval_ << std::endl;
  // std::cout << sphere_param.get_minimum()
  //<< "  " << sphere_param.get_normal_minimum() << std::endl;

  ScalarT min_detA = sphere_param.get_minimum();
  for (int i(0); i < 3; ++i) {
    direction(i) = (sphere_param.get_normal_minimum())(i);
  }

  for (int i(0); i < 2; ++i) {
    arg_minimum(i) = (sphere_param.get_arg_minimum())(i);
  }

  return min_detA;
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
typename EvalT::ScalarT BifurcationCheck<EvalT, Traits>::stereographic_sweep(
    minitensor::Tensor4<ScalarT, 3> const& tangent,
    minitensor::Vector<ScalarT, 2>&        arg_minimum,
    minitensor::Vector<ScalarT, 3>&        direction,
    double const&                          interval)
{
  minitensor::Index const p_number = floor(1.0 / interval);

  ScalarT const domain_min = -1.0;

  ScalarT const domain_max = 1.0;

  ScalarT const p_mean = (domain_max + domain_min) / 2.0;

  ScalarT const p_span = domain_max - domain_min;

  ScalarT const p_min = p_mean - p_span / 2.0 * interval * p_number;
  // p_min = domain_min;

  ScalarT const p_max = p_mean + p_span / 2.0 * interval * p_number;
  // p_max = domain_min + p_span * interval * p_number;

  // Initialize parametres
  ScalarT const x_min = p_min;

  ScalarT const x_max = p_max;

  ScalarT const y_min = p_min;

  ScalarT const y_max = p_max;

  minitensor::Index const x_num_points = p_number * 2 + 1;
  // x_num_points = p_number + 1;

  minitensor::Index const y_num_points = p_number * 2 + 1;
  // y_num_points = p_number + 1;

  minitensor::Vector<ScalarT, 2> const stereographic_min(x_min, y_min);

  minitensor::Vector<ScalarT, 2> const stereographic_max(x_max, y_max);

  minitensor::Vector<minitensor::Index, 2> const stereographic_num_points(
      x_num_points, y_num_points);

  // Build the parametric grid with the specified parameters.
  minitensor::ParametricGrid<ScalarT, 2> stereographic_grid(
      stereographic_min, stereographic_max, stereographic_num_points);

  // Build a stereographic parametrization for this elasticity.
  minitensor::StereographicParametrization<ScalarT, 3> stereographic_param(
      tangent);

  // Traverse the grid with the parametrization.
  stereographic_grid.traverse(stereographic_param);

  // Query the parametrization for the minimum and maximum found on the grid.
  // std::cout << "\n*** STEREOGRAPHIC PARAMETRIZATION ***\n";
  // std::cout << "Interval: " << parametrization_interval_ << std::endl;
  // std::cout << stereographic_param.get_minimum()
  //<< "  " << stereographic_param.get_arg_minimum() << std::endl
  //<< "  " << stereographic_param.get_normal_minimum() << std::endl;

  ScalarT min_detA = stereographic_param.get_minimum();
  for (int i(0); i < 3; ++i) {
    direction(i) = (stereographic_param.get_normal_minimum())(i);
  }

  for (int i(0); i < 2; ++i) {
    arg_minimum(i) = (stereographic_param.get_arg_minimum())(i);
  }

  return min_detA;
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
typename EvalT::ScalarT BifurcationCheck<EvalT, Traits>::projective_sweep(
    minitensor::Tensor4<ScalarT, 3> const& tangent,
    minitensor::Vector<ScalarT, 3>&        arg_minimum,
    minitensor::Vector<ScalarT, 3>&        direction,
    double const&                          interval)
{
  minitensor::Index const p_number = floor(1.0 / interval);

  ScalarT const domain_min = -1.0;

  ScalarT const domain_max = 1.0;

  ScalarT const p_mean = (domain_max + domain_min) / 2.0;

  ScalarT const p_span = domain_max - domain_min;

  ScalarT const p_min = p_mean - p_span / 2.0 * interval * p_number;
  // p_min = domain_min;

  ScalarT const p_max = p_mean + p_span / 2.0 * interval * p_number;
  // p_max = domain_min + p_span * interval * p_number;

  // Initialize parametres
  ScalarT const x_min = p_min;

  ScalarT const x_max = p_max;

  ScalarT const y_min = p_min;

  ScalarT const y_max = p_max;

  ScalarT const z_min = p_min;

  ScalarT const z_max = p_max;

  minitensor::Index const x_num_points = p_number * 2 + 1;
  // x_num_points = p_number + 1;

  minitensor::Index const y_num_points = p_number * 2 + 1;
  // y_num_points = p_number + 1;

  minitensor::Index const z_num_points = p_number * 2 + 1;
  // z_num_points = p_number + 1;

  minitensor::Vector<ScalarT, 3> const projective_min(x_min, y_min, z_min);

  minitensor::Vector<ScalarT, 3> const projective_max(x_max, y_max, z_max);

  minitensor::Vector<minitensor::Index, 3> const projective_num_points(
      x_num_points, y_num_points, z_num_points);

  // Build the parametric grid with the specified parameters.
  minitensor::ParametricGrid<ScalarT, 3> projective_grid(
      projective_min, projective_max, projective_num_points);

  // Build a projective parametrization for this elasticity.
  minitensor::ProjectiveParametrization<ScalarT, 3> projective_param(tangent);

  // Traverse the grid with the parametrization.
  projective_grid.traverse(projective_param);

  // Query the parametrization for the minimum and maximum found on the grid.
  // std::cout << "\n*** PROJECTIVE PARAMETRIZATION ***\n";
  // std::cout << "Interval: " << parametrization_interval_ << std::endl;
  // std::cout << projective_param.get_minimum()
  //<< "  " << projective_param.get_normal_minimum() << std::endl;

  ScalarT min_detA = projective_param.get_minimum();
  for (int i(0); i < 3; ++i) {
    direction(i) = (projective_param.get_normal_minimum())(i);
  }

  for (int i(0); i < 3; ++i) {
    arg_minimum(i) = (projective_param.get_arg_minimum())(i);
  }

  return min_detA;
}
//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
typename EvalT::ScalarT BifurcationCheck<EvalT, Traits>::tangent_sweep(
    minitensor::Tensor4<ScalarT, 3> const& tangent,
    minitensor::Vector<ScalarT, 2>&        arg_minimum,
    minitensor::Vector<ScalarT, 3>&        direction,
    double const&                          interval)
{
  minitensor::Index const p_number = floor(1.0 / interval);

  ScalarT const domain_min = -std::acos(-1.0) / 2.0;

  ScalarT const domain_max = std::acos(-1.0) / 2.0;

  ScalarT const p_mean = (domain_max + domain_min) / 2.0;

  ScalarT const p_span = domain_max - domain_min;

  ScalarT const p_min = p_mean - p_span / 2.0 * interval * p_number;
  // p_min = domain_min;

  ScalarT const p_max = p_mean + p_span / 2.0 * interval * p_number;
  // p_max = domain_min + p_span * interval * p_number;

  // Initialize parametres
  ScalarT const x_min = p_min;

  ScalarT const x_max = p_max;

  ScalarT const y_min = p_min;

  ScalarT const y_max = p_max;

  minitensor::Index const x_num_points = p_number * 2 + 1;
  // x_num_points = p_number + 1;

  minitensor::Index const y_num_points = p_number * 2 + 1;
  // y_num_points = p_number + 1;

  minitensor::Vector<ScalarT, 2> const tangent_min(x_min, y_min);

  minitensor::Vector<ScalarT, 2> const tangent_max(x_max, y_max);

  minitensor::Vector<minitensor::Index, 2> const tangent_num_points(
      x_num_points, y_num_points);

  // Build the parametric grid with the specified parameters.
  minitensor::ParametricGrid<ScalarT, 2> tangent_grid(
      tangent_min, tangent_max, tangent_num_points);

  // Build a tangent parametrization for this elasticity.
  minitensor::TangentParametrization<ScalarT, 3> tangent_param(tangent);

  // Traverse the grid with the parametrization.
  tangent_grid.traverse(tangent_param);

  // Query the parametrization for the minimum and maximum found on the grid.
  // std::cout << "\n*** TANGENT PARAMETRIZATION ***\n";
  // std::cout << "Interval: " << parametrization_interval_ << std::endl;
  // std::cout << tangent_param.get_minimum()
  //<< "  " << tangent_param.get_normal_minimum() << std::endl;

  ScalarT min_detA = tangent_param.get_minimum();
  for (int i(0); i < 3; ++i) {
    direction(i) = (tangent_param.get_normal_minimum())(i);
  }

  for (int i(0); i < 2; ++i) {
    arg_minimum(i) = (tangent_param.get_arg_minimum())(i);
  }

  return min_detA;
}
//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
typename EvalT::ScalarT BifurcationCheck<EvalT, Traits>::cartesian_sweep(
    minitensor::Tensor4<ScalarT, 3> const& tangent,
    minitensor::Vector<ScalarT, 2>&        arg_minimum,
    int                                    surface_index,
    minitensor::Vector<ScalarT, 3>&        direction,
    double const&                          interval)
{
  minitensor::Index const p_number = floor(1.0 / interval);

  ScalarT const domain_min = -1.0;

  ScalarT const domain_max = 1.0;

  ScalarT const p_mean = (domain_max + domain_min) / 2.0;

  ScalarT const p_span = domain_max - domain_min;

  ScalarT const p_min = p_mean - p_span / 2.0 * interval * p_number;
  // p_min = domain_min;

  ScalarT const p_max = p_mean + p_span / 2.0 * interval * p_number;
  // p_max = domain_min + p_span * interval * p_number;

  // Initialize parametres
  ScalarT const p_surface = 1.0;

  minitensor::Index const p_num_points = p_number * 2 + 1;
  // p_num_points = p_number + 1;

  minitensor::Index const p_surface_num_points = 1;

  ScalarT min_detA(1.0);

  if (surface_index == 1) {
    // x surface
    minitensor::Vector<ScalarT, 3> const cartesian1_min(
        p_surface, p_min, p_min);

    minitensor::Vector<ScalarT, 3> const cartesian1_max(
        p_surface, p_max, p_max);

    minitensor::Vector<minitensor::Index, 3> const cartesian1_num_points(
        p_surface_num_points, p_num_points, p_num_points);

    // Build the parametric grid with the specified parameters.
    minitensor::ParametricGrid<ScalarT, 3> cartesian1_grid(
        cartesian1_min, cartesian1_max, cartesian1_num_points);

    // Build a cartesian parametrization for this elasticity.
    minitensor::CartesianParametrization<ScalarT, 3> cartesian1_param(tangent);

    // Traverse the grid with the parametrization.
    cartesian1_grid.traverse(cartesian1_param);

    // Query the parametrization for the minimum and maximum found on the grid.
    // std::cout << "\n*** CARTESIAN PARAMETRIZATION ***\n";
    // std::cout << "Interval: " << parametrization_interval_ << std::endl;
    // std::cout << cartesian1_param.get_minimum()
    // << "  " << cartesian1_param.get_normal_minimum() << std::endl;

    min_detA       = cartesian1_param.get_minimum();
    arg_minimum(0) = (cartesian1_param.get_arg_minimum())(1);
    arg_minimum(1) = (cartesian1_param.get_arg_minimum())(2);
    for (int i(0); i < 3; ++i) {
      direction(i) = (cartesian1_param.get_normal_minimum())(i);
    }
  }

  if (surface_index == 2) {
    // y surface
    minitensor::Vector<ScalarT, 3> const cartesian2_min(
        p_min, p_surface, p_min);

    minitensor::Vector<ScalarT, 3> const cartesian2_max(
        p_max, p_surface, p_max);

    minitensor::Vector<minitensor::Index, 3> const cartesian2_num_points(
        p_num_points, p_surface_num_points, p_num_points);

    // Build the parametric grid with the specified parameters.
    minitensor::ParametricGrid<ScalarT, 3> cartesian2_grid(
        cartesian2_min, cartesian2_max, cartesian2_num_points);

    // Build a cartesian parametrization for this elasticity.
    minitensor::CartesianParametrization<ScalarT, 3> cartesian2_param(tangent);

    // Traverse the grid with the parametrization.
    cartesian2_grid.traverse(cartesian2_param);

    // Query the parametrization for the minimum and maximum found on the grid.
    // std::cout << "\n*** CARTESIAN PARAMETRIZATION ***\n";
    // std::cout << "Interval: " << parametrization_interval_ << std::endl;
    // std::cout << cartesian2_param.get_minimum()
    // << "  " << cartesian2_param.get_normal_minimum() << std::endl;

    min_detA       = cartesian2_param.get_minimum();
    arg_minimum(0) = (cartesian2_param.get_arg_minimum())(0);
    arg_minimum(1) = (cartesian2_param.get_arg_minimum())(2);
    for (int i(0); i < 3; ++i) {
      direction(i) = (cartesian2_param.get_normal_minimum())(i);
    }
  }

  if (surface_index == 3) {
    // z surface
    minitensor::Vector<ScalarT, 3> const cartesian3_min(
        p_min, p_min, p_surface);

    minitensor::Vector<ScalarT, 3> const cartesian3_max(
        p_max, p_max, p_surface);

    minitensor::Vector<minitensor::Index, 3> const cartesian3_num_points(
        p_num_points, p_num_points, p_surface_num_points);

    // Build the parametric grid with the specified parameters.
    minitensor::ParametricGrid<ScalarT, 3> cartesian3_grid(
        cartesian3_min, cartesian3_max, cartesian3_num_points);

    // Build a cartesian parametrization for this elasticity.
    minitensor::CartesianParametrization<ScalarT, 3> cartesian3_param(tangent);

    // Traverse the grid with the parametrization.
    cartesian3_grid.traverse(cartesian3_param);

    // Query the parametrization for the minimum and maximum found on the grid.
    // std::cout << "\n*** CARTESIAN PARAMETRIZATION ***\n";
    // std::cout << "Interval: " << parametrization_interval_ << std::endl;
    // std::cout << cartesian3_param.get_minimum()
    //<< "  " << cartesian3_param.get_normal_minimum() << std::endl;

    min_detA       = cartesian3_param.get_minimum();
    arg_minimum(0) = (cartesian3_param.get_arg_minimum())(0);
    arg_minimum(1) = (cartesian3_param.get_arg_minimum())(1);
    for (int i(0); i < 3; ++i) {
      direction(i) = (cartesian3_param.get_normal_minimum())(i);
    }
  }

  return min_detA;
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void BifurcationCheck<EvalT, Traits>::spherical_newton_raphson(
    minitensor::Tensor4<ScalarT, 3> const& tangent,
    minitensor::Vector<ScalarT, 2>&        parameters,
    minitensor::Vector<ScalarT, 3>&        direction,
    ScalarT&                               min_detA)
{
  minitensor::Vector<ScalarT, 2>   Xval;
  minitensor::Vector<DFadType, 2>  Xfad;
  minitensor::Vector<D2FadType, 2> Xfad2;
  minitensor::Vector<DFadType, 2>  Rfad;
  minitensor::Vector<D2FadType, 3> n;

  D2FadType detA;

  // in std:vector form to work with nonlinear solve
  // size of Jacobian, 4 = 2 * 2
  std::vector<ScalarT> dRdX(4);
  std::vector<ScalarT> X(2);
  std::vector<ScalarT> R(2);

  for (int i(0); i < 2; ++i) X[i] = parameters[i];

  // local nonlinear solver for Newton iterative solve
  LocalNonlinearSolver<EvalT, Traits> solver;

  ScalarT normR(0.0), normR0(0.0), relativeR(0.0);
  bool    converged = false;
  int     iter      = 0;

  while (!converged) {
    // std::cout << "iter: " << iter << std::endl;

    for (int i = 0; i < 2; ++i) {
      Xval[i]  = Sacado::ScalarValue<ScalarT>::eval(X[i]);
      Xfad[i]  = DFadType(2, i, Xval[i]);
      Xfad2[i] = D2FadType(2, i, Xfad[i]);
    }

    n = spherical_get_normal(Xfad2);

    detA = minitensor::det(minitensor::dot2(n, minitensor::dot(tangent, n)));

    // std::cout << "parameters: " << parameters << std::endl;
    // std::cout << "determinant: " << (detA.val()).val() << std::endl;

    for (int i = 0; i < 2; ++i) {
      Rfad[i] = detA.dx(i);
      R[i]    = Rfad[i].val();
    }

    // std::cout << "R: " << R << std::endl;

    normR = sqrt(R[0] * R[0] + R[1] * R[1]);

    if (iter == 0) normR0 = normR;

    // std::cout << "normR: " << normR << std::endl;

    if (normR0 != 0)
      relativeR = normR / normR0;
    else
      relativeR = normR0;

    if (relativeR < 1.0e-8 || normR < 1.0e-8) break;

    if (iter > 50) {
      std::cout << "Newton's loop for bifurcation check not converging after "
                << 50 << " iterations" << std::endl;
      break;
    }

    // compute Jacobian
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) dRdX[i + 2 * j] = Rfad[i].dx(j);

    // call local nonlinear solver
    solver.solve(dRdX, X, R);

    iter++;

  }  // Newton Raphson iteration end

  // compute sensitivity information w.r.t. system parameters
  // and pack the sensitivity back to X
  // solver.computeFadInfo(dRdX, X, R);

  // update
  if ((detA.val()).val() <= min_detA) {
    for (int i(0); i < 3; ++i) direction[i] = (n[i].val()).val();

    for (int i(0); i < 2; ++i) parameters[i] = X[i];

    min_detA = (detA.val()).val();

  } else {
    std::cout << "Newnton's loop for bifurcation check fails to identify "
                 "minimum det(A)"
              << std::endl;
  }

}  // Function end

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void BifurcationCheck<EvalT, Traits>::stereographic_newton_raphson(
    minitensor::Tensor4<ScalarT, 3> const& tangent,
    minitensor::Vector<ScalarT, 2>&        parameters,
    minitensor::Vector<ScalarT, 3>&        direction,
    ScalarT&                               min_detA)
{
  minitensor::Vector<ScalarT, 2>   Xval;
  minitensor::Vector<DFadType, 2>  Xfad;
  minitensor::Vector<D2FadType, 2> Xfad2;
  minitensor::Vector<DFadType, 2>  Rfad;
  minitensor::Vector<D2FadType, 3> n;

  D2FadType detA;

  // in std:vector form to work with nonlinear solve
  // size of Jacobian, 4 = 2 * 2
  std::vector<ScalarT> dRdX(4);
  std::vector<ScalarT> X(2);
  std::vector<ScalarT> R(2);

  for (int i(0); i < 2; ++i) X[i] = parameters[i];

  // local nonlinear solver for Newton iterative solve
  LocalNonlinearSolver<EvalT, Traits> solver;

  ScalarT normR(0.0), normR0(0.0), relativeR(0.0);
  bool    converged = false;
  int     iter      = 0;

  while (!converged) {
    // std::cout << "iter: " << iter << std::endl;

    for (int i = 0; i < 2; ++i) {
      Xval[i]  = Sacado::ScalarValue<ScalarT>::eval(X[i]);
      Xfad[i]  = DFadType(2, i, Xval[i]);
      Xfad2[i] = D2FadType(2, i, Xfad[i]);
    }

    n = stereographic_get_normal(Xfad2);

    detA = minitensor::det(minitensor::dot2(n, minitensor::dot(tangent, n)));

    // std::cout << "parameters: " << parameters << std::endl;
    // std::cout << "determinant: " << (detA.val()).val() << std::endl;

    for (int i = 0; i < 2; ++i) {
      Rfad[i] = detA.dx(i);
      R[i]    = Rfad[i].val();
    }

    // std::cout << "R: " << R << std::endl;

    normR = sqrt(R[0] * R[0] + R[1] * R[1]);

    if (iter == 0) normR0 = normR;

    // std::cout << "normR: " << normR << std::endl;

    if (normR0 != 0)
      relativeR = normR / normR0;
    else
      relativeR = normR0;

    if (relativeR < 1.0e-8 || normR < 1.0e-8) break;

    if (iter > 50) {
      std::cout << "Newton's loop for bifurcation check not converging after "
                << 50 << " iterations" << std::endl;
      break;
    }

    // compute Jacobian
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) dRdX[i + 2 * j] = Rfad[i].dx(j);

    // call local nonlinear solver
    solver.solve(dRdX, X, R);

    iter++;

  }  // Newton Raphson iteration end

  // compute sensitivity information w.r.t. system parameters
  // and pack the sensitivity back to X
  // solver.computeFadInfo(dRdX, X, R);

  // update
  if ((detA.val()).val() <= min_detA) {
    for (int i(0); i < 3; ++i) direction[i] = (n[i].val()).val();

    for (int i(0); i < 2; ++i) parameters[i] = X[i];

    min_detA = (detA.val()).val();

  } else {
    std::cout << "Newnton's loop for bifurcation check fails to identify "
                 "minimum det(A)"
              << std::endl;
  }

}  // Function end

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void BifurcationCheck<EvalT, Traits>::projective_newton_raphson(
    minitensor::Tensor4<ScalarT, 3> const& tangent,
    minitensor::Vector<ScalarT, 3>&        parameters,
    minitensor::Vector<ScalarT, 3>&        direction,
    ScalarT&                               min_detA)
{
  minitensor::Vector<ScalarT, 4> parameters_new;
  ScalarT                        nNorm = minitensor::norm(parameters);
  for (int i = 0; i < 3; ++i) {
    parameters_new[i] = parameters[i];
    if (nNorm == 0) parameters_new[i] = 1.0;
  }
  parameters_new[3] = 0;

  minitensor::Vector<ScalarT, 4>   Xval;
  minitensor::Vector<DFadType, 4>  Xfad;
  minitensor::Vector<D2FadType, 4> Xfad2;
  minitensor::Vector<DFadType, 4>  Rfad;
  minitensor::Vector<D2FadType, 3> n;

  D2FadType detA;

  // in std:vector form to work with nonlinear solve
  // size of Jacobian, 4 = 2 * 2
  std::vector<ScalarT> dRdX(16);
  std::vector<ScalarT> X(4);
  std::vector<ScalarT> R(4);

  for (int i(0); i < 4; ++i) X[i] = parameters_new[i];

  // local nonlinear solver for Newton iterative solve
  LocalNonlinearSolver<EvalT, Traits> solver;

  ScalarT normR(0.0), normR0(0.0), relativeR(0.0);
  bool    converged = false;
  int     iter      = 0;

  while (!converged) {
    // std::cout << "iter: " << iter << std::endl;

    for (int i = 0; i < 4; ++i) {
      Xval[i]  = Sacado::ScalarValue<ScalarT>::eval(X[i]);
      Xfad[i]  = DFadType(4, i, Xval[i]);
      Xfad2[i] = D2FadType(4, i, Xfad[i]);
    }

    minitensor::Vector<D2FadType, 3> Xfad2_sub;
    for (int i = 0; i < 3; ++i) { Xfad2_sub[i] = Xfad2[i]; }
    n = projective_get_normal(Xfad2_sub);

    detA = minitensor::det(minitensor::dot2(n, minitensor::dot(tangent, n))) +
           Xfad2[3] * (Xfad2[0] * Xfad2[0] + Xfad2[1] * Xfad2[1] +
                       Xfad2[2] * Xfad2[2] - 1);

    // std::cout << "parameters: " << parameters << std::endl;
    // std::cout << "determinant: " << (detA.val()).val() << std::endl;

    for (int i = 0; i < 4; ++i) {
      Rfad[i] = detA.dx(i);
      R[i]    = Rfad[i].val();
    }

    // std::cout << "R: " << R << std::endl;

    normR = sqrt(R[0] * R[0] + R[1] * R[1] + R[2] * R[2] + R[3] * R[3]);

    if (iter == 0) normR0 = normR;

    // std::cout << "normR: " << normR << std::endl;

    if (normR0 != 0)
      relativeR = normR / normR0;
    else
      relativeR = normR0;

    if (relativeR < 1.0e-8 || normR < 1.0e-8) break;

    if (iter > 50) {
      std::cout << "Newton's loop for bifurcation check not converging after "
                << 50 << " iterations" << std::endl;
      break;
    }

    // compute Jacobian
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) dRdX[i + 4 * j] = Rfad[i].dx(j);

    // call local nonlinear solver
    solver.solve(dRdX, X, R);

    iter++;

  }  // Newton Raphson iteration end

  // compute sensitivity information w.r.t. system parameters
  // and pack the sensitivity back to X
  // solver.computeFadInfo(dRdX, X, R);

  // update
  if ((detA.val()).val() <= min_detA) {
    for (int i(0); i < 3; ++i) direction[i] = (n[i].val()).val();

    for (int i(0); i < 3; ++i) parameters[i] = X[i];

    min_detA = (detA.val()).val();

  } else {
    std::cout << "Newnton's loop for bifurcation check fails to identify "
                 "minimum det(A)"
              << std::endl;
  }

}  // Function end

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void BifurcationCheck<EvalT, Traits>::tangent_newton_raphson(
    minitensor::Tensor4<ScalarT, 3> const& tangent,
    minitensor::Vector<ScalarT, 2>&        parameters,
    minitensor::Vector<ScalarT, 3>&        direction,
    ScalarT&                               min_detA)
{
  minitensor::Vector<ScalarT, 2>   Xval;
  minitensor::Vector<DFadType, 2>  Xfad;
  minitensor::Vector<D2FadType, 2> Xfad2;
  minitensor::Vector<DFadType, 2>  Rfad;
  minitensor::Vector<D2FadType, 3> n;

  D2FadType detA;

  // in std:vector form to work with nonlinear solve
  // size of Jacobian, 4 = 2 * 2
  std::vector<ScalarT> dRdX(4);
  std::vector<ScalarT> X(2);
  std::vector<ScalarT> R(2);

  for (int i(0); i < 2; ++i) X[i] = parameters[i];

  // local nonlinear solver for Newton iterative solve
  LocalNonlinearSolver<EvalT, Traits> solver;

  ScalarT normR(0.0), normR0(0.0), relativeR(0.0);
  bool    converged = false;
  int     iter      = 0;

  while (!converged) {
    // std::cout << "iter: " << iter << std::endl;

    for (int i = 0; i < 2; ++i) {
      Xval[i]  = Sacado::ScalarValue<ScalarT>::eval(X[i]);
      Xfad[i]  = DFadType(2, i, Xval[i]);
      Xfad2[i] = D2FadType(2, i, Xfad[i]);
    }

    n = tangent_get_normal(Xfad2);

    detA = minitensor::det(minitensor::dot2(n, minitensor::dot(tangent, n)));

    // std::cout << "parameters: " << parameters << std::endl;
    // std::cout << "determinant: " << (detA.val()).val() << std::endl;

    for (int i = 0; i < 2; ++i) {
      Rfad[i] = detA.dx(i);
      R[i]    = Rfad[i].val();
    }

    // std::cout << "R: " << R << std::endl;

    normR = sqrt(R[0] * R[0] + R[1] * R[1]);

    if (iter == 0) normR0 = normR;

    // std::cout << "normR: " << normR << std::endl;

    if (normR0 != 0)
      relativeR = normR / normR0;
    else
      relativeR = normR0;

    if (relativeR < 1.0e-8 || normR < 1.0e-8) break;

    if (iter > 50) {
      std::cout << "Newton's loop for bifurcation check not converging after "
                << 50 << " iterations" << std::endl;
      break;
    }

    // compute Jacobian
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) dRdX[i + 2 * j] = Rfad[i].dx(j);

    // call local nonlinear solver
    solver.solve(dRdX, X, R);

    iter++;

  }  // Newton Raphson iteration end

  // compute sensitivity information w.r.t. system parameters
  // and pack the sensitivity back to X
  // solver.computeFadInfo(dRdX, X, R);

  // update
  if ((detA.val()).val() <= min_detA) {
    for (int i(0); i < 3; ++i) direction[i] = (n[i].val()).val();

    for (int i(0); i < 2; ++i) parameters[i] = X[i];

    min_detA = (detA.val()).val();

  } else {
    std::cout << "Newnton's loop for bifurcation check fails to identify "
                 "minimum det(A)"
              << std::endl;
  }

}  // Function end

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void BifurcationCheck<EvalT, Traits>::cartesian_newton_raphson(
    minitensor::Tensor4<ScalarT, 3> const& tangent,
    minitensor::Vector<ScalarT, 2>&        parameters,
    int                                    surface_index,
    minitensor::Vector<ScalarT, 3>&        direction,
    ScalarT&                               min_detA)
{
  minitensor::Vector<ScalarT, 2>   Xval;
  minitensor::Vector<DFadType, 2>  Xfad;
  minitensor::Vector<D2FadType, 2> Xfad2;
  minitensor::Vector<DFadType, 2>  Rfad;
  minitensor::Vector<D2FadType, 3> n;

  D2FadType detA;

  // in std:vector form to work with nonlinear solve
  // size of Jacobian, 4 = 2 * 2
  std::vector<ScalarT> dRdX(4);
  std::vector<ScalarT> X(2);
  std::vector<ScalarT> R(2);

  for (int i(0); i < 2; ++i) X[i] = parameters[i];

  // local nonlinear solver for Newton iterative solve
  LocalNonlinearSolver<EvalT, Traits> solver;

  ScalarT normR(0.0), normR0(0.0), relativeR(0.0);
  bool    converged = false;
  int     iter      = 0;

  while (!converged) {
    // std::cout << "iter: " << iter << std::endl;

    for (int i = 0; i < 2; ++i) {
      Xval[i]  = Sacado::ScalarValue<ScalarT>::eval(X[i]);
      Xfad[i]  = DFadType(2, i, Xval[i]);
      Xfad2[i] = D2FadType(2, i, Xfad[i]);
    }

    switch (surface_index) {
      case 1: n = cartesian_get_normal1(Xfad2); break;
      case 2: n = cartesian_get_normal2(Xfad2); break;
      case 3: n = cartesian_get_normal3(Xfad2); break;
      default: n = cartesian_get_normal1(Xfad2); break;
    }

    detA = minitensor::det(minitensor::dot2(n, minitensor::dot(tangent, n)));

    // std::cout << "parameters: " << parameters << std::endl;
    // std::cout << "determinant: " << (detA.val()).val() << std::endl;

    for (int i = 0; i < 2; ++i) {
      Rfad[i] = detA.dx(i);
      R[i]    = Rfad[i].val();
    }

    // std::cout << "R: " << R << std::endl;

    normR = sqrt(R[0] * R[0] + R[1] * R[1]);

    if (iter == 0) normR0 = normR;

    // std::cout << "normR: " << normR << std::endl;

    if (normR0 != 0)
      relativeR = normR / normR0;
    else
      relativeR = normR0;

    if (relativeR < 1.0e-8 || normR < 1.0e-8) break;

    if (iter > 50) {
      std::cout << "Newton's loop for bifurcation check not converging after "
                << 50 << " iterations" << std::endl;
      break;
    }

    // compute Jacobian
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) dRdX[i + 2 * j] = Rfad[i].dx(j);

    // call local nonlinear solver
    solver.solve(dRdX, X, R);

    iter++;

  }  // Newton Raphson iteration end

  // compute sensitivity information w.r.t. system parameters
  // and pack the sensitivity back to X
  // solver.computeFadInfo(dRdX, X, R);

  // update
  if ((detA.val()).val() <= min_detA) {
    for (int i(0); i < 3; ++i) direction[i] = (n[i].val()).val();

    ScalarT dirNorm = minitensor::norm(direction);
    for (int i(0); i < 3; ++i) direction[i] /= dirNorm;

    for (int i(0); i < 2; ++i) parameters[i] = X[i];

    min_detA = (detA.val()).val() / std::pow(dirNorm, 6);

  } else {
    std::cout << "Newnton's loop for bifurcation check fails to identify "
                 "minimum det(A)"
              << std::endl;
  }

}  // Function end

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
typename EvalT::ScalarT BifurcationCheck<EvalT, Traits>::stereographic_pso(
    minitensor::Tensor4<ScalarT, 3> const& tangent,
    minitensor::Vector<ScalarT, 2>&        arg_minimum,
    minitensor::Vector<ScalarT, 3>&        direction)
{
  double w  = 0.7;
  double c1 = 0.5;
  double c2 = 0.5;
  double r  = 1.0;

  int const group_size = 10;

  std::vector<minitensor::Vector<ScalarT, 2>> arg_group(group_size);
  std::vector<minitensor::Vector<ScalarT, 2>> arg_velocity_group(group_size);

  std::vector<minitensor::Vector<ScalarT, 2>> arg_ibest(group_size);
  std::vector<ScalarT>                        detA_ibest(group_size);

  minitensor::Vector<ScalarT, 2> arg_gbest;
  ScalarT detA_gbest = std::numeric_limits<ScalarT>::max();

  std::random_device                     rd;
  std::mt19937                           mt_eng(rd());
  std::uniform_real_distribution<double> real_dist(-1.0, 1.0);

  for (int i = 0; i < group_size; i++) {
    minitensor::Vector<ScalarT, 2> arg_tmp;
    minitensor::Vector<ScalarT, 2> arg_velocity_tmp;

    for (int j = 0; j < 2; j++) {
      arg_tmp(j)          = real_dist(mt_eng);
      arg_velocity_tmp(j) = real_dist(mt_eng) * 0.2;
    }

    arg_group[i]          = arg_tmp;
    arg_velocity_group[i] = arg_velocity_tmp;

    ScalarT r2 = arg_tmp[0] * arg_tmp[0] + arg_tmp[1] * arg_tmp[1];

    minitensor::Vector<ScalarT, 3> n(
        2.0 * arg_tmp[0], 2.0 * arg_tmp[1], r2 - 1.0);
    n /= (r2 + 1.0);

    arg_ibest[i] = arg_tmp;
    detA_ibest[i] =
        minitensor::det(minitensor::dot2(n, minitensor::dot(tangent, n)));

    if (detA_gbest > detA_ibest[i]) {
      detA_gbest = detA_ibest[i];
      arg_gbest  = arg_group[i];
    }
  }  // group initialization

  bool      converged = false;
  int       iter      = 0;
  int const iter_max  = 1000;
  ScalarT   error0    = 1.0;
  while (!converged) {
    ScalarT error = 0.0;

    for (int i = 0; i < group_size; i++) {
      arg_velocity_group[i] =
          w * arg_velocity_group[i] +
          c1 * real_dist(mt_eng) * (arg_ibest[i] - arg_group[i]) +
          c2 * real_dist(mt_eng) * (arg_gbest - arg_group[i]);
      arg_group[i] += r * arg_velocity_group[i];

      minitensor::Vector<ScalarT, 2> arg_tmp = arg_group[i];

      ScalarT r2 = arg_tmp[0] * arg_tmp[0] + arg_tmp[1] * arg_tmp[1];

      minitensor::Vector<ScalarT, 3> n(
          2.0 * arg_tmp[0], 2.0 * arg_tmp[1], r2 - 1.0);
      n /= (r2 + 1.0);

      ScalarT detA_tmp =
          minitensor::det(minitensor::dot2(n, minitensor::dot(tangent, n)));

      if (detA_ibest[i] > detA_tmp) {
        detA_ibest[i] = detA_tmp;
        arg_ibest[i]  = arg_tmp;
      }

      if (detA_gbest > detA_ibest[i]) {
        detA_gbest = detA_ibest[i];
        arg_gbest  = arg_group[i];
      }

      // error += abs(detA_ibest[i] - detA_gbest);
      error += (detA_tmp - detA_gbest) * (detA_tmp - detA_gbest);
    }

    // error /= group_size;
    error = sqrt(error / (group_size - 1));
    if (iter == 0) error0 = error;
    if (error <= 1E-11 || error / error0 <= 1E-11) break;

    iter++;
    if (iter > iter_max) break;

  }  // group generation iteration

  ScalarT r2 = arg_gbest[0] * arg_gbest[0] + arg_gbest[1] * arg_gbest[1];

  minitensor::Vector<ScalarT, 3> n(
      2.0 * arg_gbest[0], 2.0 * arg_gbest[1], r2 - 1.0);
  n /= (r2 + 1.0);

  for (int i(0); i < 3; ++i) { direction(i) = n(i); }

  for (int i(0); i < 2; ++i) { arg_minimum(i) = arg_gbest(i); }

  return detA_gbest;
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
minitensor::Vector<typename BifurcationCheck<EvalT, Traits>::D2FadType, 3>
    BifurcationCheck<EvalT, Traits>::spherical_get_normal(
        minitensor::Vector<D2FadType, 2>& parameters)
{
  minitensor::Vector<D2FadType, 3> normal(
      sin(parameters[0]) * cos(parameters[1]),
      sin(parameters[0]) * sin(parameters[1]),
      cos(parameters[0]));

  return normal;
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
minitensor::Vector<typename BifurcationCheck<EvalT, Traits>::D2FadType, 3>
    BifurcationCheck<EvalT, Traits>::stereographic_get_normal(
        minitensor::Vector<D2FadType, 2>& parameters)
{
  D2FadType r2 = parameters[0] * parameters[0] + parameters[1] * parameters[1];

  minitensor::Vector<D2FadType, 3> normal(
      2.0 * parameters[0], 2.0 * parameters[1], r2 - 1.0);
  normal /= (r2 + 1.0);

  return normal;
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
minitensor::Vector<typename BifurcationCheck<EvalT, Traits>::D2FadType, 3>
    BifurcationCheck<EvalT, Traits>::projective_get_normal(
        minitensor::Vector<D2FadType, 3>& parameters)
{
  minitensor::Vector<D2FadType, 3>& normal = parameters;

  D2FadType const n = minitensor::norm(normal);

  if ((n.val()).val() != 0) {
    // normal /= n;
  } else {
    minitensor::Vector<DFadType, 3>  Xfad;
    minitensor::Vector<D2FadType, 3> Xfad2;
    for (int i = 0; i < 3; ++i) {
      Xfad[i]       = DFadType(3, i, 1.0 / sqrt(3.0));
      Xfad2[i]      = D2FadType(3, i, Xfad[i]);
      parameters[i] = Xfad2[i];
    }
    normal[0] = Xfad2[0];
    normal[1] = Xfad2[1];
    normal[2] = Xfad2[2];
  }

  return normal;
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
minitensor::Vector<typename BifurcationCheck<EvalT, Traits>::D2FadType, 3>
    BifurcationCheck<EvalT, Traits>::tangent_get_normal(
        minitensor::Vector<D2FadType, 2>& parameters)
{
  D2FadType const r =
      sqrt(parameters[0] * parameters[0] + parameters[1] * parameters[1]);

  minitensor::Vector<D2FadType, 3> normal(3, minitensor::Filler::ZEROS);

  if ((r.val()).val() > 0.0) {
    normal[0] = parameters[0] * sin(r) / r;
    normal[1] = parameters[1] * sin(r) / r;
    normal[2] = cos(r);
  } else {
    normal[0] = parameters[0];
    normal[1] = parameters[1];
    normal[2] = cos(r);
  }

  return normal;
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
minitensor::Vector<typename BifurcationCheck<EvalT, Traits>::D2FadType, 3>
    BifurcationCheck<EvalT, Traits>::cartesian_get_normal1(
        minitensor::Vector<D2FadType, 2>& parameters)
{
  minitensor::Vector<D2FadType, 3> normal(
      D2FadType(1), parameters[0], parameters[1]);

  return normal;
}

template <typename EvalT, typename Traits>
minitensor::Vector<typename BifurcationCheck<EvalT, Traits>::D2FadType, 3>
    BifurcationCheck<EvalT, Traits>::cartesian_get_normal2(
        minitensor::Vector<D2FadType, 2>& parameters)
{
  minitensor::Vector<D2FadType, 3> normal(
      parameters[0], D2FadType(1), parameters[1]);

  return normal;
}

template <typename EvalT, typename Traits>
minitensor::Vector<typename BifurcationCheck<EvalT, Traits>::D2FadType, 3>
    BifurcationCheck<EvalT, Traits>::cartesian_get_normal3(
        minitensor::Vector<D2FadType, 2>& parameters)
{
  minitensor::Vector<D2FadType, 3> normal(
      parameters[0], parameters[1], D2FadType(1));

  return normal;
}
}  // namespace LCM
