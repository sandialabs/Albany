//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
// Simple mesh partitioning program
//

#include <iomanip>

#include <Intrepid_MiniTensor.h>

// No idea why this is required but build fails if missing.
bool TpetraBuild = false;

//
// Simple tests for parametrizations of the bifurcation tensor.
//
int main(int ac, char* av[])
{
  // Set the random seed for reproducibility.
  Teuchos::ScalarTraits<double>().seedrandom(0);

  // Build an elasticity tensor and perturb it with some random values.
  // Warning: this is linear elasticity.
  // Steel properties.
  double const
  lambda = 9.7e10;

  double const
  mu = 7.6e10;

  Intrepid::Tensor4<double, 3> const
  I1 = Intrepid::identity_1<double, 3>() +
    0.1 * Intrepid::Tensor4<double, 3>(Intrepid::RANDOM_NORMAL);

  Intrepid::Tensor4<double, 3> const
  I2 = Intrepid::identity_2<double, 3>() +
    0.1 * Intrepid::Tensor4<double, 3>(Intrepid::RANDOM_NORMAL);

  Intrepid::Tensor4<double, 3> const
  I3 = Intrepid::identity_3<double, 3>() +
    0.1 * Intrepid::Tensor4<double, 3>(Intrepid::RANDOM_NORMAL);

  Intrepid::Tensor4<double, 3> const
  CC = lambda * I3 + mu * (I1 + I2);

  // Build a spherical parametrization for this elasticity.
  Intrepid::SphericalParametrization<double, 3>
  sphere_param(CC);

  // Now build a grid to sample the parametrization.
  // The spherical parametrization has two parameters.
  // Set the limits for those parameters and the density of sampling
  // points for each parameter in vectors.
  double const
  pi = std::acos(-1.0);

  double const
  phi_min = 0.0;

  double const
  phi_max = pi;

  double const
  theta_min = 0.0;

  double const
  theta_max = 2.0 * pi;

  Intrepid::Index const
  phi_num_points = 256;

  Intrepid::Index const
  theta_num_points = 256;

  Intrepid::Vector<double, 2> const
  sphere_min(phi_min, theta_min);

  Intrepid::Vector<double, 2> const
  sphere_max(phi_max, theta_max);

  Intrepid::Vector<Intrepid::Index, 2> const
  sphere_num_points(phi_num_points, theta_num_points);

  // Build the parametric grid with the specified parameters.
  Intrepid::ParametricGrid<double, 2>
  sphere_grid(sphere_min, sphere_max, sphere_num_points);

  // Traverse the grid with the parametrization.
  sphere_grid.traverse(sphere_param);

  // Query the parametrization for the minimum and maximum found on the grid.
  std::cout << "\n*** SPHERICAL PARAMETRIZATION ***\n";

  std::cout << std::scientific << std::setprecision(16);

  std::cout << "Minimum: " << sphere_param.get_minimum();
  std::cout << " at " << sphere_param.get_arg_minimum() << '\n';
  std::cout << "Normal: " << sphere_param.get_normal_minimum() << '\n';

  std::cout << "Maximum: " << sphere_param.get_maximum();
  std::cout << " at " << sphere_param.get_arg_maximum() << '\n';
  std::cout << "Normal: " << sphere_param.get_normal_maximum() << '\n';

  return 0;

}
