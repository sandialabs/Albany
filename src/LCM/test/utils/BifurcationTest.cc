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
// Spherical parametrization sweep
//
void
spherical_sweep(Intrepid::Tensor4<double, 3> const & CC)
{
  // Build a grid to sample the parametrization.
  // The spherical parametrization has two parameters.
  // Set the limits and the density of sampling
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
  theta_max = pi;

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

  // Build a spherical parametrization for this elasticity.
  Intrepid::SphericalParametrization<double, 3>
  sphere_param(CC);

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

  return;
}

//
// Stereographic parametrization sweep
//
void
stereographic_sweep(Intrepid::Tensor4<double, 3> const & CC)
{
  // Build a grid to sample the parametrization.
  // The stereographic parametrization has two parameters.
  // Set the limits and the density of sampling
  // points for each parameter in vectors.
  double const
  x_min = -1.0;

  double const
  x_max = 1.0;

  double const
  y_min = -1.0;

  double const
  y_max = 1.0;

  Intrepid::Index const
  x_num_points = 256;

  Intrepid::Index const
  y_num_points = 256;

  Intrepid::Vector<double, 2> const
  stereo_min(x_min, y_min);

  Intrepid::Vector<double, 2> const
  stereo_max(x_max, y_max);

  Intrepid::Vector<Intrepid::Index, 2> const
  stereo_num_points(x_num_points, y_num_points);

  // Build the parametric grid with the specified parameters.
  Intrepid::ParametricGrid<double, 2>
  stereo_grid(stereo_min, stereo_max, stereo_num_points);

  // Build a stereographic parametrization for this elasticity.
  Intrepid::StereographicParametrization<double, 3>
  stereo_param(CC);

  // Traverse the grid with the parametrization.
  stereo_grid.traverse(stereo_param);

  // Query the parametrization for the minimum and maximum found on the grid.
  std::cout << "\n*** STEREOGRAPHIC PARAMETRIZATION ***\n";

  std::cout << std::scientific << std::setprecision(16);

  std::cout << "Minimum: " << stereo_param.get_minimum();
  std::cout << " at " << stereo_param.get_arg_minimum() << '\n';
  std::cout << "Normal: " << stereo_param.get_normal_minimum() << '\n';

  std::cout << "Maximum: " << stereo_param.get_maximum();
  std::cout << " at " << stereo_param.get_arg_maximum() << '\n';
  std::cout << "Normal: " << stereo_param.get_normal_maximum() << '\n';

  return;
}

//
// Projective parametrization sweep
//
void
projective_sweep(Intrepid::Tensor4<double, 3> const & CC)
{
  // Build a grid to sample the parametrization.
  // The projective parametrization has three parameters.
  // Set the limits and the density of sampling
  // points for each parameter in vectors.
  double const
  x_min = -1.0;

  double const
  x_max = 1.0;

  double const
  y_min = -1.0;

  double const
  y_max = 1.0;

  double const
  z_min = -1.0;

  double const
  z_max = 1.0;

  Intrepid::Index const
  x_num_points = 64;

  Intrepid::Index const
  y_num_points = 64;

  Intrepid::Index const
  z_num_points = 64;

  Intrepid::Vector<double, 3> const
  project_min(x_min, y_min, z_min);

  Intrepid::Vector<double, 3> const
  project_max(x_max, y_max, z_max);

  Intrepid::Vector<Intrepid::Index, 3> const
  project_num_points(x_num_points, y_num_points, z_num_points);

  // Build the parametric grid with the specified parameters.
  Intrepid::ParametricGrid<double, 3>
  project_grid(project_min, project_max, project_num_points);

  // Build a projective parametrization for this elasticity.
  Intrepid::ProjectiveParametrization<double, 3>
  project_param(CC);

  // Traverse the grid with the parametrization.
  project_grid.traverse(project_param);

  // Query the parametrization for the minimum and maximum found on the grid.
  std::cout << "\n*** PROJECTIVE PARAMETRIZATION ***\n";

  std::cout << std::scientific << std::setprecision(16);

  std::cout << "Minimum: " << project_param.get_minimum();
  std::cout << " at " << project_param.get_arg_minimum() << '\n';
  std::cout << "Normal: " << project_param.get_normal_minimum() << '\n';

  std::cout << "Maximum: " << project_param.get_maximum();
  std::cout << " at " << project_param.get_arg_maximum() << '\n';
  std::cout << "Normal: " << project_param.get_normal_maximum() << '\n';

  return;
}

//
// Tangent parametrization sweep
//
void
tangent_sweep(Intrepid::Tensor4<double, 3> const & CC)
{
  // Build a grid to sample the parametrization.
  // The tangent parametrization has two parameters.
  // Set the limits and the density of sampling
  // points for each parameter in vectors.
  double const
  x_min = -1.0;

  double const
  x_max = 1.0;

  double const
  y_min = -1.0;

  double const
  y_max = 1.0;

  Intrepid::Index const
  x_num_points = 256;

  Intrepid::Index const
  y_num_points = 256;

  Intrepid::Vector<double, 2> const
  tangent_min(x_min, y_min);

  Intrepid::Vector<double, 2> const
  tangent_max(x_max, y_max);

  Intrepid::Vector<Intrepid::Index, 2> const
  tangent_num_points(x_num_points, y_num_points);

  // Build the parametric grid with the specified parameters.
  Intrepid::ParametricGrid<double, 2>
  tangent_grid(tangent_min, tangent_max, tangent_num_points);

  // Build a tangent parametrization for this elasticity.
  Intrepid::TangentParametrization<double, 3>
  tangent_param(CC);

  // Traverse the grid with the parametrization.
  tangent_grid.traverse(tangent_param);

  // Query the parametrization for the minimum and maximum found on the grid.
  std::cout << "\n*** TANGENT PARAMETRIZATION ***\n";

  std::cout << std::scientific << std::setprecision(16);

  std::cout << "Minimum: " << tangent_param.get_minimum();
  std::cout << " at " << tangent_param.get_arg_minimum() << '\n';
  std::cout << "Normal: " << tangent_param.get_normal_minimum() << '\n';

  std::cout << "Maximum: " << tangent_param.get_maximum();
  std::cout << " at " << tangent_param.get_arg_maximum() << '\n';
  std::cout << "Normal: " << tangent_param.get_normal_maximum() << '\n';

  return;
}

//
// Cartesian parametrization sweep
//
void
cartesian_sweep(Intrepid::Tensor4<double, 3> const & CC)
{
  // Build a grid to sample the parametrization.
  // The cartesian parametrization has three parameters.
  // Set the limits and the density of sampling
  // points for each parameter in vectors.
  double const
  x_min = -1.0;

  double const
  x_max = 1.0;

  double const
  y_min = -1.0;

  double const
  y_max = 1.0;

  double const
  z_min = -1.0;

  double const
  z_max = 1.0;

  Intrepid::Index const
  x_num_points = 64;

  Intrepid::Index const
  y_num_points = 64;

  Intrepid::Index const
  z_num_points = 64;

  Intrepid::Vector<double, 3> const
  cartesian_min(x_min, y_min, z_min);

  Intrepid::Vector<double, 3> const
  cartesian_max(x_max, y_max, z_max);

  Intrepid::Vector<Intrepid::Index, 3> const
  cartesian_num_points(x_num_points, y_num_points, z_num_points);

  // Build the parametric grid with the specified parameters.
  Intrepid::ParametricGrid<double, 3>
  cartesian_grid(cartesian_min, cartesian_max, cartesian_num_points);

  // Build a projective parametrization for this elasticity.
  Intrepid::CartesianParametrization<double, 3>
  cartesian_param(CC);

  // Traverse the grid with the parametrization.
  cartesian_grid.traverse(cartesian_param);

  // Query the parametrization for the minimum and maximum found on the grid.
  std::cout << "\n*** CARTESIAN PARAMETRIZATION ***\n";

  std::cout << std::scientific << std::setprecision(16);

  std::cout << "Minimum: " << cartesian_param.get_minimum();
  std::cout << " at " << cartesian_param.get_arg_minimum() << '\n';
  std::cout << "Normal: " << cartesian_param.get_normal_minimum() << '\n';

  std::cout << "Maximum: " << cartesian_param.get_maximum();
  std::cout << " at " << cartesian_param.get_arg_maximum() << '\n';
  std::cout << "Normal: " << cartesian_param.get_normal_maximum() << '\n';

  return;
}

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

  spherical_sweep(CC);

  stereographic_sweep(CC);

  projective_sweep(CC);

  tangent_sweep(CC);

  cartesian_sweep(CC);

  return 0;

}
