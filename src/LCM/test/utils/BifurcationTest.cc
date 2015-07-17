//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
// Simple mesh partitioning program
//

#include <iomanip>
#include <iostream>
#include <random>

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_as.hpp>
#include <QCAD_MaterialDatabase.hpp>
#include <Phalanx.hpp>

#include <PHAL_AlbanyTraits.hpp>
#include <PHAL_SaveStateField.hpp>
#include <Albany_Utils.hpp>
#include <Albany_StateManager.hpp>
#include <Albany_TmplSTKMeshStruct.hpp>
#include <Albany_STKDiscretization.hpp>
#include <Albany_Layouts.hpp>
#include <Intrepid_MiniTensor.h>
#include <typeinfo>

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "Sacado.hpp"

#include "LocalNonlinearSolver.hpp"

typedef PHAL::AlbanyTraits::Residual Residual;
typedef PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
typedef PHAL::AlbanyTraits Traits;
typedef Sacado::mpl::apply<FadType,ScalarT>::type DFadType;
typedef Sacado::mpl::apply<FadType,DFadType>::type D2FadType;

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

//----------------------------------------------------------------------------
Intrepid::Vector<D2FadType, 3> 
spherical_get_normal(Intrepid::Vector<D2FadType, 2> & parameters)
{
  Intrepid::Vector<D2FadType, 3> 
  normal(sin(parameters[0]) * sin(parameters[1]), 
  cos(parameters[0]), sin(parameters[0]) * cos(parameters[1]));
  
  return normal;
}
  
//----------------------------------------------------------------------------  
Intrepid::Vector<D2FadType, 3> 
stereographic_get_normal(Intrepid::Vector<D2FadType, 2> & parameters)
{
  D2FadType r2 = parameters[0] * parameters[0] + parameters[1] * parameters[1];

  Intrepid::Vector<D2FadType, 3> 
  normal(2.0 * parameters[0], 2.0 * parameters[1], r2 - 1.0);
  normal /= (r2 + 1.0);
      
  return normal;
}
  
//----------------------------------------------------------------------------
Intrepid::Vector<D2FadType, 3> 
projective_get_normal(Intrepid::Vector<D2FadType, 3> & parameters)
{
  Intrepid::Vector<D2FadType, 3>
  normal(parameters[0], parameters[1], parameters[2]);

  D2FadType const
  n = Intrepid::norm(normal);
     
  if ( (n.val()).val()!=0 ) {
    normal /= n;
  }
  else {
    Intrepid::Vector<DFadType, 3> Xfad;
    Intrepid::Vector<D2FadType, 3> Xfad2;
    for ( int i = 0; i < 3; ++i ) {
      Xfad[i] = DFadType(3, i, 1.0/sqrt(3.0));
      Xfad2[i] = D2FadType(3, i, Xfad[i]);
      parameters[i] = Xfad2[i];
    }
    normal[0] = Xfad2[0];
    normal[1] = Xfad2[1];
    normal[2] = Xfad2[2];
  }
           
  return normal;
}
  
//----------------------------------------------------------------------------
Intrepid::Vector<D2FadType, 3> 
tangent_get_normal(Intrepid::Vector<D2FadType, 2> & parameters)
{
  D2FadType const
  r = sqrt(parameters[0] * parameters[0] + parameters[1] * parameters[1]);

  Intrepid::Vector<D2FadType, 3>
  normal(3, Intrepid::ZEROS);

   if ( (r.val()).val() > 0.0 ) {
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
Intrepid::Vector<D2FadType, 3> 
cartesian_get_normal1(Intrepid::Vector<D2FadType, 2> & parameters)
{
  Intrepid::Vector<D2FadType, 3> 
  normal(1, parameters[0], parameters[1]);
            
  return normal;
}

Intrepid::Vector<D2FadType, 3> 
cartesian_get_normal2(Intrepid::Vector<D2FadType, 2> & parameters)
{
  Intrepid::Vector<D2FadType, 3> 
  normal(parameters[0], 1, parameters[1]);
            
  return normal;
}

Intrepid::Vector<D2FadType, 3> 
cartesian_get_normal3(Intrepid::Vector<D2FadType, 2> & parameters)
{
  Intrepid::Vector<D2FadType, 3> 
  normal(parameters[0], parameters[1], 1);
            
  return normal;
}

//----------------------------------------------------------------------------//
void
spherical_newton_raphson(Intrepid::Tensor4<ScalarT, 3> & tangent,
  Intrepid::Vector<ScalarT, 2> & parameters,
  Intrepid::Vector<ScalarT> & direction, ScalarT & min_detA)
{    
  Intrepid::Vector<ScalarT, 2> Xval;
  Intrepid::Vector<DFadType, 2> Xfad;
  Intrepid::Vector<D2FadType, 2> Xfad2;
  Intrepid::Vector<DFadType, 2> Rfad;
  Intrepid::Vector<D2FadType, 3> n;

  D2FadType detA;
    
  // in std:vector form to work with nonlinear solve
  // size of Jacobian, 4 = 2 * 2
  std::vector<ScalarT> dRdX(4);
  std::vector<ScalarT> X(2);    
  std::vector<ScalarT> R(2);    

  for (int i(0); i < 2; ++i)
      X[i] = parameters[i];
           
  // local nonlinear solver for Newton iterative solve
  LCM::LocalNonlinearSolver<Residual, Traits> solver;
               
  ScalarT normR(0.0), normR0(0.0), relativeR(0.0);    
  bool converged = false;
  int iter = 0;
       
  while ( !converged ) {
    std::cout << "iter: " << iter << std::endl;
      
    for ( int i = 0; i < 2; ++i ) {
      Xval[i] = Sacado::ScalarValue<ScalarT>::eval(X[i]);
      Xfad[i] = DFadType(2, i, Xval[i]);
      Xfad2[i] = D2FadType(2, i, Xfad[i]);
    }
      
    n = spherical_get_normal(Xfad2);     

    detA = Intrepid::det(Intrepid::dot(n,Intrepid::dot(tangent, n)));
     
    std::cout << "parameters: " << Xval << std::endl;
    std::cout << "determinant: " << (detA.val()).val() << std::endl;
            
    for (int i = 0; i < 2; ++i){
      Rfad[i] = detA.dx(i);
      R[i] = Rfad[i].val();
    }
      
    //std::cout << "R: " << R << std::endl;
      
    normR = sqrt( R[0]*R[0] + R[1]*R[1] );
      
    if ( iter == 0 ) 
      normR0 = normR;
      
    std::cout << "normR: " << normR << std::endl;
      
    if (normR0 != 0)
      relativeR = normR / normR0;
    else
      relativeR = normR0;
      
    if (relativeR < 1.0e-8 || normR < 1.0e-8)
      break;
      
    if (iter > 50){
      std::cout << "Newton's loop for bifurcation check not converging after "
        << 50 << " iterations" << std::endl;      
      break;
    }
        
    // compute Jacobian
    for ( int i = 0; i < 2; ++i )
      for ( int j = 0; j < 2; ++j )
        dRdX[i + 2 * j] = Rfad[i].dx(j);
          
    // call local nonlinear solver
    solver.solve(dRdX, X, R);
            
    iter++;
      
  } // Newton Raphson iteration end

  // compute sensitivity information w.r.t. system parameters
  // and pack the sensitivity back to X
  // solver.computeFadInfo(dRdX, X, R);
    
  //update
  if ( (detA.val()).val() <= min_detA ) {
      
    for (int i(0); i < 3; ++i)
      direction[i] = (n[i].val()).val();
     
    for (int i(0); i < 2; ++i)
      parameters[i] = X[i];

    min_detA = (detA.val()).val();
      
  } else {
    std::cout << "Newnton's loop for bifurcation check fails to identify minimum det(A)" 
      << std::endl;
  }    
        
} // Function end
  
//----------------------------------------------------------------------------
void
stereographic_newton_raphson(Intrepid::Tensor4<ScalarT, 3> & tangent,
  Intrepid::Vector<ScalarT, 2> & parameters,
  Intrepid::Vector<ScalarT> & direction, ScalarT & min_detA)
{    
  Intrepid::Vector<ScalarT, 2> Xval;
  Intrepid::Vector<DFadType, 2> Xfad;
  Intrepid::Vector<D2FadType, 2> Xfad2;
  Intrepid::Vector<DFadType, 2> Rfad;
  Intrepid::Vector<D2FadType, 3> n;

  D2FadType detA;
    
  // in std:vector form to work with nonlinear solve
  // size of Jacobian, 4 = 2 * 2
  std::vector<ScalarT> dRdX(4);
  std::vector<ScalarT> X(2);    
  std::vector<ScalarT> R(2);    

  for (int i(0); i < 2; ++i)
      X[i] = parameters[i];
           
  // local nonlinear solver for Newton iterative solve
  LCM::LocalNonlinearSolver<Residual, Traits> solver;
               
  ScalarT normR(0.0), normR0(0.0), relativeR(0.0);    
  bool converged = false;
  int iter = 0;
      
  while ( !converged ) {
    std::cout << "iter: " << iter << std::endl;
      
    for ( int i = 0; i < 2; ++i ) {
      Xval[i] = Sacado::ScalarValue<ScalarT>::eval(X[i]);
      Xfad[i] = DFadType(2, i, Xval[i]);
      Xfad2[i] = D2FadType(2, i, Xfad[i]);
    }
      
    n = stereographic_get_normal(Xfad2);     

    detA = Intrepid::det(Intrepid::dot(n,Intrepid::dot(tangent, n)));
     
    std::cout << "parameters: " << Xval << std::endl;
    std::cout << "determinant: " << (detA.val()).val() << std::endl;
           
    for (int i = 0; i < 2; ++i){
      Rfad[i] = detA.dx(i);
      R[i] = Rfad[i].val();
    }
      
    //std::cout << "R: " << R << std::endl;
     
    normR = sqrt( R[0]*R[0] + R[1]*R[1] );
      
    if ( iter == 0 ) 
      normR0 = normR;
      
    std::cout << "normR: " << normR << std::endl;
      
    if (normR0 != 0)
      relativeR = normR / normR0;
    else
      relativeR = normR0;
      
    if (relativeR < 1.0e-8 || normR < 1.0e-8)
      break;
      
    if (iter > 50){
      std::cout << "Newton's loop for bifurcation check not converging after "
        << 50 << " iterations" << std::endl;      
      break;
    }
        
    // compute Jacobian
    for ( int i = 0; i < 2; ++i )
      for ( int j = 0; j < 2; ++j )
        dRdX[i + 2 * j] = Rfad[i].dx(j);
          
    // call local nonlinear solver
    solver.solve(dRdX, X, R);
           
    iter++;
      
  } // Newton Raphson iteration end

  // compute sensitivity information w.r.t. system parameters
  // and pack the sensitivity back to X
  // solver.computeFadInfo(dRdX, X, R);
    
  //update
  if ( (detA.val()).val() <= min_detA ) {
      
    for (int i(0); i < 3; ++i)
      direction[i] = (n[i].val()).val();
      
    for (int i(0); i < 2; ++i)
      parameters[i] = X[i];

    min_detA = (detA.val()).val();
      
  } else {
    std::cout << "Newnton's loop for bifurcation check fails to identify minimum det(A)" 
      << std::endl;
  }    
        
} // Function end
  
//----------------------------------------------------------------------------
void
projective_newton_raphson(Intrepid::Tensor4<ScalarT, 3> & tangent,
  Intrepid::Vector<ScalarT, 3> & parameters,
  Intrepid::Vector<ScalarT> & direction, ScalarT & min_detA)
{ 
  Intrepid::Vector<ScalarT, 4> parameters_new;
  ScalarT nNorm = Intrepid::norm(parameters);
  for ( int i = 0; i < 3; ++i ) {
    parameters_new[i] = parameters[i];
    if ( nNorm==0 ) parameters_new[i] = 1.0;
  }
  parameters_new[3] = 0;
      
  Intrepid::Vector<ScalarT, 4> Xval;
  Intrepid::Vector<DFadType, 4> Xfad;
  Intrepid::Vector<D2FadType, 4> Xfad2;
  Intrepid::Vector<DFadType, 4> Rfad;
  Intrepid::Vector<D2FadType, 3> n;

  D2FadType detA;
    
  // in std:vector form to work with nonlinear solve
  // size of Jacobian, 4 = 2 * 2
  std::vector<ScalarT> dRdX(16);
  std::vector<ScalarT> X(4);    
  std::vector<ScalarT> R(4);    

  for (int i(0); i < 4; ++i)
      X[i] = parameters_new[i];
            
  // local nonlinear solver for Newton iterative solve
  LCM::LocalNonlinearSolver<Residual, Traits> solver;
               
  ScalarT normR(0.0), normR0(0.0), relativeR(0.0);    
  bool converged = false;
  int iter = 0;
       
  while ( !converged ) {
    std::cout << "iter: " << iter << std::endl;
      
    for ( int i = 0; i < 4; ++i ) {
      Xval[i] = Sacado::ScalarValue<ScalarT>::eval(X[i]);
      Xfad[i] = DFadType(4, i, Xval[i]);
      Xfad2[i] = D2FadType(4, i, Xfad[i]);
    }
      
    Intrepid::Vector<D2FadType, 3> Xfad2_sub;
    for ( int i = 0; i < 3; ++i ) {
      Xfad2_sub[i] = Xfad2[i];
    }
    n = projective_get_normal(Xfad2_sub);    

    detA = Intrepid::det(Intrepid::dot(n,Intrepid::dot(tangent, n))) 
      + Xfad2[3] 
      * (Xfad2[0] * Xfad2[0] + Xfad2[1] * Xfad2[1] + Xfad2[2] * Xfad2[2] - 1);
     
    std::cout << "parameters: " << Xval << std::endl;
    std::cout << "determinant: " << (detA.val()).val() << std::endl;
            
    for (int i = 0; i < 4; ++i){
      Rfad[i] = detA.dx(i);
      R[i] = Rfad[i].val();
    }
      
    //std::cout << "R: " << R << std::endl;
     
    normR = sqrt( R[0]*R[0] + R[1]*R[1] + R[2]*R[2] + R[3]*R[3] );
      
    if ( iter == 0 ) 
      normR0 = normR;
      
    std::cout << "normR: " << normR << std::endl;
     
    if (normR0 != 0)
      relativeR = normR / normR0;
    else
      relativeR = normR0;
      
    if (relativeR < 1.0e-8 || normR < 1.0e-8)
      break;
      
    if (iter > 50){
      std::cout << "Newton's loop for bifurcation check not converging after "
        << 50 << " iterations" << std::endl;      
      break;
    }
        
    // compute Jacobian
    for ( int i = 0; i < 4; ++i )
      for ( int j = 0; j < 4; ++j )
        dRdX[i + 4 * j] = Rfad[i].dx(j);
          
    // call local nonlinear solver
    solver.solve(dRdX, X, R);
           
    iter++;
      
  } // Newton Raphson iteration end

  // compute sensitivity information w.r.t. system parameters
  // and pack the sensitivity back to X
  // solver.computeFadInfo(dRdX, X, R);
    
  //update
  if ( (detA.val()).val() <= min_detA ) {
      
    for (int i(0); i < 3; ++i)
      direction[i] = (n[i].val()).val();
      
    for (int i(0); i < 3; ++i)
      parameters[i] = X[i];

    min_detA = (detA.val()).val();
      
  } else {
    std::cout << "Newnton's loop for bifurcation check fails to identify minimum det(A)" 
      << std::endl;
  }    
        
} // Function end
  
//----------------------------------------------------------------------------
void
tangent_newton_raphson(Intrepid::Tensor4<ScalarT, 3> & tangent,
  Intrepid::Vector<ScalarT, 2> & parameters,
  Intrepid::Vector<ScalarT> & direction, ScalarT & min_detA)
{    
  Intrepid::Vector<ScalarT, 2> Xval;
  Intrepid::Vector<DFadType, 2> Xfad;
  Intrepid::Vector<D2FadType, 2> Xfad2;
  Intrepid::Vector<DFadType, 2> Rfad;
  Intrepid::Vector<D2FadType, 3> n;

  D2FadType detA;
    
  // in std:vector form to work with nonlinear solve
  // size of Jacobian, 4 = 2 * 2
  std::vector<ScalarT> dRdX(4);
  std::vector<ScalarT> X(2);    
  std::vector<ScalarT> R(2);    

  for (int i(0); i < 2; ++i)
      X[i] = parameters[i];
            
  // local nonlinear solver for Newton iterative solve
  LCM::LocalNonlinearSolver<Residual, Traits> solver;
               
  ScalarT normR(0.0), normR0(0.0), relativeR(0.0);    
  bool converged = false;
  int iter = 0;
       
  while ( !converged ) {
    std::cout << "iter: " << iter << std::endl;
      
    for ( int i = 0; i < 2; ++i ) {
      Xval[i] = Sacado::ScalarValue<ScalarT>::eval(X[i]);
      Xfad[i] = DFadType(2, i, Xval[i]);
      Xfad2[i] = D2FadType(2, i, Xfad[i]);
    }
      
    n = tangent_get_normal(Xfad2);     

    detA = Intrepid::det(Intrepid::dot(n,Intrepid::dot(tangent, n)));
     
    std::cout << "parameters: " << Xval << std::endl;
    std::cout << "determinant: " << (detA.val()).val() << std::endl;
            
    for (int i = 0; i < 2; ++i){
      Rfad[i] = detA.dx(i);
      R[i] = Rfad[i].val();
    }
      
    //std::cout << "R: " << R << std::endl;
      
    normR = sqrt( R[0]*R[0] + R[1]*R[1] );
      
    if ( iter == 0 ) 
      normR0 = normR;
      
    std::cout << "normR: " << normR << std::endl;
      
    if (normR0 != 0)
      relativeR = normR / normR0;
    else
      relativeR = normR0;
      
    if (relativeR < 1.0e-8 || normR < 1.0e-8)
      break;
      
    if (iter > 50){
      std::cout << "Newton's loop for bifurcation check not converging after "
        << 50 << " iterations" << std::endl;      
      break;
    }
        
    // compute Jacobian
    for ( int i = 0; i < 2; ++i )
      for ( int j = 0; j < 2; ++j )
        dRdX[i + 2 * j] = Rfad[i].dx(j);
          
    // call local nonlinear solver
    solver.solve(dRdX, X, R);
            
    iter++;
      
  } // Newton Raphson iteration end

  // compute sensitivity information w.r.t. system parameters
  // and pack the sensitivity back to X
  // solver.computeFadInfo(dRdX, X, R);
    
  //update
  if ( (detA.val()).val() <= min_detA ) {
      
    for (int i(0); i < 3; ++i)
      direction[i] = (n[i].val()).val();
      
    for (int i(0); i < 2; ++i)
      parameters[i] = X[i];

    min_detA = (detA.val()).val();
      
  } else {
    std::cout << "Newnton's loop for bifurcation check fails to identify minimum det(A)" 
      << std::endl;
  }    
        
} // Function end

//----------------------------------------------------------------------------
void
cartesian_newton_raphson(Intrepid::Tensor4<ScalarT, 3> & tangent,
  Intrepid::Vector<ScalarT, 2> & parameters, int surface_index,
  Intrepid::Vector<ScalarT> & direction, ScalarT & min_detA)
{    
  Intrepid::Vector<ScalarT, 2> Xval;
  Intrepid::Vector<DFadType, 2> Xfad;
  Intrepid::Vector<D2FadType, 2> Xfad2;
  Intrepid::Vector<DFadType, 2> Rfad;
  Intrepid::Vector<D2FadType, 3> n;

  D2FadType detA;
    
  // in std:vector form to work with nonlinear solve
  // size of Jacobian, 4 = 2 * 2
  std::vector<ScalarT> dRdX(4);
  std::vector<ScalarT> X(2);    
  std::vector<ScalarT> R(2);    

  for (int i(0); i < 2; ++i)
      X[i] = parameters[i];
            
  // local nonlinear solver for Newton iterative solve
  LCM::LocalNonlinearSolver<Residual, Traits> solver;
               
  ScalarT normR(0.0), normR0(0.0), relativeR(0.0);    
  bool converged = false;
  int iter = 0;
       
  while ( !converged ) {
    std::cout << "iter: " << iter << std::endl;
      
    for ( int i = 0; i < 2; ++i ) {
      Xval[i] = Sacado::ScalarValue<ScalarT>::eval(X[i]);
      Xfad[i] = DFadType(2, i, Xval[i]);
      Xfad2[i] = D2FadType(2, i, Xfad[i]);
    }
      
    switch ( surface_index ) {
      case 1: 
        n = cartesian_get_normal1(Xfad2);
        break;
      case 2: 
        n = cartesian_get_normal2(Xfad2);
        break;
      case 3: 
        n = cartesian_get_normal3(Xfad2);
        break;
      default:
        n = cartesian_get_normal1(Xfad2);
        break;
    }    

    detA = Intrepid::det(Intrepid::dot(n,Intrepid::dot(tangent, n)));
     
    std::cout << "parameters: " << Xval << std::endl;
    std::cout << "determinant: " << (detA.val()).val() << std::endl;
            
    for (int i = 0; i < 2; ++i){
      Rfad[i] = detA.dx(i);
      R[i] = Rfad[i].val();
    }
      
    //std::cout << "R: " << R << std::endl;
     
    normR = sqrt( R[0]*R[0] + R[1]*R[1] );
     
    if ( iter == 0 ) 
      normR0 = normR;
      
    std::cout << "normR: " << normR << std::endl;
      
    if (normR0 != 0)
      relativeR = normR / normR0;
    else
      relativeR = normR0;
      
    if (relativeR < 1.0e-8 || normR < 1.0e-8)
      break;
      
    if (iter > 50){
      std::cout << "Newton's loop for bifurcation check not converging after "
        << 50 << " iterations" << std::endl;      
      break;
    }
        
    // compute Jacobian
    for ( int i = 0; i < 2; ++i )
      for ( int j = 0; j < 2; ++j )
        dRdX[i + 2 * j] = Rfad[i].dx(j);
          
    // call local nonlinear solver
    solver.solve(dRdX, X, R);
            
    iter++;
      
  } // Newton Raphson iteration end

  // compute sensitivity information w.r.t. system parameters
  // and pack the sensitivity back to X
  // solver.computeFadInfo(dRdX, X, R);
    
  //update
  if ( (detA.val()).val() <= min_detA ) {
      
    for (int i(0); i < 3; ++i)
      direction[i] = (n[i].val()).val();
      
    for (int i(0); i < 2; ++i)
      parameters[i] = X[i];

    min_detA = (detA.val()).val();
      
  } else {
    std::cout << "Newnton's loop for bifurcation check fails to identify minimum det(A)" 
      << std::endl;
  }    
        
} // Function end
          
//
// Simple tests for parametrizations of the bifurcation tensor.
//
int main(int ac, char* av[])
{
  Intrepid::Tensor4<ScalarT, 3> tangent;
  
  // To get tangent for testing:
  // Option 1: read tangent from file
  
  //std::ifstream file_in("Tangent.txt");      
  //for (int i(0); i < 3; ++i) {
    //for (int j(0); j < 3; ++j) {
      //for (int k(0); k < 3; ++k) {
        //for (int l(0); l < 3; ++l) {
          //file_in >> tangent(i, j, k, l);
        //}
      //}
    //}
  //}        
  //file_in.close();
  
  // Option 2: creat a tangent
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

  tangent = lambda * I3 + mu * (I1 + I2);
 
    
  // Define time variables
  //struct timeval start, end;
  
  // Generate random initial guess for Newton Raphson 
  // provided sweep is not used  
  std::random_device rd;
  std::mt19937 mt_eng(rd());
  std::uniform_real_distribution<ScalarT> real_dist(0, 2);
  
  Intrepid::Vector<ScalarT, 2> arg_minimum;
  arg_minimum(0) =  real_dist(mt_eng) - 1;
  arg_minimum(1) =  real_dist(mt_eng) - 1;
  
  Intrepid::Vector<ScalarT> direction(1.0, 0.0, 0.0);
  ScalarT min_detA(1.0);
  
  // Get start time 
  //gettimeofday( &start, NULL );
  
  // Newton-Raphson to find exact bifurcation direction
  spherical_newton_raphson(tangent, arg_minimum, direction, min_detA);
  //stereographic_newton_raphson(tangent, arg_minimum, direction, min_detA);
  //projective_newton_raphson(tangent, arg_minimum, direction, min_detA);
  //tangent_newton_raphson(tangent, arg_minimum, direction, min_detA);
  //cartesian_newton_raphson(tangent, arg_minimum, 1, direction, min_detA);
  
  // Get end time
  //gettimeofday( &end, NULL );
  
  // Get time interval in the unit of micro second
  //int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) 
    //+ end.tv_usec - start.tv_usec;
  
  // Print time cost onto screen        
  //std::cout << std::endl;
  //std::cout << "Time cost: " << timeuse << std::endl;
  //std::cout << std::endl;

  return 0;

}
