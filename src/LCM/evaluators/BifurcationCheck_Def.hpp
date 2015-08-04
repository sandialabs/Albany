//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include <Intrepid_MiniTensor.h>
#include "LocalNonlinearSolver.hpp"

#include <typeinfo>

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  BifurcationCheck<EvalT, Traits>::
  BifurcationCheck(const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl) :
    parametrization_type_(p.get<std::string>("Parametrization Type Name")),
    parametrization_interval_(p.get<double>("Parametrization Interval Name")),
    tangent_(p.get<std::string>("Material Tangent Name"),dl->qp_tensor4),
    ellipticity_flag_(p.get<std::string>("Ellipticity Flag Name"),dl->qp_scalar),
    direction_(p.get<std::string>("Bifurcation Direction Name"),dl->qp_vector),
    min_detA_(p.get<std::string>("Min detA Name"),dl->qp_scalar)
  {
    std::vector<PHX::DataLayout::size_type> dims;
    dl->qp_tensor->dimensions(dims);
    num_pts_  = dims[1];
    num_dims_ = dims[2];

    this->addDependentField(tangent_);
    this->addEvaluatedField(ellipticity_flag_);
    this->addEvaluatedField(direction_);
    this->addEvaluatedField(min_detA_);

    this->setName("BifurcationCheck"+PHX::typeAsString<EvalT>());

  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void BifurcationCheck<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(tangent_,fm);
    this->utils.setFieldData(ellipticity_flag_,fm);
    this->utils.setFieldData(direction_,fm);
    this->utils.setFieldData(min_detA_,fm);   
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void BifurcationCheck<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    Intrepid::Vector<ScalarT> direction(1.0, 0.0, 0.0);
    Intrepid::Tensor4<ScalarT, 3> tangent;
    bool ellipticity_flag(false);
    ScalarT min_detA(1.0);

    for (int cell(0); cell < workset.numCells; ++cell) {
      for (int pt(0); pt < num_pts_; ++pt) {

        tangent.fill( tangent_,cell,pt,0,0,0,0);
        ellipticity_flag_(cell,pt) = 0;

        //boost::tie(ellipticity_flag, direction) 
         // = Intrepid::check_strong_ellipticity(tangent);
        
        double interval = parametrization_interval_;

	    if (parametrization_type_ == "Spherical") {
	      
	      Intrepid::Vector<ScalarT, 2> arg_minimum;
		  
		  min_detA = spherical_sweep(tangent, arg_minimum, direction, interval);		  
		  spherical_newton_raphson(tangent, arg_minimum, direction, min_detA);
	    
	    } 
	    else if(parametrization_type_ == "Stereographic") {
	      
	      Intrepid::Vector<ScalarT, 2> arg_minimum;
		  
		  min_detA = stereographic_sweep(tangent, arg_minimum, direction, interval);		  
		  stereographic_newton_raphson(tangent, arg_minimum, direction, min_detA);
	    
	    } 
	    else if(parametrization_type_ == "Projective") {
	    
	      Intrepid::Vector<ScalarT, 3> arg_minimum;      
		
		  min_detA = projective_sweep(tangent, arg_minimum, direction, interval);
		  projective_newton_raphson(tangent, arg_minimum, direction, min_detA);
	    
	    } 
	    else if(parametrization_type_ == "Tangent") {
	    
	      Intrepid::Vector<ScalarT, 2> arg_minimum;
		
		  min_detA = tangent_sweep(tangent, arg_minimum, direction, interval);		
		  tangent_newton_raphson(tangent, arg_minimum, direction, min_detA);
	    
	    } 
	    else if(parametrization_type_ == "Cartesian") {
		
		  Intrepid::Vector<ScalarT, 2> arg_minimum1;
		  Intrepid::Vector<ScalarT, 2> arg_minimum2;
		  Intrepid::Vector<ScalarT, 2> arg_minimum3;
		  Intrepid::Vector<ScalarT> direction1(1.0, 0.0, 0.0);
		  Intrepid::Vector<ScalarT> direction2(0.0, 1.0, 0.0);
		  Intrepid::Vector<ScalarT> direction3(0.0, 0.0, 1.0);
	      
		  ScalarT min_detA1 = cartesian_sweep(tangent, 
		    arg_minimum1, 1, direction1, interval);
		  
		  ScalarT min_detA2 = cartesian_sweep(tangent, 
		    arg_minimum2, 2, direction2, interval);
		  
		  ScalarT min_detA3 = cartesian_sweep(tangent, 
		    arg_minimum3, 3, direction3, interval);
		  
		  if ( min_detA1 <= min_detA2 && min_detA1 <= min_detA3 ) {
		  
		    cartesian_newton_raphson(tangent, 
		      arg_minimum1, 1, direction1, min_detA1);
		  
		    min_detA = min_detA1;
		    direction = direction1;
		  
		  }
		  else if ( min_detA2 <= min_detA1 && min_detA2 <= min_detA3 ) {
		  
		    cartesian_newton_raphson(tangent, 
		      arg_minimum2, 2, direction2, min_detA2);
		  
		    min_detA = min_detA2;
		    direction = direction2;
		  
		  }
		  else if ( min_detA3 <= min_detA1 && min_detA3 <= min_detA2 ) {
		  
		  	cartesian_newton_raphson(tangent, 
		  	  arg_minimum3, 3, direction3, min_detA3);
		  
		    min_detA = min_detA3;		  
		    direction = direction3;
		  }
	    
	    } 
	    else {
	    
	      Intrepid::Vector<ScalarT, 2> arg_minimum;
		
		  min_detA = spherical_sweep(tangent, arg_minimum, direction, interval);		
		  spherical_newton_raphson(tangent, arg_minimum, direction, min_detA);
	    }
        
        ellipticity_flag = false;
        if(min_detA <= 0.0) ellipticity_flag = true;

        ellipticity_flag_(cell,pt) = ellipticity_flag;
        min_detA_(cell,pt) = min_detA;
        
        std::cout << "\n" << min_detA << " @ " << direction << std::endl;

        for (int i(0); i < num_dims_; ++i) {
          direction_(cell,pt,i) = direction(i);
        }
        
      }
    }
    
  }
  
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  typename EvalT::ScalarT BifurcationCheck<EvalT, Traits>::
  spherical_sweep(Intrepid::Tensor4<ScalarT, 3> const & tangent,
    Intrepid::Vector<ScalarT, 2> & arg_minimum, 
    Intrepid::Vector<ScalarT> & direction, double const & interval)
  { 
    Intrepid::Index const 
    p_number = floor(1.0/interval);
      
    ScalarT const
    domain_min = 0;
    
    ScalarT const
    domain_max = std::acos(-1.0);
    
    ScalarT const 
    p_mean = (domain_max + domain_min) /2.0;
    
    ScalarT const
    p_span =  domain_max - domain_min;
    
    ScalarT const
    p_min = p_mean - p_span / 2.0 * interval * p_number;
    //p_min = domain_min;
        
    ScalarT const
    p_max = p_mean + p_span / 2.0 * interval * p_number;
    //p_max = domain_min + p_span * interval * p_number;

    // Initialize parameters
    ScalarT const
    phi_min = p_min;

    ScalarT const
    phi_max = p_max;

    ScalarT const
    theta_min = p_min;

    ScalarT const
    theta_max = p_max;

    Intrepid::Index const
    phi_num_points = p_number * 2 + 1;
    //phi_num_points = p_number + 1;

    Intrepid::Index const
    theta_num_points = p_number * 2 + 1;
    //theta_num_points = p_number + 1;
    
    /*----------------------------- landscape data ---------------------------//
    std::ofstream fout("/home/zlai/Documents/BifurcationCheck/Tests/"
    "Anisotropic-Axial/Spherical-Sweep.txt");
    for (int i=0; i<=128; i++) {
      for (int j=0; j<=128; j++) {
        ScalarT phi = phi_min + i*(phi_max-phi_min)/128.0;
        ScalarT theta = theta_min + j*(theta_max-theta_min)/128.0;
        Intrepid::Vector<ScalarT, 3>
        normal(sin(phi) * sin(theta), cos(phi), sin(phi) * cos(theta));
      
        // Localization tensor
        Intrepid::Tensor<ScalarT, 3>
        Q = Intrepid::dot2(normal, Intrepid::dot(tangent, normal));
        ScalarT determinant = Intrepid::det(Q);
        fout.width(15);
        fout << phi;
        fout.width(15);
        fout << theta;
        fout.width(15);
        fout << determinant << std::endl;
      }
    }
    fout << std::endl;
    fout << std::flush;
    fout.close();
    //------------------------------------------------------------------------*/

    Intrepid::Vector<ScalarT, 2> const
    sphere_min(phi_min, theta_min);

    Intrepid::Vector<ScalarT, 2> const
    sphere_max(phi_max, theta_max);

    Intrepid::Vector<Intrepid::Index, 2> const
    sphere_num_points(phi_num_points, theta_num_points);

    // Build the parametric grid with the specified parameters.
    Intrepid::ParametricGrid<ScalarT, 2>
    sphere_grid(sphere_min, sphere_max, sphere_num_points);

    // Build a spherical parametrization for this elasticity.
    Intrepid::SphericalParametrization<ScalarT, 3>
    sphere_param(tangent);

    // Traverse the grid with the parametrization.
    sphere_grid.traverse(sphere_param);

    // Query the parametrization for the minimum and maximum found on the grid.
    //std::cout << "\n*** SPHERICAL PARAMETRIZATION ***\n";
    //std::cout << "Interval: " << parametrization_interval_ << std::endl;
    //std::cout << sphere_param.get_minimum() 
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
  template<typename EvalT, typename Traits>
  typename EvalT::ScalarT BifurcationCheck<EvalT, Traits>::
  stereographic_sweep(Intrepid::Tensor4<ScalarT, 3> const & tangent,
    Intrepid::Vector<ScalarT, 2> & arg_minimum,  
    Intrepid::Vector<ScalarT> & direction, double const & interval)
  {    
    Intrepid::Index const 
    p_number = floor(1.0/interval);
          
    ScalarT const
    domain_min = -1.0;
    
    ScalarT const
    domain_max = 1.0;
    
    ScalarT const 
    p_mean = (domain_max + domain_min) /2.0;
    
    ScalarT const
    p_span =  domain_max - domain_min;
    
    ScalarT const
    p_min = p_mean - p_span / 2.0 * interval * p_number;
    //p_min = domain_min;
        
    ScalarT const
    p_max = p_mean + p_span / 2.0 * interval * p_number;
    //p_max = domain_min + p_span * interval * p_number;
    
    // Initialize parametres
    ScalarT const
    x_min = p_min;

    ScalarT const
    x_max = p_max;

    ScalarT const
    y_min = p_min;

    ScalarT const
    y_max = p_max;

    Intrepid::Index const
    x_num_points = p_number * 2 + 1;
    //x_num_points = p_number + 1;

    Intrepid::Index const
    y_num_points = p_number * 2 + 1;
    //y_num_points = p_number + 1;

    /*----------------------------- landscape data ---------------------------//
    std::ofstream fout("/home/zlai/Documents/BifurcationCheck/Tests/"
    "Anisotropic-Axial/Stereographic-Sweep.txt");
    for (int i=0; i<=128; i++) {
      for (int j=0; j<=128; j++) {
        ScalarT x = x_min + i*(x_max-x_min)/128.0;
        ScalarT y = y_min + j*(y_max-y_min)/128.0;

        ScalarT r2 = x * x + y * y;

        Intrepid::Vector<ScalarT, 3> 
        normal(2.0 * x, 2.0 * y, r2 - 1.0);
        normal /= (r2 + 1.0);
      
        // Localization tensor
        Intrepid::Tensor<ScalarT, 3>
        Q = Intrepid::dot2(normal, Intrepid::dot(tangent, normal));
        ScalarT determinant = Intrepid::det(Q);
        fout.width(15);
        fout << x;
        fout.width(15);
        fout << y;
        fout.width(15);
        fout << determinant << std::endl;
      }
    }
    fout << std::endl;
    fout << std::flush;
    fout.close();
    //------------------------------------------------------------------------*/

    Intrepid::Vector<ScalarT, 2> const
    stereographic_min(x_min, y_min);

    Intrepid::Vector<ScalarT, 2> const
    stereographic_max(x_max, y_max);

    Intrepid::Vector<Intrepid::Index, 2> const
    stereographic_num_points(x_num_points, y_num_points);

    // Build the parametric grid with the specified parameters.
    Intrepid::ParametricGrid<ScalarT, 2>
    stereographic_grid
      (stereographic_min, stereographic_max, stereographic_num_points);

    // Build a stereographic parametrization for this elasticity.
    Intrepid::StereographicParametrization<ScalarT, 3>
    stereographic_param(tangent);

    // Traverse the grid with the parametrization.
    stereographic_grid.traverse(stereographic_param);

    // Query the parametrization for the minimum and maximum found on the grid.
    //std::cout << "\n*** STEREOGRAPHIC PARAMETRIZATION ***\n";
    //std::cout << "Interval: " << parametrization_interval_ << std::endl;
    //std::cout << stereographic_param.get_minimum()
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
  template<typename EvalT, typename Traits>
  typename EvalT::ScalarT BifurcationCheck<EvalT, Traits>::
  projective_sweep(Intrepid::Tensor4<ScalarT, 3> const & tangent,
    Intrepid::Vector<ScalarT, 3> & arg_minimum,  
    Intrepid::Vector<ScalarT> & direction, double const & interval)
  {   
    Intrepid::Index const 
    p_number = floor(1.0/interval);
          
    ScalarT const
    domain_min = -1.0;
    
    ScalarT const
    domain_max = 1.0;
    
    ScalarT const 
    p_mean = (domain_max + domain_min) /2.0;
    
    ScalarT const
    p_span =  domain_max - domain_min;
    
    ScalarT const
    p_min = p_mean - p_span / 2.0 * interval * p_number;
    //p_min = domain_min;
        
    ScalarT const
    p_max = p_mean + p_span / 2.0 * interval * p_number;
    //p_max = domain_min + p_span * interval * p_number;
    
    // Initialize parametres
    ScalarT const
    x_min = p_min;

    ScalarT const
    x_max = p_max;

    ScalarT const
    y_min = p_min;

    ScalarT const
    y_max = p_max;

    ScalarT const
    z_min = p_min;

    ScalarT const
    z_max = p_max;
    
    Intrepid::Index const
    x_num_points = p_number * 2 + 1;
    //x_num_points = p_number + 1;

    Intrepid::Index const
    y_num_points = p_number * 2 + 1;
    //y_num_points = p_number + 1;

    Intrepid::Index const
    z_num_points = p_number * 2 + 1;
    //z_num_points = p_number + 1;
    
    /*----------------------------- landscape data ---------------------------//
    std::ofstream fout("/home/zlai/Documents/BifurcationCheck/Tests/"
    "Anisotropic-Axial/Projective-Sweep.txt");
    for (int i=0; i<=x_num_points; i++) {
      for (int j=0; j<=y_num_points; j++) {
        for (int k=0; k<=z_num_points; k++) {
        ScalarT x = x_min + i*(x_max-x_min)/(x_num_points-1);
        ScalarT y = y_min + j*(y_max-y_min)/(y_num_points-1);
        ScalarT z = z_min + k*(z_max-z_min)/(z_num_points-1);

        Intrepid::Vector<ScalarT, 3>
        normal(x, y, z);

        ScalarT const
        n = Intrepid::norm(normal);

        if (n > 0.0) {
          normal /= n;
        } else {
          normal = Intrepid::Vector<ScalarT, 3>(1.0, 1.0, 1.0);
        }
      
        // Localization tensor
        Intrepid::Tensor<ScalarT, 3>
        Q = Intrepid::dot2(normal, Intrepid::dot(tangent, normal));
        ScalarT determinant = Intrepid::det(Q);
        fout.width(15);
        fout << x;
        fout.width(15);
        fout << y;
        fout.width(15);
        fout << z;
        fout.width(15);
        fout << determinant << std::endl;
        }
      }
    }
    fout << std::endl;
    fout << std::flush;
    fout.close();
    //------------------------------------------------------------------------*/    
    
    Intrepid::Vector<ScalarT, 3> const
    projective_min(x_min, y_min, z_min);

    Intrepid::Vector<ScalarT, 3> const
    projective_max(x_max, y_max, z_max);

    Intrepid::Vector<Intrepid::Index, 3> const
    projective_num_points(x_num_points, y_num_points, z_num_points);

    // Build the parametric grid with the specified parameters.
    Intrepid::ParametricGrid<ScalarT, 3>
    projective_grid(projective_min, projective_max, projective_num_points);

    // Build a projective parametrization for this elasticity.
    Intrepid::ProjectiveParametrization<ScalarT, 3>
    projective_param(tangent);

    // Traverse the grid with the parametrization.
    projective_grid.traverse(projective_param);

    // Query the parametrization for the minimum and maximum found on the grid.
    //std::cout << "\n*** PROJECTIVE PARAMETRIZATION ***\n";
    //std::cout << "Interval: " << parametrization_interval_ << std::endl;
    //std::cout << projective_param.get_minimum() 
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
  template<typename EvalT, typename Traits>
  typename EvalT::ScalarT BifurcationCheck<EvalT, Traits>::
  tangent_sweep(Intrepid::Tensor4<ScalarT, 3> const & tangent,
    Intrepid::Vector<ScalarT, 2> & arg_minimum,  
    Intrepid::Vector<ScalarT> & direction, double const & interval)
  {   
    Intrepid::Index const 
    p_number = floor(1.0/interval);
          
    ScalarT const
    domain_min = -std::acos(-1.0) / 2.0;
    
    ScalarT const
    domain_max = std::acos(-1.0) / 2.0;
    
    ScalarT const 
    p_mean = (domain_max + domain_min) /2.0;
    
    ScalarT const
    p_span =  domain_max - domain_min;
    
    ScalarT const
    p_min = p_mean - p_span / 2.0 * interval * p_number;
    //p_min = domain_min;
        
    ScalarT const
    p_max = p_mean + p_span / 2.0 * interval * p_number;
    //p_max = domain_min + p_span * interval * p_number;
    
    // Initialize parametres
    ScalarT const
    x_min = p_min;

    ScalarT const
    x_max = p_max;

    ScalarT const
    y_min = p_min;

    ScalarT const
    y_max = p_max;

    Intrepid::Index const
    x_num_points = p_number * 2 + 1;
    //x_num_points = p_number + 1;

    Intrepid::Index const
    y_num_points = p_number * 2 + 1;
    //y_num_points = p_number + 1;
    
    /*----------------------------- landscape data ---------------------------//
    std::ofstream fout("/home/zlai/Documents/BifurcationCheck/Tests/"
    "Anisotropic-Axial/Tangent-Sweep.txt");
    for (int i=0; i<=128; i++) {
      for (int j=0; j<=128; j++) {
        ScalarT x = x_min + i*(x_max-x_min)/128.0;
        ScalarT y = y_min + j*(y_max-y_min)/128.0;

        ScalarT const
        r = std::sqrt(x * x + y * y);

        Intrepid::Vector<ScalarT, 3>
        normal(3, Intrepid::ZEROS);

        if (r > 0.0) {
          normal(0) = x * std::sin(r) / r;
          normal(1) = y * std::sin(r) / r;
          normal(2) = std::cos(r);
        } else {
          normal(0) = 0.0;
          normal(1) = 0.0;
          normal(2) = 1.0;
        }

      
        // Localization tensor
        Intrepid::Tensor<ScalarT, 3>
        Q = Intrepid::dot2(normal, Intrepid::dot(tangent, normal));
        ScalarT determinant = Intrepid::det(Q);
        fout.width(15);
        fout << x;
        fout.width(15);
        fout << y;
        fout.width(15);
        fout << determinant << std::endl;
      }
    }
    fout << std::endl;
    fout << std::flush;
    fout.close();
    //------------------------------------------------------------------------*/    

    Intrepid::Vector<ScalarT, 2> const
    tangent_min(x_min, y_min);

    Intrepid::Vector<ScalarT, 2> const
    tangent_max(x_max, y_max);

    Intrepid::Vector<Intrepid::Index, 2> const
    tangent_num_points(x_num_points, y_num_points);

    // Build the parametric grid with the specified parameters.
    Intrepid::ParametricGrid<ScalarT, 2>
    tangent_grid(tangent_min, tangent_max, tangent_num_points);

    // Build a tangent parametrization for this elasticity.
    Intrepid::TangentParametrization<ScalarT, 3>
    tangent_param(tangent);

    // Traverse the grid with the parametrization.
    tangent_grid.traverse(tangent_param);

    // Query the parametrization for the minimum and maximum found on the grid.
    //std::cout << "\n*** TANGENT PARAMETRIZATION ***\n";
    //std::cout << "Interval: " << parametrization_interval_ << std::endl;
    //std::cout << tangent_param.get_minimum()
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
  template<typename EvalT, typename Traits>
  typename EvalT::ScalarT BifurcationCheck<EvalT, Traits>::
  cartesian_sweep(Intrepid::Tensor4<ScalarT, 3> const & tangent, 
    Intrepid::Vector<ScalarT, 2> & arg_minimum, int surface_index,  
    Intrepid::Vector<ScalarT> & direction, double const & interval)
  {    
    Intrepid::Index const 
    p_number = floor(1.0/interval);
          
    ScalarT const
    domain_min = -1.0;
    
    ScalarT const
    domain_max = 1.0;
    
    ScalarT const 
    p_mean = (domain_max + domain_min) /2.0;
    
    ScalarT const
    p_span =  domain_max - domain_min;
    
    ScalarT const
    p_min = p_mean - p_span / 2.0 * interval * p_number;
    //p_min = domain_min;
        
    ScalarT const
    p_max = p_mean + p_span / 2.0 * interval * p_number;
    //p_max = domain_min + p_span * interval * p_number;
    
    // Initialize parametres
    ScalarT const
    p_surface = 1.0;

    Intrepid::Index const
    p_num_points = p_number * 2 + 1;
    //p_num_points = p_number + 1;

    Intrepid::Index const
    p_surface_num_points = 1;

    /*----------------------------- landscape data ---------------------------//
    std::ofstream fout("/home/zlai/Documents/BifurcationCheck/Tests/"
    "Anisotropic-Axial/Cartesian-Sweep.txt");
    for (int i=0; i<=128; i++) {
      for (int j=0; j<=128; j++) {
        ScalarT x = p_min + i*(p_max-p_min)/128;
        ScalarT y = p_min + j*(p_max-p_min)/128;
        ScalarT z = 1.0;

        Intrepid::Vector<ScalarT, 3>
        normal(z, x, y);
      
        // Localization tensor
        Intrepid::Tensor<ScalarT, 3>
        Q = Intrepid::dot2(normal, Intrepid::dot(tangent, normal));
        ScalarT determinant = Intrepid::det(Q);
        fout.width(15);
        fout << x;
        fout.width(15);
        fout << y;
        fout.width(15);
        fout << determinant << std::endl;
      }
    }
    fout << std::endl;
    fout << std::flush;
    fout.close();
    //------------------------------------------------------------------------*/
    
    ScalarT min_detA(1.0);
    
    if (surface_index == 1) {
      // x surface
      Intrepid::Vector<ScalarT, 3> const
      cartesian1_min(p_surface, p_min, p_min);

      Intrepid::Vector<ScalarT, 3> const
      cartesian1_max(p_surface, p_max, p_max);

      Intrepid::Vector<Intrepid::Index, 3> const
      cartesian1_num_points(p_surface_num_points, p_num_points, p_num_points);

      // Build the parametric grid with the specified parameters.
      Intrepid::ParametricGrid<ScalarT, 3>
      cartesian1_grid(cartesian1_min, cartesian1_max, cartesian1_num_points);

      // Build a cartesian parametrization for this elasticity.
      Intrepid::CartesianParametrization<ScalarT, 3>
      cartesian1_param(tangent);

      // Traverse the grid with the parametrization.
      cartesian1_grid.traverse(cartesian1_param);
    
      // Query the parametrization for the minimum and maximum found on the grid.
      //std::cout << "\n*** CARTESIAN PARAMETRIZATION ***\n";
      //std::cout << "Interval: " << parametrization_interval_ << std::endl;
      //std::cout << cartesian1_param.get_minimum() 
       // << "  " << cartesian1_param.get_normal_minimum() << std::endl;
          
      min_detA = cartesian1_param.get_minimum();
      arg_minimum(0) = (cartesian1_param.get_arg_minimum())(1);
      arg_minimum(1) = (cartesian1_param.get_arg_minimum())(2);
      for (int i(0); i < 3; ++i) {
        direction(i) = (cartesian1_param.get_normal_minimum())(i);
      }
    }

    if (surface_index == 2) {
      // y surface
      Intrepid::Vector<ScalarT, 3> const
      cartesian2_min(p_min, p_surface, p_min);

      Intrepid::Vector<ScalarT, 3> const
      cartesian2_max(p_max, p_surface, p_max);

      Intrepid::Vector<Intrepid::Index, 3> const
      cartesian2_num_points(p_num_points, p_surface_num_points, p_num_points);

      // Build the parametric grid with the specified parameters.
      Intrepid::ParametricGrid<ScalarT, 3>
      cartesian2_grid(cartesian2_min, cartesian2_max, cartesian2_num_points);

      // Build a cartesian parametrization for this elasticity.
      Intrepid::CartesianParametrization<ScalarT, 3>
      cartesian2_param(tangent);

      // Traverse the grid with the parametrization.
      cartesian2_grid.traverse(cartesian2_param);

      // Query the parametrization for the minimum and maximum found on the grid.
      //std::cout << "\n*** CARTESIAN PARAMETRIZATION ***\n";
      //std::cout << "Interval: " << parametrization_interval_ << std::endl;
      //std::cout << cartesian2_param.get_minimum() 
       // << "  " << cartesian2_param.get_normal_minimum() << std::endl;
          
      min_detA = cartesian2_param.get_minimum();
      arg_minimum(0) = (cartesian2_param.get_arg_minimum())(0);
      arg_minimum(1) = (cartesian2_param.get_arg_minimum())(2);
      for (int i(0); i < 3; ++i) {
        direction(i) = (cartesian2_param.get_normal_minimum())(i);
      }
    }

    if (surface_index == 3) {
      // z surface
      Intrepid::Vector<ScalarT, 3> const
      cartesian3_min(p_min, p_min, p_surface);

      Intrepid::Vector<ScalarT, 3> const
      cartesian3_max(p_max, p_max, p_surface);

      Intrepid::Vector<Intrepid::Index, 3> const
      cartesian3_num_points(p_num_points, p_num_points, p_surface_num_points);

      // Build the parametric grid with the specified parameters.
      Intrepid::ParametricGrid<ScalarT, 3>
      cartesian3_grid(cartesian3_min, cartesian3_max, cartesian3_num_points);

      // Build a cartesian parametrization for this elasticity.
      Intrepid::CartesianParametrization<ScalarT, 3>
      cartesian3_param(tangent);

      // Traverse the grid with the parametrization.
      cartesian3_grid.traverse(cartesian3_param);

      // Query the parametrization for the minimum and maximum found on the grid.
      //std::cout << "\n*** CARTESIAN PARAMETRIZATION ***\n";
      //std::cout << "Interval: " << parametrization_interval_ << std::endl;
      //std::cout << cartesian3_param.get_minimum() 
        //<< "  " << cartesian3_param.get_normal_minimum() << std::endl;
          
      min_detA = cartesian3_param.get_minimum();
      arg_minimum(0) = (cartesian3_param.get_arg_minimum())(0);
      arg_minimum(1) = (cartesian3_param.get_arg_minimum())(1);
      for (int i(0); i < 3; ++i) {
        direction(i) = (cartesian3_param.get_normal_minimum())(i);
      }
    }
    
    return min_detA;      
  }
  
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void BifurcationCheck<EvalT, Traits>::
  spherical_newton_raphson(Intrepid::Tensor4<ScalarT, 3> const & tangent,
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
    LocalNonlinearSolver<EvalT, Traits> solver;
               
    ScalarT normR(0.0), normR0(0.0), relativeR(0.0);    
    bool converged = false;
    int iter = 0;
       
    while ( !converged ) {
      //std::cout << "iter: " << iter << std::endl;
      
      for ( int i = 0; i < 2; ++i ) {
        Xval[i] = Sacado::ScalarValue<ScalarT>::eval(X[i]);
        Xfad[i] = DFadType(2, i, Xval[i]);
        Xfad2[i] = D2FadType(2, i, Xfad[i]);
      }
      
      n = spherical_get_normal(Xfad2);     

      detA = Intrepid::det(Intrepid::dot2(n,Intrepid::dot(tangent, n)));
     
      //std::cout << "parameters: " << parameters << std::endl;
      //std::cout << "determinant: " << (detA.val()).val() << std::endl;
            
      for (int i = 0; i < 2; ++i){
        Rfad[i] = detA.dx(i);
        R[i] = Rfad[i].val();
      }
      
      //std::cout << "R: " << R << std::endl;
      
      normR = sqrt( R[0]*R[0] + R[1]*R[1] );
      
      if ( iter == 0 ) 
        normR0 = normR;
      
      //std::cout << "normR: " << normR << std::endl;
      
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
  template<typename EvalT, typename Traits>
  void BifurcationCheck<EvalT, Traits>::
  stereographic_newton_raphson(Intrepid::Tensor4<ScalarT, 3> const & tangent,
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
    LocalNonlinearSolver<EvalT, Traits> solver;
               
    ScalarT normR(0.0), normR0(0.0), relativeR(0.0);    
    bool converged = false;
    int iter = 0;
       
    while ( !converged ) {
      //std::cout << "iter: " << iter << std::endl;
      
      for ( int i = 0; i < 2; ++i ) {
        Xval[i] = Sacado::ScalarValue<ScalarT>::eval(X[i]);
        Xfad[i] = DFadType(2, i, Xval[i]);
        Xfad2[i] = D2FadType(2, i, Xfad[i]);
      }
      
      n = stereographic_get_normal(Xfad2);     

      detA = Intrepid::det(Intrepid::dot2(n,Intrepid::dot(tangent, n)));
     
      //std::cout << "parameters: " << parameters << std::endl;
      //std::cout << "determinant: " << (detA.val()).val() << std::endl;
            
      for (int i = 0; i < 2; ++i){
        Rfad[i] = detA.dx(i);
        R[i] = Rfad[i].val();
      }
      
      //std::cout << "R: " << R << std::endl;
      
      normR = sqrt( R[0]*R[0] + R[1]*R[1] );
      
      if ( iter == 0 ) 
        normR0 = normR;
      
      //std::cout << "normR: " << normR << std::endl;
      
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
  template<typename EvalT, typename Traits>
  void BifurcationCheck<EvalT, Traits>::
  projective_newton_raphson(Intrepid::Tensor4<ScalarT, 3> const & tangent,
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
    LocalNonlinearSolver<EvalT, Traits> solver;
               
    ScalarT normR(0.0), normR0(0.0), relativeR(0.0);    
    bool converged = false;
    int iter = 0;
       
    while ( !converged ) {
      //std::cout << "iter: " << iter << std::endl;
      
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

      detA = Intrepid::det(Intrepid::dot2(n,Intrepid::dot(tangent, n))) 
        + Xfad2[3] 
        * (Xfad2[0] * Xfad2[0] + Xfad2[1] * Xfad2[1] + Xfad2[2] * Xfad2[2] - 1);
     
      //std::cout << "parameters: " << parameters << std::endl;
      //std::cout << "determinant: " << (detA.val()).val() << std::endl;
            
      for (int i = 0; i < 4; ++i){
        Rfad[i] = detA.dx(i);
        R[i] = Rfad[i].val();
      }
      
      //std::cout << "R: " << R << std::endl;
      
      normR = sqrt( R[0]*R[0] + R[1]*R[1] + R[2]*R[2] + R[3]*R[3] );
      
      if ( iter == 0 ) 
        normR0 = normR;
      
      //std::cout << "normR: " << normR << std::endl;
      
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
  template<typename EvalT, typename Traits>
  void BifurcationCheck<EvalT, Traits>::
  tangent_newton_raphson(Intrepid::Tensor4<ScalarT, 3> const & tangent,
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
    LocalNonlinearSolver<EvalT, Traits> solver;
               
    ScalarT normR(0.0), normR0(0.0), relativeR(0.0);    
    bool converged = false;
    int iter = 0;
       
    while ( !converged ) {
      //std::cout << "iter: " << iter << std::endl;
      
      for ( int i = 0; i < 2; ++i ) {
        Xval[i] = Sacado::ScalarValue<ScalarT>::eval(X[i]);
        Xfad[i] = DFadType(2, i, Xval[i]);
        Xfad2[i] = D2FadType(2, i, Xfad[i]);
      }
      
      n = tangent_get_normal(Xfad2);     

      detA = Intrepid::det(Intrepid::dot2(n,Intrepid::dot(tangent, n)));
     
      //std::cout << "parameters: " << parameters << std::endl;
      //std::cout << "determinant: " << (detA.val()).val() << std::endl;
            
      for (int i = 0; i < 2; ++i){
        Rfad[i] = detA.dx(i);
        R[i] = Rfad[i].val();
      }
      
      //std::cout << "R: " << R << std::endl;
      
      normR = sqrt( R[0]*R[0] + R[1]*R[1] );
      
      if ( iter == 0 ) 
        normR0 = normR;
      
      //std::cout << "normR: " << normR << std::endl;
      
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
  template<typename EvalT, typename Traits>
  void BifurcationCheck<EvalT, Traits>::
  cartesian_newton_raphson(Intrepid::Tensor4<ScalarT, 3> const & tangent,
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
    LocalNonlinearSolver<EvalT, Traits> solver;
               
    ScalarT normR(0.0), normR0(0.0), relativeR(0.0);    
    bool converged = false;
    int iter = 0;
       
    while ( !converged ) {
      //std::cout << "iter: " << iter << std::endl;
      
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

      detA = Intrepid::det(Intrepid::dot2(n,Intrepid::dot(tangent, n)));
     
      //std::cout << "parameters: " << parameters << std::endl;
      //std::cout << "determinant: " << (detA.val()).val() << std::endl;
            
      for (int i = 0; i < 2; ++i){
        Rfad[i] = detA.dx(i);
        R[i] = Rfad[i].val();
      }
      
      //std::cout << "R: " << R << std::endl;
      
      normR = sqrt( R[0]*R[0] + R[1]*R[1] );
      
      if ( iter == 0 ) 
        normR0 = normR;
      
      //std::cout << "normR: " << normR << std::endl;
      
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
  template<typename EvalT, typename Traits>
  Intrepid::Vector<typename BifurcationCheck<EvalT, Traits>::D2FadType, 3> 
  BifurcationCheck<EvalT, Traits>::
  spherical_get_normal(Intrepid::Vector<D2FadType, 2> & parameters)
  {
    Intrepid::Vector<D2FadType, 3> 
    normal(sin(parameters[0]) * sin(parameters[1]), 
    cos(parameters[0]), sin(parameters[0]) * cos(parameters[1]));
    
    return normal;
  }
  
  //----------------------------------------------------------------------------  
  template<typename EvalT, typename Traits>
  Intrepid::Vector<typename BifurcationCheck<EvalT, Traits>::D2FadType, 3> 
  BifurcationCheck<EvalT, Traits>::
  stereographic_get_normal(Intrepid::Vector<D2FadType, 2> & parameters)
  {
    D2FadType r2 = parameters[0] * parameters[0] + parameters[1] * parameters[1];

    Intrepid::Vector<D2FadType, 3> 
    normal(2.0 * parameters[0], 2.0 * parameters[1], r2 - 1.0);
    normal /= (r2 + 1.0);
      
    return normal;
  }
  
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>   
  Intrepid::Vector<typename BifurcationCheck<EvalT, Traits>::D2FadType, 3> 
  BifurcationCheck<EvalT, Traits>::
  projective_get_normal(Intrepid::Vector<D2FadType, 3> & parameters)
  {
    Intrepid::Vector<D2FadType, 3>
    normal(parameters[0], parameters[1], parameters[2]);

    D2FadType const
    n = Intrepid::norm(normal);
     
    if ( (n.val()).val()!=0 ) {
      //normal /= n;
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
  template<typename EvalT, typename Traits>   
  Intrepid::Vector<typename BifurcationCheck<EvalT, Traits>::D2FadType, 3> 
  BifurcationCheck<EvalT, Traits>::
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
  template<typename EvalT, typename Traits>  
  Intrepid::Vector<typename BifurcationCheck<EvalT, Traits>::D2FadType, 3> 
  BifurcationCheck<EvalT, Traits>::
  cartesian_get_normal1(Intrepid::Vector<D2FadType, 2> & parameters)
  {
    Intrepid::Vector<D2FadType, 3> 
    normal(1, parameters[0], parameters[1]);
            
    return normal;
  }

  template<typename EvalT, typename Traits>   
  Intrepid::Vector<typename BifurcationCheck<EvalT, Traits>::D2FadType, 3> 
  BifurcationCheck<EvalT, Traits>::
  cartesian_get_normal2(Intrepid::Vector<D2FadType, 2> & parameters)
  {
    Intrepid::Vector<D2FadType, 3> 
    normal(parameters[0], 1, parameters[1]);
            
    return normal;
  }

  template<typename EvalT, typename Traits>    
  Intrepid::Vector<typename BifurcationCheck<EvalT, Traits>::D2FadType, 3> 
  BifurcationCheck<EvalT, Traits>::
  cartesian_get_normal3(Intrepid::Vector<D2FadType, 2> & parameters)
  {
    Intrepid::Vector<D2FadType, 3> 
    normal(parameters[0], parameters[1], 1);
            
    return normal;
  }
}
