//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include <Intrepid_MiniTensor.h>

#include <typeinfo>

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  BifurcationCheck<EvalT, Traits>::
  BifurcationCheck(const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl) :
    tangent_(p.get<std::string>("Material Tangent Name"),dl->qp_tensor4),
    ellipticity_flag_(p.get<std::string>("Ellipticity Flag Name"),dl->qp_scalar),
    direction_(p.get<std::string>("Bifurcation Direction Name"),dl->qp_vector)
  {
    std::vector<PHX::DataLayout::size_type> dims;
    dl->qp_tensor->dimensions(dims);
    num_pts_  = dims[1];
    num_dims_ = dims[2];

    this->addDependentField(tangent_);

    this->addEvaluatedField(ellipticity_flag_);
    this->addEvaluatedField(direction_);

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
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void BifurcationCheck<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    Intrepid::Vector<ScalarT> direction(1.0, 0.0, 0.0);
    Intrepid::Tensor4<ScalarT, 3> tangent;
    bool ellipticity_flag(0);

    // Compute DefGrad tensor from displacement gradient
    for (int cell(0); cell < workset.numCells; ++cell) {
      for (int pt(0); pt < num_pts_; ++pt) {

        tangent.fill( tangent_,cell,pt,0,0,0,0);
        ellipticity_flag_(cell,pt) = 0;

        ellipticity_flag = false;
        //boost::tie(ellipticity_flag, direction) 
         // = Intrepid::check_strong_ellipticity(tangent);
         
        bool sphereical_flag = spherical_sweep(tangent);
        
        bool stereo_flag = stereographic_sweep(tangent);
        
        bool project_flag = projective_sweep(tangent);
         
        ellipticity_flag = sphereical_flag;

        ellipticity_flag_(cell,pt) = ellipticity_flag;

        for (int i(0); i < num_dims_; ++i) {
          direction_(cell,pt,i) = direction(i);
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  bool BifurcationCheck<EvalT, Traits>::
  spherical_sweep(Intrepid::Tensor4<ScalarT, 3> const & tangent)
  {
    ScalarT const
    pi = std::acos(-1.0);

    ScalarT const
    phi_min = 0.0;

    ScalarT const
    phi_max = pi;

    ScalarT const
    theta_min = 0.0;

    ScalarT const
    theta_max = pi;

    Intrepid::Index const
    phi_num_points = 256;

    Intrepid::Index const
    theta_num_points = 256;

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
    std::cout << "\n*** SPHERICAL PARAMETRIZATION ***\n";

    std::cout << std::scientific << std::setprecision(16);

    std::cout << "Minimum: " << sphere_param.get_minimum();
    std::cout << " at " << sphere_param.get_arg_minimum() << '\n';
    std::cout << "Normal: " << sphere_param.get_normal_minimum() << '\n';

    //std::cout << "Maximum: " << sphere_param.get_maximum();
    //std::cout << " at " << sphere_param.get_arg_maximum() << '\n';
    //std::cout << "Normal: " << sphere_param.get_normal_maximum() << '\n';
    
    bool sphere_flag = false;
    if (sphere_param.get_minimum() <= 0) sphere_flag = true;
    
    return sphere_flag;
  
  }
  
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  bool BifurcationCheck<EvalT, Traits>::
  stereographic_sweep(Intrepid::Tensor4<ScalarT, 3> const & tangent)
  {
    ScalarT const
    x_min = -1.0;

    ScalarT const
    x_max = 1.0;

    ScalarT const
    y_min = -1.0;

    ScalarT const
    y_max = 1.0;

    Intrepid::Index const
    x_num_points = 256;

    Intrepid::Index const
    y_num_points = 256;

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
    std::cout << "\n*** STEREOGRAPHIC PARAMETRIZATION ***\n";

    std::cout << std::scientific << std::setprecision(16);

    std::cout << "Minimum: " << stereographic_param.get_minimum();
    std::cout << " at " << stereographic_param.get_arg_minimum() << '\n';
    std::cout << "Normal: " << stereographic_param.get_normal_minimum() << '\n';

    //std::cout << "Maximum: " << stereographic_param.get_maximum();
    //std::cout << " at " << stereographic_param.get_arg_maximum() << '\n';
    //std::cout << "Normal: " << stereographic_param.get_normal_maximum() << '\n';
    
    bool stereographic_flag = false;
    if (stereographic_param.get_minimum() <= 0) stereographic_flag = true;
    
    return stereographic_flag;
  
  }
  
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  bool BifurcationCheck<EvalT, Traits>::
  projective_sweep(Intrepid::Tensor4<ScalarT, 3> const & tangent)
  {
    ScalarT const
    x_min = -1.0;

    ScalarT const
    x_max = 1.0;

    ScalarT const
    y_min = -1.0;

    ScalarT const
    y_max = 1.0;

    ScalarT const
    z_min = -1.0;

    ScalarT const
    z_max = 1.0;
    
    Intrepid::Index const
    x_num_points = 64;

    Intrepid::Index const
    y_num_points = 64;

    Intrepid::Index const
    z_num_points = 64;
    
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
    std::cout << "\n*** PROJECTIVE PARAMETRIZATION ***\n";

    std::cout << std::scientific << std::setprecision(16);

    std::cout << "Minimum: " << projective_param.get_minimum();
    std::cout << " at " << projective_param.get_arg_minimum() << '\n';
    std::cout << "Normal: " << projective_param.get_normal_minimum() << '\n';

    //std::cout << "Maximum: " << projective_param.get_maximum();
    //std::cout << " at " << projective_param.get_arg_maximum() << '\n';
    //std::cout << "Normal: " << pprojective_param.get_normal_maximum() << '\n';
    
    bool projective_flag = false;
    if (projective_param.get_minimum() <= 0) projective_flag = true;
    
    return projective_flag;  
  }    
  //----------------------------------------------------------------------------
}
