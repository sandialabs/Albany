//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Author: Mario J. Juha (juham@rpi.edu)

#include <cmath>
#include "Intrepid_MiniTensor.h"
#include "Intrepid_MiniTensor_Definitions.h"
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM
{

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  ElasticCrystalModel< EvalT, Traits > ::
  ElasticCrystalModel( Teuchos::ParameterList* p,
		       const Teuchos::RCP< Albany::Layouts > & dl ) :
    LCM::ConstitutiveModel< EvalT, Traits >(p, dl)
  {

    // Read elastic coefficients
    Teuchos::ParameterList e_list = p->sublist("Elastic Coefficients");
    // Assuming isotropy
    c11_ = e_list.get<RealType>("C11");
    c22_ = e_list.get<RealType>("C22");
    c33_ = e_list.get<RealType>("C33");
    c44_ = e_list.get<RealType>("C44");
    c55_ = e_list.get<RealType>("C55");
    c66_ = e_list.get<RealType>("C66");
    c12_ = e_list.get<RealType>("C12");
    c13_ = e_list.get<RealType>("C13");
    c23_ = e_list.get<RealType>("C23");
    c15_ = e_list.get<RealType>("C15");
    c25_ = e_list.get<RealType>("C25");
    c35_ = e_list.get<RealType>("C35");
    c46_ = e_list.get<RealType>("C46");

    // Read Bunge Angle in degrees
    e_list = p->sublist("Bunge Angles");
    RealType phi1d = e_list.get<RealType>("phi1");
    RealType Phid = e_list.get<RealType>("Phi");
    RealType phi2d = e_list.get<RealType>("phi2");

    
    // From degree to rad
    RealType degtorad = atan(1.0)/45.0;
    phi1_ = phi1d*degtorad;
    Phi_ = Phid*degtorad;
    phi2_ = phi2d*degtorad;

    const Intrepid::Index IndexM = 3;
    const Intrepid::Index IndexN = 3;

    // Initialize rotation matrix
    Intrepid::Matrix<double, IndexM, IndexN> rl;
    rl.fill(Intrepid::ZEROS);

    // Compute rotation matrix
    rl(0,0) = cos(phi1_)*cos(phi2_) - sin(phi1_)*cos(Phi_)*sin(phi2_);
    rl(1,0) = sin(phi1_)*cos(phi2_) + cos(phi1_)*cos(Phi_)*sin(phi2_);
    rl(2,0) = sin(Phi_)*sin(phi2_);
    rl(0,1) = -cos(phi1_)*sin(phi2_) - sin(phi1_)*cos(Phi_)*cos(phi2_);
    rl(1,1) = -sin(phi1_)*sin(phi2_) + cos(phi1_)*cos(Phi_)*cos(phi2_);
    rl(2,1) = sin(Phi_)*cos(phi2_);
    rl(0,2) = sin(phi1_)*sin(Phi_);
    rl(1,2) = -cos(phi1_)*sin(Phi_);
    rl(2,2) = cos(Phi_);

    // Set elastic tensor in lattice frame
    Intrepid::Tensor4<RealType, EC::MAX_NUM_DIM> C;
    C.set_dimension(num_dims_);
    // Initialize with zeros
    C.fill(Intrepid::ZEROS);
    // fill tensor
    C(0,0,0,0) = c11_;
    C(1,1,1,1) = c22_;
    C(2,2,2,2) = c33_;
    C(0,0,1,1) = c12_;
    C(1,1,0,0) = c12_;
    C(0,0,2,2) = c13_;
    C(2,2,0,0) = c13_;
    C(1,1,2,2) = c23_;
    C(2,2,1,1) = c23_;
    C(0,1,0,1) = c66_;
    C(1,0,1,0) = c66_;
    C(0,1,1,0) = c66_;
    C(1,0,0,1) = c66_;
    C(2,0,2,0) = c55_;
    C(0,2,0,2) = c55_;
    C(2,0,0,2) = c55_;
    C(0,2,0,0) = c55_;
    C(2,1,2,1) = c44_;
    C(1,2,1,2) = c44_;
    C(1,2,2,1) = c44_;
    C(2,1,1,2) = c44_;
    C(0,0,0,2) = c15_;
    C(0,0,2,0) = c15_;
    C(0,2,0,0) = c15_;
    C(2,0,0,0) = c15_;
    C(1,1,0,2) = c25_;
    C(1,1,2,0) = c25_;
    C(0,2,1,1) = c25_;
    C(2,0,1,1) = c25_;
    C(2,2,0,2) = c35_;
    C(2,2,2,0) = c35_;
    C(0,2,2,2) = c35_;
    C(2,0,2,2) = c35_;
    C(1,2,0,1) = c46_;
    C(1,2,1,0) = c46_;
    C(2,1,0,1) = c46_;
    C(2,1,1,0) = c46_;
    C(0,1,1,2) = c46_;
    C(1,0,1,2) = c46_;
    C(0,1,2,1) = c46_;
    C(1,0,2,1) = c46_;
    
    // Form rotate elasticity tensor
    for ( int i = 0; i < num_dims_; ++i )
      {
	for ( int j = 0; j < num_dims_; ++j )
	  {
	    for ( int k = 0; k < num_dims_; ++k )
	      {
		for ( int l = 0; l < num_dims_; ++l )
		  {
		    C_(i,j,k,l) = 0.0;
		    for ( int i1 = 0; i1 < num_dims_; ++i1 )
		      {
			for ( int j1 = 0; j1 < num_dims_; ++j1 )
			  {
			    for ( int k1 = 0; k1 < num_dims_; ++k1 )
			      {
				for ( int l1 = 0; l1 < num_dims_; ++l1 )
				  {
				    C_(i,j,k,l) = C_(i,j,k,l) + rl(i,i1)*rl(j,j1)*rl(k,k1)*rl(l,l1)*C(i1,j1,k1,l1);
				  }
			      }
			  }
		      }
		  }
	      }
	  }
      }

    std::string F_string = (*field_name_map_)["F"];
    std::string J_string = (*field_name_map_)["J"];
    std::string cauchy = (*field_name_map_)["Cauchy_Stress"];

    // define the dependent fields
    this->dep_field_map_.insert( std::make_pair(F_string, dl->qp_tensor) );
    this->dep_field_map_.insert( std::make_pair(J_string, dl->qp_scalar) );
    // this->dep_field_map_.insert( std::make_pair("Poissons Ratio", dl->qp_scalar) );
    // this->dep_field_map_.insert( std::make_pair("Elastic Modulus", dl->qp_scalar) );

    // define the evaluated fields
    this->eval_field_map_.insert( std::make_pair(cauchy, dl->qp_tensor) );
    this->eval_field_map_.insert( std::make_pair("Energy", dl->qp_scalar) );
    this->eval_field_map_.insert( std::make_pair("Material Tangent", dl->qp_tensor4) );

    // define the state variables
    this->num_state_variables_++;
    this->state_var_names_.push_back(cauchy);
    this->state_var_layouts_.push_back(dl->qp_tensor);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(false);
    this->state_var_output_flags_.push_back(p->get<bool>("Output Cauchy Stress", false));
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void ElasticCrystalModel< EvalT, Traits > ::
  computeState( typename Traits::EvalData workset,
		std::map< std::string, Teuchos::RCP< PHX::MDField< ScalarT > > > dep_fields,
		std::map< std::string, Teuchos::RCP< PHX::MDField< ScalarT > > > eval_fields )
  {
    // std::string F_string = (*field_name_map_)["F"];
  //   std::string J_string = (*field_name_map_)["J"];
  //   std::string cauchy = (*field_name_map_)["Cauchy_Stress"];

  //   // extract dependent MDFields
  //   PHX::MDField< ScalarT > def_grad = *dep_fields[F_string];
  //   PHX::MDField< ScalarT > J = *dep_fields[J_string];
  //   // extract evaluated MDFields
  //   PHX::MDField< ScalarT > stress = *eval_fields[cauchy];
  //   PHX::MDField< ScalarT > energy = *eval_fields["Energy"];
  //   PHX::MDField< ScalarT > tangent = *eval_fields["Material Tangent"];
  //   ScalarT kappa;
  //   ScalarT mu, mubar;
  //   ScalarT Jm53, Jm23;
  //   ScalarT smag;

  //   Intrepid::Tensor<ScalarT> F(num_dims_), b(num_dims_), sigma(num_dims_);
  //   Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));
  //   Intrepid::Tensor<ScalarT> s(num_dims_), n(num_dims_);

  //   Intrepid::Tensor4<ScalarT> dsigmadb;
  //   Intrepid::Tensor4<ScalarT> I1(Intrepid::identity_1<ScalarT>(num_dims_));
  //   Intrepid::Tensor4<ScalarT> I3(Intrepid::identity_3<ScalarT>(num_dims_));

  //   for (int cell(0); cell < workset.numCells; ++cell) 
  //     {
  //   	for (int pt(0); pt < num_pts_; ++pt) 
  //   	  {
  //   	    kappa =
  //   	      elastic_modulus(cell, pt)
  //             / (3. * (1. - 2. * poissons_ratio(cell, pt)));
  //   	    mu =
  //   	      elastic_modulus(cell, pt) / (2. * (1. + poissons_ratio(cell, pt)));
  //   	    Jm53 = std::pow(J(cell, pt), -5. / 3.);
  //   	    Jm23 = Jm53 * J(cell, pt);

  //   	    F.fill(def_grad,cell, pt,0,0);
  //   	    b = F * transpose(F);
  //   	    mubar = (1.0 / 3.0) * mu * Jm23 * Intrepid::trace(b);

  //   	    sigma = 0.5 * kappa * (J(cell, pt) - 1. / J(cell, pt)) * I
  //   	      + mu * Jm53 * Intrepid::dev(b);

  //   	    for (int i = 0; i < num_dims_; ++i) 
  //   	      {
  //   		for (int j = 0; j < num_dims_; ++j) 
  //   		  {
  //   		    stress(cell, pt, i, j) = sigma(i, j);
  //   		  }
  //   	      }

  //   	    if (compute_energy_) 
  //   	      { // compute energy
  //   		energy(cell, pt) =
  //   		  0.5 * kappa
  //   		  * (0.5 * (J(cell, pt) * J(cell, pt) - 1.0)
  //   		     - std::log(J(cell, pt)))
  //   		  + 0.5 * mu * (Jm23 * Intrepid::trace(b) - 3.0);
  //   	      }

  //   	    if (compute_tangent_) 
  //   	      { // compute tangent
  //   		s = Intrepid::dev(sigma);
  //   		smag = Intrepid::norm(s);
  //   		n = s / smag;

  //   		dsigmadb =
  //   		  kappa * J(cell, pt) * J(cell, pt) * I3
  //   		  - kappa * (J(cell, pt) * J(cell, pt) - 1.0) * I1
  //   		  + 2.0 * mubar * (I1 - (1.0 / 3.0) * I3)
  //   		  - 2.0 / 3.0 * smag
  //   		  * (Intrepid::tensor(n, I) + Intrepid::tensor(I, n));

  //   		for (int i = 0; i < num_dims_; ++i) 
  //   		  {
  //   		    for (int j = 0; j < num_dims_; ++j) 
  //   		      {
  //   			for (int k = 0; k < num_dims_; ++k) 
  //   			  {
  //   			    for (int l = 0; l < num_dims_; ++l) 
  //   			      {
  //   				tangent(cell, pt, i, j, k, l) = dsigmadb(i, j, k, l);
  //   			      }
  //   			  }
  //   		      }
  //   		  }
  //   	      }
  //   	  }
  //     }

  //   if (have_temperature_) 
  //     {
  //   	for (int cell(0); cell < workset.numCells; ++cell) 
  //   	  {
  //   	    for (int pt(0); pt < num_pts_; ++pt) 
  //   	      {
  //   		F.fill(def_grad,cell,pt,0,0);
  //   		ScalarT J = Intrepid::det(F);
  //   		sigma.fill(stress,cell,pt,0,0);
  //   		sigma -= 3.0 * expansion_coeff_ * (1.0 + 1.0 / (J*J))
  //   		  * (temperature_(cell,pt) - ref_temperature_) * I;

  //   		for (int i = 0; i < num_dims_; ++i) 
  //   		  {
  //   		    for (int j = 0; j < num_dims_; ++j)
  //   		      {
  //   			stress(cell, pt, i, j) = sigma(i, j);
  //   		      }
  //   		  }
  //   	      }
  //   	  }
  //     }
  }
  //----------------------------------------------------------------------------

}

