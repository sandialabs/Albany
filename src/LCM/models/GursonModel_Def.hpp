//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "LocalNonlinearSolver.h"

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  GursonModel<EvalT, Traits>::
  GursonModel(Teuchos::ParameterList* p,
              const Teuchos::RCP<Albany::Layouts>& dl):
    LCM::ConstitutiveModel<EvalT,Traits>(p,dl),
    sat_mod_(p->get<RealType>("Saturation Modulus", 0.0)),
    sat_exp_(p->get<RealType>("Saturation Exponent", 0.0)),
    f0_(p->get<RealType>("Initial Void Volume", 0.0)),
    kw_(p->get<RealType>("Shear Damage Parameter", 0.0)),
    eN_(p->get<RealType>("Void Nucleation Parameter 1", 0.0)),
    sN_(p->get<RealType>("Void Nucleation Parameter 2", 0.0)),
    fN_(p->get<RealType>("Void Nucleation Parameter 3", 0.0)),
    fc_(p->get<RealType>("Critical Void Volume", 1.0)),
    ff_(p->get<RealType>("Failure Void Volume", 1.0)),
    q1_(p->get<RealType>("Yield Parameter Q1", 1.0)),
    q2_(p->get<RealType>("Yield Parameter Q2", 1.0)),
    q3_(p->get<RealType>("Yield Parameter Q3", 1.0))
  {
    // define the dependent fields
    this->dep_field_map_.insert( std::make_pair("F", dl->qp_tensor) );
    this->dep_field_map_.insert( std::make_pair("J", dl->qp_scalar) );
    this->dep_field_map_.insert( std::make_pair("Poissons Ratio", dl->qp_scalar) );
    this->dep_field_map_.insert( std::make_pair("Elastic Modulus", dl->qp_scalar) );
    this->dep_field_map_.insert( std::make_pair("Yield Strength", dl->qp_scalar) );
    this->dep_field_map_.insert( std::make_pair("Hardening Modulus", dl->qp_scalar) );

    // retrive appropriate field name strings
    std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
    std::string Fp_string = (*field_name_map_)["Fp"];
    std::string eqps_string = (*field_name_map_)["eqps"];
    std::string void_string = (*field_name_map_)["Void_Volume"];

    // define the evaluated fields
    this->eval_field_map_.insert( std::make_pair(cauchy_string, dl->qp_tensor) );
    this->eval_field_map_.insert( std::make_pair(Fp_string, dl->qp_tensor) );
    this->eval_field_map_.insert( std::make_pair(eqps_string, dl->qp_scalar) );
    this->eval_field_map_.insert( std::make_pair(void_string, dl->qp_scalar) );

    // define the state variables
    //
    // stress
    this->num_state_variables_++;
    this->state_var_names_.push_back(cauchy_string);
    this->state_var_layouts_.push_back(dl->qp_tensor);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(false);
    this->state_var_output_flags_.push_back(true);
    //
    // Fp
    this->num_state_variables_++;
    this->state_var_names_.push_back(Fp_string);
    this->state_var_layouts_.push_back(dl->qp_tensor);
    this->state_var_init_types_.push_back("identity");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(false);
    //
    // eqps
    this->num_state_variables_++;
    this->state_var_names_.push_back(eqps_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(true);
    //
    // void volume
    this->num_state_variables_++;
    this->state_var_names_.push_back(void_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(true);
  }
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void GursonModel<EvalT, Traits>::
  computeEnergy(typename Traits::EvalData workset,
                std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
                std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
  {
    // not implemented
  }
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void GursonModel<EvalT, Traits>::
  computeState(typename Traits::EvalData workset,
               std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
               std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
  {
    // extract dependent MDFields
    PHX::MDField<ScalarT> def_grad          = *dep_fields["F"];
    PHX::MDField<ScalarT> J                 = *dep_fields["J"];
    PHX::MDField<ScalarT> poissons_ratio    = *dep_fields["Poissons Ratio"];
    PHX::MDField<ScalarT> elastic_modulus   = *dep_fields["Elastic Modulus"];
    PHX::MDField<ScalarT> yield_strength    = *dep_fields["Yield Strength"];
    PHX::MDField<ScalarT> hardening_modulus = *dep_fields["Hardening Modulus"];

    // retrive appropriate field name strings
    std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
    std::string Fp_string     = (*field_name_map_)["Fp"];
    std::string eqps_string   = (*field_name_map_)["eqps"];
    std::string void_string   = (*field_name_map_)["Void_Volume"];

    // extract evaluated MDFields
    PHX::MDField<ScalarT> stress        = *eval_fields[cauchy_string];
    PHX::MDField<ScalarT> Fp            = *eval_fields[Fp_string];
    PHX::MDField<ScalarT> eqps          = *eval_fields[eqps_string];
    PHX::MDField<ScalarT> void_volume   = *eval_fields[void_string];

    // get State Variables
    Albany::MDArray Fp_old   = 
      (*workset.stateArrayPtr)[Fp_string+"_old"];
    Albany::MDArray eqps_old = 
      (*workset.stateArrayPtr)[eqps_string+"_old"];
    Albany::MDArray void_volume_old = 
      (*workset.stateArrayPtr)[void_string+"_old"];

    ScalarT kappa, mu, mubar, K, Y;
    ScalarT Jm23, trace, smag2, smag, f, p, dgam;
    ScalarT sq23(std::sqrt(2./3.));

    Intrepid::Tensor<ScalarT> F(num_dims_), be(num_dims_), s(num_dims_), sigma(num_dims_);
    Intrepid::Tensor<ScalarT> N(num_dims_), A(num_dims_), expA(num_dims_), Fpnew(num_dims_);
    Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));
    Intrepid::Tensor<ScalarT> Fpn(num_dims_), Fpinv(num_dims_), Cpinv(num_dims_);

    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t pt(0); pt < num_pts_; ++pt) {
        kappa =  elastic_modulus(cell,pt)/(3.*(1.-2.*poissons_ratio(cell,pt)));
        mu = elastic_modulus(cell,pt)/(2.*(1.+poissons_ratio(cell,pt)));
        K = hardening_modulus(cell,pt);
        Y = yield_strength(cell,pt);
        Jm23 = std::pow(J(cell,pt),-2./3.);

        // fill local tensors
        F.fill( &def_grad(cell,pt,0,0) );
        //Fpn.fill( &Fpold(cell,pt,std::size_t(0),std::size_t(0)) );
        for ( std::size_t i(0); i < num_dims_; ++i) {
          for ( std::size_t j(0); j < num_dims_; ++j) {
            Fpn(i,j) = static_cast<ScalarT>(Fp_old(cell,pt,i,j));
          }                    
        }
        
        // compute trial state
        Fpinv = Intrepid::inverse(Fpn);
        Cpinv = Fpinv * Intrepid::transpose(Fpinv);
        be = Jm23 * F * Cpinv * Intrepid::transpose(F);
        s = mu * Intrepid::dev(be);
        mubar = Intrepid::trace(be)*mu;
        
        // check yield condition
        smag = Intrepid::norm(s);
        f = smag - sq23*(Y + K*eqps_old(cell,pt) 
                         + sat_mod_*(1.-std::exp(-sat_exp_*eqps_old(cell,pt))));

        if (f > 1E-12) {
          // return mapping algorithm
          bool converged = false;
          ScalarT g = f;
          ScalarT H = 0.0;
          ScalarT dH = 0.0;
          ScalarT alpha = 0.0;
          ScalarT res = 0.0;
          int count = 0;
          dgam = 0.0;

          LocalNonlinearSolver<EvalT, Traits> solver;

          std::vector<ScalarT> F(1);
          std::vector<ScalarT> dFdX(1);
          std::vector<ScalarT> X(1);

          F[0] = f;
          X[0] = 0.0;
          dFdX[0] = ( -2. * mubar ) * ( 1. + H / ( 3. * mubar ) );
          while (!converged && count < 30)
          {
            count++;
            solver.solve(dFdX,X,F);
            alpha = eqps_old(cell, pt) + sq23 * X[0];
            H = K * alpha + sat_mod_*( 1. - exp( -sat_exp_ * alpha ) );
            dH = K + sat_exp_ * sat_mod_ * exp( -sat_exp_ * alpha );
            F[0] = smag -  ( 2. * mubar * X[0] + sq23 * ( Y + H ) );
            dFdX[0] = -2. * mubar * ( 1. + dH / ( 3. * mubar ) );

            res = std::abs(F[0]);
            if ( res < 1.e-11 || res/f < 1.E-11 )
              converged = true;

            TEUCHOS_TEST_FOR_EXCEPTION( count > 30, std::runtime_error,
                                        std::endl << 
                                        "Error in return mapping, count = " <<
                                        count <<
                                        "\nres = " << res <<
                                        "\nrelres = " << res/f <<
                                        "\ng = " << F[0] <<
                                        "\ndg = " << dFdX[0] <<
                                        "\nalpha = " << alpha << std::endl);
          }
          solver.computeFadInfo(dFdX,X,F);
          dgam = X[0];

          // plastic direction
          N = (1/smag) * s;

          // update s
          s -= 2*mubar*dgam*N;

          // update eqps
          eqps(cell,pt) = alpha;

          // exponential map to get Fpnew
          A = dgam*N;
          expA = Intrepid::exp(A);
          Fpnew = expA * Fpn;
          for (std::size_t i(0); i < num_dims_; ++i) {
            for (std::size_t j(0); j < num_dims_; ++j) {
              Fp(cell,pt,i,j) = Fpnew(i,j);
            }
          }
        } else {
          eqps(cell, pt) = eqps_old(cell,pt);
          for (std::size_t i(0); i < num_dims_; ++i) {
            for (std::size_t j(0); j < num_dims_; ++j) {
              Fp(cell,pt,i,j) = Fpn(i,j);
            }
          }
        }

        // compute pressure
        p = 0.5*kappa*(J(cell,pt)-1./(J(cell,pt)));

        // compute stress
        sigma = p*I + s/J(cell,pt);
        for (std::size_t i(0); i < num_dims_; ++i) {
          for (std::size_t j(0); j < num_dims_; ++j) {
            stress(cell,pt,i,j) = sigma(i,j);
          }
        }
      }
    }
  }
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void GursonModel<EvalT, Traits>::
  computeTangent(typename Traits::EvalData workset,
                 std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
                 std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
  {
    // not implemented
  }
  //----------------------------------------------------------------------------
} 

