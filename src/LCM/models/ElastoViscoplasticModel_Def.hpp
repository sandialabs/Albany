//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "LocalNonlinearSolver.hpp"

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
ElastoViscoplasticModel<EvalT, Traits>::
ElastoViscoplasticModel(Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    LCM::ConstitutiveModel<EvalT, Traits>(p, dl)
{
  // retrive appropriate field name strings
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string = (*field_name_map_)["Fp"];
  std::string eqps_string = (*field_name_map_)["eqps"];
  std::string eps_ss_string = (*field_name_map_)["eps_ss"];
  std::string kappa_string = (*field_name_map_)["isotropic_hardening"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair(F_string, dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair(J_string, dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Poissons Ratio", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Elastic Modulus", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Yield Strength", dl->qp_scalar));
  this->dep_field_map_.insert(
      std::make_pair("Flow Rule Coefficient", dl->qp_scalar));
  this->dep_field_map_.insert(
      std::make_pair("Flow Rule Exponent", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Yield Strength", dl->qp_scalar));
  this->dep_field_map_.insert(
      std::make_pair("Hardening Modulus", dl->qp_scalar));
  this->dep_field_map_.insert(
      std::make_pair("Recovery Modulus", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Delta Time", dl->workset_scalar));

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(Fp_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(eqps_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(eps_ss_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(kappa_string, dl->qp_scalar));
  if (have_temperature_) {
    this->eval_field_map_.insert(std::make_pair(source_string, dl->qp_scalar));
  }

  // define the state variables
  //
  // stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(cauchy_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Cauchy Stress", false));
  //
  // Fp
  this->num_state_variables_++;
  this->state_var_names_.push_back(Fp_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("identity");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Fp", false));
  //
  // eqps
  this->num_state_variables_++;
  this->state_var_names_.push_back(eqps_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output eqps", false));
  //
  // epsilon_ss, statisically stored dislocations
  this->num_state_variables_++;
  this->state_var_names_.push_back(eps_ss_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output eps_ss", false));
  //
  // kappa - isotropic hardening
  this->num_state_variables_++;
  this->state_var_names_.push_back(kappa_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output kappa", false));
  //
  // mechanical source
  if (have_temperature_) {
    this->num_state_variables_++;
    this->state_var_names_.push_back(source_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(false);
    this->state_var_output_flags_.push_back(p->get<bool>("Output Mechanical Source", false));
  }
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ElastoViscoplasticModel<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
{
#ifdef ALBANY_SG_MP
  //amb-remove Currently does not compile when ALBANY_SG_MP is enabled.
#pragma message(__FILE__": computeState is temporarily disabled when ALBANY_SG_MP is on.")
#else
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string = (*field_name_map_)["Fp"];
  std::string eqps_string = (*field_name_map_)["eqps"];
  std::string eps_ss_string = (*field_name_map_)["eps_ss"];
  std::string kappa_string = (*field_name_map_)["isotropic_hardening"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];

  // extract dependent MDFields
  PHX::MDField<ScalarT> def_grad_field = *dep_fields[F_string];
  PHX::MDField<ScalarT> J = *dep_fields[J_string];
  PHX::MDField<ScalarT> poissons_ratio = *dep_fields["Poissons Ratio"];
  PHX::MDField<ScalarT> elastic_modulus = *dep_fields["Elastic Modulus"];
  PHX::MDField<ScalarT> yield_strength = *dep_fields["Yield Strength"];
  PHX::MDField<ScalarT> hardening_modulus = *dep_fields["Hardening Modulus"];
  PHX::MDField<ScalarT> recovery_modulus = *dep_fields["Recovery Modulus"];
  PHX::MDField<ScalarT> flow_exp = *dep_fields["Flow Rule Exponent"];
  PHX::MDField<ScalarT> flow_coeff = *dep_fields["Flow Rule Coefficient"];
  PHX::MDField<ScalarT> delta_time = *dep_fields["Delta Time"];

  // extract evaluated MDFields
  PHX::MDField<ScalarT> stress_field = *eval_fields[cauchy_string];
  PHX::MDField<ScalarT> Fp_field = *eval_fields[Fp_string];
  PHX::MDField<ScalarT> eqps_field = *eval_fields[eqps_string];
  PHX::MDField<ScalarT> eps_ss_field = *eval_fields[eps_ss_string];
  PHX::MDField<ScalarT> kappa_field = *eval_fields[kappa_string];
  PHX::MDField<ScalarT> source_field;
  if (have_temperature_) {
    source_field = *eval_fields[source_string];
  }

  // get State Variables
  Albany::MDArray Fp_field_old     = (*workset.stateArrayPtr)[Fp_string + "_old"];
  Albany::MDArray eqps_field_old   = (*workset.stateArrayPtr)[eqps_string + "_old"];
  Albany::MDArray eps_ss_field_old = (*workset.stateArrayPtr)[eps_ss_string + "_old"];
  Albany::MDArray kappa_field_old  = (*workset.stateArrayPtr)[kappa_string + "_old"];

  // define constants
  RealType sq23(std::sqrt(2. / 3.));
  RealType sq32(std::sqrt(3. / 2.));

  // pre-define some tensors that will be re-used below
  Intrepid::Tensor<ScalarT> F(num_dims_), be(num_dims_);
  Intrepid::Tensor<ScalarT> s(num_dims_), sigma(num_dims_);
  Intrepid::Tensor<ScalarT> N(num_dims_), A(num_dims_);
  Intrepid::Tensor<ScalarT> expA(num_dims_), Fpnew(num_dims_);
  Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));
  Intrepid::Tensor<ScalarT> Fpn(num_dims_), Cpinv(num_dims_), Fpinv(num_dims_);

  for (std::size_t cell(0); cell < workset.numCells; ++cell) {
    for (std::size_t pt(0); pt < num_pts_; ++pt) {
      ScalarT bulk = elastic_modulus(cell, pt)
          / (3. * (1. - 2. * poissons_ratio(cell, pt)));
      ScalarT mu = elastic_modulus(cell, pt) / (2. * (1. + poissons_ratio(cell, pt)));
      ScalarT Y = yield_strength(cell, pt);
      ScalarT Jm23 = std::pow(J(cell, pt), -2. / 3.);

      // assign local state variables
      //
      //ScalarT kappa = kappa_field(cell,pt);
      ScalarT kappa_old = kappa_field_old(cell,pt);
      ScalarT eps_ss = eps_ss_field(cell,pt);
      ScalarT eps_ss_old = eps_ss_field_old(cell,pt);
      ScalarT eqps_old = eqps_field_old(cell,pt);

      // fill local tensors
      //
      F.fill(&def_grad_field(cell, pt, 0, 0));
      for (std::size_t i(0); i < num_dims_; ++i) {
        for (std::size_t j(0); j < num_dims_; ++j) {
          Fpn(i, j) = ScalarT(Fp_field_old(cell, pt, i, j));
        }
      }

      // compute trial state
      // compute the Kirchhoff stress in the current configuration
      //
      Cpinv = Intrepid::inverse(Fpn) * Intrepid::transpose(Intrepid::inverse(Fpn));
      be = Jm23 * F * Cpinv * Intrepid::transpose(F);
      s = mu * Intrepid::dev(be);
      ScalarT smag = Intrepid::norm(s);
      ScalarT mubar = Intrepid::trace(be) * mu / (num_dims_);
      
      // check yield condition
      //
      ScalarT Phi = sq32 * smag - ( Y + kappa_old );

      std::cout << "======== Phi: " << Phi << std::endl;
      std::cout << "======== eps: " << std::numeric_limits<RealType>::epsilon() << std::endl;

      if (Phi > std::numeric_limits<RealType>::epsilon()) {

        // return mapping algorithm
        //
        bool converged = false;
        int iter = 0;
        RealType max_norm = std::numeric_limits<RealType>::min();

        // hardening and recovery parameters
        //
        ScalarT H = hardening_modulus(cell, pt);
        ScalarT Rd = recovery_modulus(cell, pt);

        // flow rule temperature dependent parameters
        //
        ScalarT f = flow_coeff(cell,pt);
        ScalarT n = flow_exp(cell,pt);

        // This solver deals with Sacado type info
        //
        LocalNonlinearSolver<EvalT, Traits> solver;

        // create some vectors to store solver data
        //
        std::vector<ScalarT> R(2);
        std::vector<ScalarT> dRdX(4);
        std::vector<ScalarT> X(2);

        // initial guess
        X[0] = 0.0;
        X[1] = eps_ss_old;

        // create a copy of be as a Fad
        Intrepid::Tensor<Fad> beF(num_dims_);
        for (std::size_t i = 0; i < num_dims_; ++i) {
          for (std::size_t j = 0; j < num_dims_; ++j) {
            beF(i, j) = be(i, j);
          }
        }
        Fad two_mubarF = 2.0 * Intrepid::trace(beF) * mu / (num_dims_);
        //Fad sq32F = std::sqrt(3.0/2.0);
        // FIXME this seems to be necessary to get PhiF to compile below
        // need to look into this more, it appears to be a conflict
        // between the Intrepid::norm and FadType operations
        //
        Fad smagF = smag;

        while (!converged) {

          // set up data types
          //
          std::vector<Fad> XFad(2);
          std::vector<Fad> RFad(2);
          std::vector<ScalarT> Xval(2);
          for (std::size_t i = 0; i < 2; ++i) {
            Xval[i] = Sacado::ScalarValue<ScalarT>::eval(X[i]);
            XFad[i] = Fad(2, i, Xval[i]);
          }

          // get solution vars
          //
          Fad dgamF = XFad[0];
          Fad eps_ssF = XFad[1];

          // compute yield function
          //
          Fad eqps_rateF = 0.0;
          if (delta_time(0) > 0) eqps_rateF = sq23 * dgamF / delta_time(0);
          Fad rate_termF = 1.0 + std::asinh( std::pow(eqps_rateF / f, n));
          Fad kappaF = two_mubarF * eps_ssF;
          Fad PhiF = sq32 * (smagF - two_mubarF * dgamF) - ( Y + kappaF ) * rate_termF;

          // compute the hardening residual
          //
          Fad eps_resF = eps_ssF - eps_ss_old - (H - Rd*eps_ssF) * dgamF;

          // for convenience put the residuals into a container
          //
          RFad[0] = PhiF;
          RFad[1] = eps_resF;

          // extract the values of the residuals
          //
          for (int i = 0; i < 2; ++i)
            R[i] = RFad[i].val();

          // extract the sensitivities of the residuals
          //
          for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
              dRdX[i + 2 * j] = RFad[i].dx(j);

          // this call invokes the solver and updates the solution in X
          //
          solver.solve(dRdX, X, R);

          // compute the norm of the residual
          //
          RealType R0 = Sacado::ScalarValue<ScalarT>::eval(R[0]); 
          RealType R1 = Sacado::ScalarValue<ScalarT>::eval(R[1]);
          RealType norm_res = std::sqrt(R0*R0 + R1*R1);
          max_norm = std::max(norm_res, max_norm);
            
          // check against too many inerations
          //
          TEUCHOS_TEST_FOR_EXCEPTION(iter == 30, std::runtime_error,
                                     std::endl <<
                                     "Error in ElastoViscoplastic return mapping\n" <<
                                     "iter count = " << iter << "\n" << std::endl);

          // check for a sufficiently small residual
          //
          std::cout << "======== norm_res : " << norm_res << std::endl;
          if ( (norm_res/max_norm < 1.e-12) || (norm_res < 1.e-12) )
            converged = true;

          // increment the iteratio counter
          //
          iter++;
        }

        solver.computeFadInfo(dRdX, X, R);
        ScalarT dgam = X[0];
        ScalarT eps_ss = X[1];
        ScalarT kappa = 2.0 * mubar * eps_ss;

        std::cout << "======== dgam : " << dgam << std::endl;
        std::cout << "======== e_ss : " << eps_ss << std::endl;
        std::cout << "======== kapp : " << kappa << std::endl;

        // plastic direction
        N = (1 / smag) * s;

        // update s
        s -= 2 * mubar * dgam * N;

        // update state variables
        eps_ss_field(cell, pt) = eps_ss;
        eqps_field(cell,pt) = eqps_old + sq23 * dgam;
        kappa_field(cell,pt) = kappa;

        // mechanical source
        // FIXME this is not correct, just a placeholder
        //
        if (have_temperature_ && delta_time(0) > 0) {
          source_field(cell, pt) = (sq23 * dgam / delta_time(0))
            * (Y + kappa) / (density_ * heat_capacity_);
        }

        // exponential map to get Fpnew
        //
        A = dgam * N;
        expA = Intrepid::exp(A);
        Fpnew = expA * Fpn;
        for (std::size_t i(0); i < num_dims_; ++i) {
          for (std::size_t j(0); j < num_dims_; ++j) {
            Fp_field(cell, pt, i, j) = Fpnew(i, j);
          }
        }
      } else {
        // we are not yielding, variables do not evolve
        //
        eps_ss_field(cell, pt) = eps_ss_old;
        eqps_field(cell,pt) = eqps_old;
        kappa_field(cell,pt) = kappa_old;
        if (have_temperature_) source_field(cell, pt) = 0.0;
        for (std::size_t i(0); i < num_dims_; ++i) {
          for (std::size_t j(0); j < num_dims_; ++j) {
            Fp_field(cell, pt, i, j) = Fpn(i, j);
          }
        }
      }

      // compute pressure
      ScalarT p = 0.5 * bulk * (J(cell, pt) - 1. / (J(cell, pt)));

      // compute stress
      sigma = p * I + s / J(cell, pt);
      for (std::size_t i(0); i < num_dims_; ++i) {
        for (std::size_t j(0); j < num_dims_; ++j) {
          stress_field(cell, pt, i, j) = sigma(i, j);
        }
      }
    }
  }

  if (have_temperature_) {
    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t pt(0); pt < num_pts_; ++pt) {
        F.fill(&def_grad_field(cell,pt,0,0));
        ScalarT J = Intrepid::det(F);
        sigma.fill(&stress_field(cell,pt,0,0));
        sigma -= 3.0 * expansion_coeff_ * (1.0 + 1.0 / (J*J))
          * (temperature_(cell,pt) - ref_temperature_) * I;
        for (std::size_t i = 0; i < num_dims_; ++i) {
          for (std::size_t j = 0; j < num_dims_; ++j) {
            stress_field(cell, pt, i, j) = sigma(i, j);
          }
        }
      }
    }
  }
#endif
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
typename EvalT::ScalarT
ElastoViscoplasticModel<EvalT, Traits>::YieldFunction()
{}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void
ElastoViscoplasticModel<EvalT, Traits>::Residual()
{}
//------------------------------------------------------------------------------
}

