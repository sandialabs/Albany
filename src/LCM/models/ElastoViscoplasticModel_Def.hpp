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
  LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
  f0_(p->get<RealType>("Initial Void Volume", 0.0)),
  kw_(p->get<RealType>("Shear Damage Parameter", 0.0)),
  eN_(p->get<RealType>("Void Nucleation Parameter eN", 0.0)),
  sN_(p->get<RealType>("Void Nucleation Parameter sN", 0.1)),
  fN_(p->get<RealType>("Void Nucleation Parameter fN", 0.0)),
  eHN_(p->get<RealType>("Hydrogen Mean Strain Nucleation Parameter", 0.0)),
  eHN_coeff_(p->get<RealType>("Mean Strain Hydrogen Linear Coefficient", 0.0)),
  sHN_(p->get<RealType>("Hydrogen Nucleation Standard Deviation", 0.1)),
  fHeN_(p->get<RealType>("Void Volume Fraction Nucleation Parameter", 0.0)),
  fHeN_coeff_(p->get<RealType>("Void Volume Fraction He Linear Coefficient", 0.0)),
  fc_(p->get<RealType>("Critical Void Volume", 1.0)),
  ff_(p->get<RealType>("Failure Void Volume", 1.0)),
  q1_(p->get<RealType>("Yield Parameter q1", 1.0)),
  q2_(p->get<RealType>("Yield Parameter q2", 1.0)),
  q3_(p->get<RealType>("Yield Parameter q3", 1.0)),
  alpha1_(p->get<RealType>("Hydrogen Yield Parameter", 0.0)),
  alpha2_(p->get<RealType>("Helium Yield Parameter", 0.0)),
  Ra_(p->get<RealType>("Helium Radius", 0.0)),
  print_(p->get<bool>("Output Convergence", false))
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
  std::string void_volume_fraction_string = (*field_name_map_)["void_volume_fraction"];

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
  this->eval_field_map_.insert(std::make_pair(void_volume_fraction_string, dl->qp_scalar));
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
  // void volume fraction
  this->num_state_variables_++;
  this->state_var_names_.push_back(void_volume_fraction_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(f0_);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output void volume fraction", false));
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
  // get strings from field_name_map in order to extract MDFields
  //
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string = (*field_name_map_)["Fp"];
  std::string eqps_string = (*field_name_map_)["eqps"];
  std::string eps_ss_string = (*field_name_map_)["eps_ss"];
  std::string kappa_string = (*field_name_map_)["isotropic_hardening"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];
  std::string void_volume_fraction_string = (*field_name_map_)["void_volume_fraction"];

  // extract dependent MDFields
  //
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
  //
  PHX::MDField<ScalarT> stress_field = *eval_fields[cauchy_string];
  PHX::MDField<ScalarT> Fp_field = *eval_fields[Fp_string];
  PHX::MDField<ScalarT> eqps_field = *eval_fields[eqps_string];
  PHX::MDField<ScalarT> eps_ss_field = *eval_fields[eps_ss_string];
  PHX::MDField<ScalarT> kappa_field = *eval_fields[kappa_string];
  PHX::MDField<ScalarT> void_volume_fraction_field = *eval_fields[void_volume_fraction_string];
  PHX::MDField<ScalarT> source_field;
  if (have_temperature_) {
    source_field = *eval_fields[source_string];
  }

  // get State Variables
  //
  Albany::MDArray Fp_field_old     = (*workset.stateArrayPtr)[Fp_string + "_old"];
  Albany::MDArray eqps_field_old   = (*workset.stateArrayPtr)[eqps_string + "_old"];
  Albany::MDArray eps_ss_field_old = (*workset.stateArrayPtr)[eps_ss_string + "_old"];
  Albany::MDArray kappa_field_old  = (*workset.stateArrayPtr)[kappa_string + "_old"];
  Albany::MDArray void_volume_fraction_field_old  = (*workset.stateArrayPtr)[void_volume_fraction_string + "_old"];

  // define constants
  //
  const RealType sq23(std::sqrt(2. / 3.));
  const RealType sq32(std::sqrt(3. / 2.));
  const RealType pi    = 3.141592653589793;
  const RealType radius_fac(3.0/(4.0*pi));

  // void nucleation constants
  ScalarT H_mean_eps_ss(eHN_), He_void_vol_frac_nuc(fHeN_);
  
  // pre-define some tensors that will be re-used below
  //
  Intrepid::Tensor<ScalarT> F(num_dims_), be(num_dims_), bebar(num_dims_);
  Intrepid::Tensor<ScalarT> s(num_dims_), sigma(num_dims_);
  Intrepid::Tensor<ScalarT> N(num_dims_), A(num_dims_);
  Intrepid::Tensor<ScalarT> expA(num_dims_), Fpnew(num_dims_);
  Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));
  Intrepid::Tensor<ScalarT> Fpn(num_dims_), Cpinv(num_dims_), Fpinv(num_dims_);

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      ScalarT bulk = elastic_modulus(cell, pt)
        / (3. * (1. - 2. * poissons_ratio(cell, pt)));
      ScalarT mu = elastic_modulus(cell, pt) / (2. * (1. + poissons_ratio(cell, pt)));
      ScalarT Y = yield_strength(cell, pt);

      // adjustment to the yield strength in the presence of hydrogen
      //
      if (have_total_concentration_) {
        Y += alpha1_ * total_concentration_(cell,pt);
        H_mean_eps_ss = eHN_ + eHN_coeff_ * total_concentration_(cell,pt);
      }

      // adjustment to the yield strength in the presence of helium
      //
      if (have_total_bubble_density_ && have_bubble_volume_fraction_) {
        if (total_bubble_density_(cell,pt) > 0.0 && bubble_volume_fraction_(cell,pt) > 0.0) {
          ScalarT Rb = std::cbrt(radius_fac * bubble_volume_fraction_(cell,pt)/total_bubble_density_(cell,pt)); 
          Y += alpha2_ * (Rb*Rb)/(Ra_*Ra_);
          He_void_vol_frac_nuc = fHeN_ + fHeN_coeff_ * bubble_volume_fraction_(cell,pt);
        }
      }

      // assign local state variables
      // eps_ss is a scalar internal strain measure
      // kappa is a scalar internal strength = 2 mu * eps_ss
      // eqps is equivalent plastic strain
      // void volume fraction ~ damage
      //
      ScalarT kappa_old = kappa_field_old(cell,pt);
      ScalarT eps_ss = eps_ss_field(cell,pt);
      ScalarT eps_ss_old = eps_ss_field_old(cell,pt);
      ScalarT eqps_old = eqps_field_old(cell,pt);
      ScalarT void_volume_fraction_old = void_volume_fraction_field_old(cell,pt);

      // check to see if this point has exceeded its critical void volume fraction
      // if so, skip and set stress to zero (below)
      //
      bool failed(false);
      if (Sacado::ScalarValue<ScalarT>::eval(void_volume_fraction_old) >= ff_) failed = true;

      if (!failed) {
        // fill local tensors
        //
        F.fill(def_grad_field, cell, pt, 0, 0);

        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            Fpn(i, j) = ScalarT(Fp_field_old(cell, pt, i, j));
          }
        }

        // compute trial state
        // compute the Kirchhoff stress in the current configuration
        //
        // calculate \f$ Cp_n^{-1} \f$
        //
        Cpinv = Intrepid::inverse(Fpn) * Intrepid::transpose(Intrepid::inverse(Fpn));

        // calculate \f$ b^{e} = F {C^{p}}^{-1} F^{T} \f$
        //
        be = F * Cpinv * Intrepid::transpose(F);

        // calculate the determinant of the deformation gradient: \f$ J = det[F] \f$
        //
        ScalarT Je = std::sqrt(Intrepid::det(be));
        bebar = std::pow(Je, -2.0/3.0) * be;
        ScalarT mubar = Intrepid::trace(be) * mu / (num_dims_);

        // calculate trial deviatoric stress \f$ s^{tr} = \mu dev(b^{e}) \f$
        //
        s = mu * Intrepid::dev(bebar);
        ScalarT smag = Intrepid::norm(s);

        // calculate trial (Kirchhoff) pressure
        //
        ScalarT p = 0.5 * bulk * (Je * Je - 1.0);

        // check yield condition
        // assumes no rate effects
        //
        ScalarT Ybar = Je * (Y + kappa_old);
        ScalarT arg = 1.5 * q2_ * p / Ybar;
        ScalarT fstar = compute_fstar(void_volume_fraction_old, fc_, ff_, q1_);
        ScalarT psi = 1.0 + q3_ * fstar * fstar - 2.0 * q1_ * fstar * std::cosh(arg);

        // Gurson quadratic yield surface
        //
        ScalarT Phi = 0.5 * Intrepid::dotdot(s,s) - psi * Ybar * Ybar / 3.0;

        // check yield condition
        //
        if (Phi > std::numeric_limits<RealType>::epsilon()) {

          // return mapping algorithm
          //
          bool converged = false;
          int iter(0);
          const int max_iter(30);
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
          const int num_vars(5);
          std::vector<ScalarT> R(num_vars);
          std::vector<ScalarT> dRdX(num_vars*num_vars);
          std::vector<ScalarT> X(num_vars);

          // FIXME: the initial guess needs some work, not active
          // initial guess
          //
          // ScalarT dgam_tr = std::sqrt(smag/(2.0 * mubar * Phi));
          // ScalarT eps_ss_tr = eps_ss_old + delta_time(0) * (H - Rd * eps_ss_old) * dgam_tr;
          // ScalarT kappa_tr = 2.0 * mu * eps_ss_tr;
          // ScalarT Ybar_tr = Je * (Y + kappa_tr);
          // ScalarT arg_tr = 1.5 * q2_ * p / Ybar_tr;
          // ScalarT p_tr = p - delta_time(0) * (dgam_tr * q1_ * q2_ * bulk * Ybar_tr * fstar * std::sinh(arg_tr)) / bulk;
          // arg_tr = 1.5 * q2_ * p_tr / Ybar_tr;
          // ScalarT void_tr = void_volume_fraction_old + delta_time(0) * (dgam_tr * q1_ * q2_ * ( 1.0 - fstar ) * fstar * Ybar_tr * std::sinh(arg_tr));
          // ScalarT eqps_tr = eqps_old + delta_time(0) * (dgam_tr * ((q1_ * q2_ * p * Ybar_tr * fstar * std::sinh(arg_tr)) / (1.0 - fstar) / Ybar_tr + smag * smag / (1.0 - fstar) / Ybar_tr));

          X[0] = 0.0;
          X[1] = eps_ss_old;
          X[2] = p;
          X[3] = void_volume_fraction_old;
          X[4] = eqps_old;

          // *!*!*
          // now below we introduce a local 'Fad' type
          // this is specifically for the nonlinear solve for our constitutive model
          // create a copy of be as a Fad
          //
          Intrepid::Tensor<Fad> beF(num_dims_);
          for (std::size_t i = 0; i < num_dims_; ++i) {
            for (std::size_t j = 0; j < num_dims_; ++j) {
              beF(i, j) = be(i, j);
            }
          }
          Fad two_mubarF = 2.0 * Intrepid::trace(beF) * mu / (num_dims_);

          // FIXME this seems to be necessary to get PhiF to compile below
          // need to look into this more, it appears to be a conflict
          // between the Intrepid::norm and FadType operations
          //
          Fad smagF = smag;

          // check for convergence
          //
          while (!converged) {

            // set up data types
            // again inside this loop everything is a local 'Fad'
            std::vector<Fad> XFad(num_vars);
            std::vector<Fad> RFad(num_vars);
            std::vector<ScalarT> Xval(num_vars);
            for (std::size_t i = 0; i < num_vars; ++i) {
              Xval[i] = Sacado::ScalarValue<ScalarT>::eval(X[i]);
              XFad[i] = Fad(num_vars, i, Xval[i]);
            }

            // get solution vars
            // NOTE: we have 5 independent variables
            // dgam - plastic increment
            // eps_ss - internal strain
            // p - pressure
            // void_volume_fraction
            // eqps
            //
            Fad dgamF = XFad[0];
            Fad eps_ssF = XFad[1];
            Fad pF = XFad[2];
            Fad void_volume_fractionF = XFad[3];
            Fad eqpsF = XFad[4];

            // account for void coalescence
            //
            Fad fstarF = compute_fstar(void_volume_fractionF, fc_, ff_, q1_);

            // compute yield stress and rate terms
            //
            Fad eqps_rateF = 0.0;
            if (delta_time(0) > 0) eqps_rateF = sq23 * dgamF / delta_time(0);
            Fad rate_termF = 1.0 + std::asinh( std::pow(eqps_rateF / f, n));
            Fad kappaF = two_mubarF * eps_ssF;
            Fad YbarF = Je * (Y + kappaF) * rate_termF;

            // arguments that feed into the yield function
            //
            Fad argF = ( 1.5 * q2_ * pF ) / YbarF;
            Fad psiF = 1. + q3_ * fstarF * fstarF - 2. * q1_ * fstarF * std::cosh(argF);
            Fad factor = 1.0 / ( 1.0 + ( two_mubarF * dgamF) );

            // deviatoric stress
            // 
            Intrepid::Tensor<Fad> sF(num_dims_);
            for (int k(0); k < num_dims_; ++k) {
              for (int l(0); l < num_dims_; ++l ) {
                sF(k,l) = factor * s(k,l);
              }
            }

            // shear dependent term for void growth
            //
            Fad omega(0.0), taue(0.0), smag(0.0);
            Fad J3 = Intrepid::det(sF);
            Fad smag2 = Intrepid::dotdot(sF,sF);
            if ( smag2 > 0.0 ) {
              smag = std::sqrt(smag2);
              taue = sq32 * smag;
            }

            if ( taue > 0.0 ) {
              Fad taue3 = taue * taue * taue;
              Fad tmp = 27.0 * J3 / 2.0 / taue3;
              omega = 1.0 - tmp * tmp;
            }

            // compute the hardening residual
            //
            Fad deps_ssF = (H - Rd*eps_ssF) * dgamF;
            Fad eps_resF = eps_ssF - eps_ss_old - deps_ssF;

            // increment in equivalent plastic strain
            //
            Fad deq = dgamF * (q1_ * q2_ * pF * YbarF * fstarF * std::sinh(argF)) / (1.0 - fstarF) / YbarF;
            if (smag != 0.0) {
              deq += dgamF * smag2 / (1.0 - fstarF) / YbarF;
            }

            // void nucleation
            // 
            Fad eratio = -0.5 * ( eqpsF - eN_ ) * ( eqpsF - eN_ ) / sN_ / sN_;
            Fad Anuc = fN_ / sN_ / ( std::sqrt( 2.0 * pi ) ) * std::exp(eratio);
            Fad dfnuc = Anuc * deq;

            // void nucleation with H, He
            // 
            Fad Heratio = -0.5 * ( eps_ssF - H_mean_eps_ss ) * ( eps_ssF - H_mean_eps_ss ) / sHN_ / sHN_;
            Fad HAnuc = He_void_vol_frac_nuc / sHN_ / ( std::sqrt( 2.0 * pi ) ) * std::exp(Heratio);
            Fad dHfnuc = HAnuc * deps_ssF;

            // void growth
            //
            Fad dfg = dgamF * q1_ * q2_ * ( 1.0 - fstarF ) * fstarF * YbarF * std::sinh(argF);
            if ( taue > 0.0 ) {
              dfg += sq23 * dgamF * kw_ * fstarF * omega * smag;
            }

            // yield surface
            //
            Fad PhiF = 0.5 * smag2 - psiF * YbarF * YbarF / 3.0;

            // for convenience put the residuals into a container
            //
            RFad[0] = PhiF;
            RFad[1] = eps_resF;
            RFad[2] = (pF - p + dgamF * q1_ * q2_ * bulk * YbarF * fstarF * std::sinh(argF) ) / bulk;
            RFad[3] = void_volume_fractionF - void_volume_fraction_old - dfg - dfnuc - dHfnuc;
            RFad[4] = eqpsF - eqps_old - deq;

            // extract the values of the residuals
            //
            for (int i = 0; i < num_vars; ++i) {
              R[i] = RFad[i].val();
            }

            // compute the norm of the residual
            //
            // (ahh! this hurts my eyes!)
            RealType R0 = Sacado::ScalarValue<ScalarT>::eval(R[0]); 
            RealType R1 = Sacado::ScalarValue<ScalarT>::eval(R[1]);
            RealType R2 = Sacado::ScalarValue<ScalarT>::eval(R[2]);
            RealType R3 = Sacado::ScalarValue<ScalarT>::eval(R[3]);
            RealType R4 = Sacado::ScalarValue<ScalarT>::eval(R[4]);
            RealType norm_res = std::sqrt(R0*R0 + R1*R1 + R2*R2 + R3*R3 + R4*R4);
            max_norm = std::max(norm_res, max_norm);
            
            // check against too many iterations and failure
            //
            // if we have iterated the maximum number of times, just quit.
            // we are banking on the global (NOX/LOCA) solver strategy to detect
            // global convergence failure and cut back if necessary.
            // this is not ideal and needs more work.
            //
            if (iter == max_iter) {
              if (void_volume_fractionF.val() >= ff_) {
                failed = true;
              }
              break;
            }

            // check for a sufficiently small residual
            //
            if ( (norm_res/max_norm < 1.e-12) || (norm_res < 1.e-12) ) {
              converged = true;
              if(print_) std::cout << "!!!CONVERGED!!! in " << iter << " iterations" << std::endl;
            }

            // extract the sensitivities of the residuals
            //
            for (int i = 0; i < num_vars; ++i)
              for (int j = 0; j < num_vars; ++j)
                dRdX[i + num_vars * j] = RFad[i].dx(j);

            // this call invokes the solver and updates the solution in X
            //
            solver.solve(dRdX, X, R);

            // increment the iteration counter
            //
            iter++;
          }

          // patch local sensistivities into global 
          // (magic!)
          //
          solver.computeFadInfo(dRdX, X, R);
          
          // extract solution
          //
          ScalarT dgam = X[0];
          ScalarT eps_ss = X[1];
          ScalarT kappa = 2.0 * mubar * eps_ss;
          p = X[2];
          ScalarT void_volume_fraction = X[3];
          ScalarT eqps = X[4];

          // compute modified void volume fraction
          //
          fstar = compute_fstar(void_volume_fraction, fc_, ff_, q1_);

          // return mapping of stress state
          //
          s = (1.0 / (1.0 + 2.0 * mubar * dgam) ) * s;

          // mechanical source
          // FIXME this is not correct, just a placeholder
          //
          if (have_temperature_ && delta_time(0) > 0) {
            source_field(cell, pt) = (sq23 * dgam / delta_time(0))
              * (Y + kappa) / (density_ * heat_capacity_);
          }

          // exponential map to get Fpnew
          //
          Ybar = Je * (Y + kappa);
          arg = 1.5 * q2_ * p / Ybar;
          Intrepid::Tensor<ScalarT> dPhi = s + 1.0 / 3.0 * q1_ * q2_ * Ybar * fstar * std::sinh(arg) * I;
          Fpnew = Intrepid::exp(dgam * dPhi) * Fpn;
          for (std::size_t i(0); i < num_dims_; ++i) {
            for (std::size_t j(0); j < num_dims_; ++j) {
              Fp_field(cell, pt, i, j) = Fpnew(i, j);
            }
          }

          // update other plasticity state variables
          //
          eps_ss_field(cell, pt) = eps_ss;
          eqps_field(cell,pt) = eqps;
          kappa_field(cell,pt) = kappa;
          void_volume_fraction_field(cell,pt) = void_volume_fraction;

        } else {
          // we are not yielding, variables do not evolve
          //
          eps_ss_field(cell, pt) = eps_ss_old;
          eqps_field(cell,pt) = eqps_old;
          kappa_field(cell,pt) = kappa_old;
          void_volume_fraction_field(cell,pt) = void_volume_fraction_old;
          if (have_temperature_) source_field(cell, pt) = 0.0;
          for (std::size_t i(0); i < num_dims_; ++i) {
            for (std::size_t j(0); j < num_dims_; ++j) {
              Fp_field(cell, pt, i, j) = Fpn(i, j);
            }
          }
        }

        // compute stress
        //
        sigma = p / Je * I + s / Je;
        for (std::size_t i(0); i < num_dims_; ++i) {
          for (std::size_t j(0); j < num_dims_; ++j) {
            stress_field(cell, pt, i, j) = sigma(i, j);
          }
        }
      } else {  // this point has failed
        eps_ss_field(cell,pt) = eps_ss_field_old(cell,pt);
        eqps_field(cell,pt) = eqps_field_old(cell,pt);
        kappa_field(cell,pt) = kappa_field_old(cell,pt);
        if (have_temperature_) source_field(cell, pt) = 0.0;
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            Fp_field(cell,pt,i,j) = Fp_field_old(cell,pt,i,j);
            stress_field(cell,pt,i,j) = 0.0;
          }
        }
      }
    }
  }
  if (have_temperature_) {
    for (int cell(0); cell < workset.numCells; ++cell) {
      for (int pt(0); pt < num_pts_; ++pt) {
        F.fill(def_grad_field,cell,pt,0,0);
        ScalarT J = Intrepid::det(F);
        sigma.fill(stress_field,cell,pt,0,0);
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
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
template<typename T>
T ElastoViscoplasticModel<EvalT, Traits>::
compute_fstar( T f, double fcrit, double ffail, double q1 ) {
  T fstar = f;
  if ( ( f > fcrit ) && ( f < ffail ) ) {
    if ( ( ffail - fcrit ) != 0.0 ) {
      fstar = fcrit + ( f - fcrit ) * ( ( 1.0 / q1 ) - fcrit ) / ( ffail - fcrit );
    }
  } else if ( f >= ffail ) {
    fstar -= ( f - ( 1.0 / q1 ) );
  }
  
  if ( fstar > 1.0 ) fstar = 1.0;
  return fstar;
}
//------------------------------------------------------------------------------
}
