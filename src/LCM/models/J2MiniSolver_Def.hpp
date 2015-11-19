//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Intrepid_MiniTensor_Solvers.h"
#include "Intrepid_MiniTensor.h"
#include "MiniNonlinearSolver.h"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM
{

//
//
//
template<typename EvalT, typename Traits>
J2MiniSolver<EvalT, Traits>::
J2MiniSolver(Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
    sat_mod_(p->get<RealType>("Saturation Modulus", 0.0)),
    sat_exp_(p->get<RealType>("Saturation Exponent", 0.0))
{
  // retrive appropriate field name strings
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string = (*field_name_map_)["Fp"];
  std::string eqps_string = (*field_name_map_)["eqps"];
  std::string yieldSurface_string = (*field_name_map_)["Yield_Surface"];
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
      std::make_pair("Hardening Modulus", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Delta Time", dl->workset_scalar));

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(Fp_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(eqps_string, dl->qp_scalar));
  this->eval_field_map_.insert(
      std::make_pair(yieldSurface_string, dl->qp_scalar));
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
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Cauchy Stress", false));
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
  // yield surface
  this->num_state_variables_++;
  this->state_var_names_.push_back(yieldSurface_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Yield Surface", false));
  //
  // mechanical source
  if (have_temperature_) {
    this->num_state_variables_++;
    this->state_var_names_.push_back(source_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(false);
    this->state_var_output_flags_.push_back(
        p->get<bool>("Output Mechanical Source", false));
  }
}

//
// J2 nonlinear system
//
template<typename S>
class J2NLS: public Intrepid::Function_Base<J2NLS<S>, S>
{
public:
  J2NLS(
      RealType sat_mod_,
      RealType sat_exp_,
      RealType eqps_old_,
      S const & K,
      S const & smag,
      S const & mubar,
      S const & Y) :
      sat_mod(sat_mod_),
      sat_exp(sat_exp_),
      eqps_old(eqps_old_),
      K_(K),
      smag_(smag),
      mubar_(mubar),
      Y_(Y)
  {
  }

  static constexpr Intrepid::Index
  DIMENSION{1};

  static constexpr
  char const * const
  NAME{"J2 NLS"};

  // Default value.
  template<typename T, Intrepid::Index N>
  T
  value(Intrepid::Vector<T, N> const & x)
  {
    return Intrepid::Function_Base<J2NLS<S>, S>::value(*this, x);
  }

  // Explicit gradient.
  template<typename T, Intrepid::Index N>
  Intrepid::Vector<T, N>
  gradient(Intrepid::Vector<T, N> const & x)
  {
    // Firewalls.
    Intrepid::Index const
    dimension = x.get_dimension();

    assert(dimension == DIMENSION);

    // Variables that potentially have Albany::Traits sensitivity
    // information need to be handled by the peel functor so that
    // proper conversions take place.
    T const
    K = peel<S, T, N>()(K_);

    T const
    smag = peel<S, T, N>()(smag_);

    T const
    mubar = peel<S, T, N>()(mubar_);

    T const
    Y = peel<S, T, N>()(Y_);

    // This is the actual computation of the gradient.
    Intrepid::Vector<T, N>
    r(dimension);

    T const &
    X = x(0);

    T const
    alpha = eqps_old + sq23 * X;

    T const
    H = K * alpha + sat_mod * (1.0 - std::exp(-sat_exp * alpha));

    T const
    R = smag - (2.0 * mubar * X + sq23 * (Y + H));

    r(0) = R;

    return r;
  }

  // Default AD hessian.
  template<typename T, Intrepid::Index N>
  Intrepid::Tensor<T, N>
  hessian(Intrepid::Vector<T, N> const & x)
  {
    return Intrepid::Function_Base<J2NLS<S>, S>::hessian(*this, x);
  }

  // Constants.
  RealType const
  sq23{std::sqrt(2.0 / 3.0)};

  // RealType data (fixed non-AD type)
  RealType const
  sat_mod{0.0};

  RealType const
  sat_exp{0.0};

  RealType const
  eqps_old{0.0};

  // Inputs
  S const &
  K_;

  S const &
  smag_;

  S const &
  mubar_;

  S const &
  Y_;
};

//
//
//
template<typename EvalT, typename Traits>
void J2MiniSolver<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>>dep_fields,
std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields)
{
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string = (*field_name_map_)["Fp"];
  std::string eqps_string = (*field_name_map_)["eqps"];
  std::string yieldSurface_string = (*field_name_map_)["Yield_Surface"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];

  // extract dependent MDFields
  PHX::MDField<ScalarT> def_grad = *dep_fields[F_string];
  PHX::MDField<ScalarT> J = *dep_fields[J_string];
  PHX::MDField<ScalarT> poissons_ratio = *dep_fields["Poissons Ratio"];
  PHX::MDField<ScalarT> elastic_modulus = *dep_fields["Elastic Modulus"];
  PHX::MDField<ScalarT> yieldStrength = *dep_fields["Yield Strength"];
  PHX::MDField<ScalarT> hardeningModulus = *dep_fields["Hardening Modulus"];
  PHX::MDField<ScalarT> delta_time = *dep_fields["Delta Time"];

  // extract evaluated MDFields
  PHX::MDField<ScalarT> stress = *eval_fields[cauchy_string];
  PHX::MDField<ScalarT> Fp = *eval_fields[Fp_string];
  PHX::MDField<ScalarT> eqps = *eval_fields[eqps_string];
  PHX::MDField<ScalarT> yieldSurf = *eval_fields[yieldSurface_string];
  PHX::MDField<ScalarT> source;
  if (have_temperature_) {
    source = *eval_fields[source_string];
  }

  // get State Variables
  Albany::MDArray Fpold = (*workset.stateArrayPtr)[Fp_string + "_old"];
  Albany::MDArray eqpsold = (*workset.stateArrayPtr)[eqps_string + "_old"];

  ScalarT kappa, mu, mubar, K, Y;
  ScalarT Jm23, trace, smag2, smag, f, p, dgam;
  ScalarT sq23(std::sqrt(2. / 3.));
  ScalarT H {0.0};

  Intrepid::Tensor<ScalarT>
  F(num_dims_), be(num_dims_), s(num_dims_), sigma(num_dims_);

  Intrepid::Tensor<ScalarT>
  N(num_dims_), A(num_dims_), expA(num_dims_), Fpnew(num_dims_);

  Intrepid::Tensor<ScalarT>
  I(Intrepid::eye<ScalarT>(num_dims_));

  Intrepid::Tensor<ScalarT>
  Fpn(num_dims_), Fpinv(num_dims_), Cpinv(num_dims_);

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      kappa = elastic_modulus(cell, pt)
      / (3. * (1. - 2. * poissons_ratio(cell, pt)));
      mu = elastic_modulus(cell, pt) / (2. * (1. + poissons_ratio(cell, pt)));
      K = hardeningModulus(cell, pt);
      Y = yieldStrength(cell, pt);
      Jm23 = std::pow(J(cell, pt), -2. / 3.);
      // fill local tensors
      F.fill(def_grad,cell, pt,0,0);
      //Fpn.fill( &Fpold(cell,pt,int(0),int(0)) );
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          Fpn(i, j) = ScalarT(Fpold(cell, pt, i, j));
        }
      }

      // compute trial state
      Fpinv = Intrepid::inverse(Fpn);

      Cpinv = Fpinv * Intrepid::transpose(Fpinv);
      be = Jm23 * F * Cpinv * Intrepid::transpose(F);
      s = mu * Intrepid::dev(be);

      mubar = Intrepid::trace(be) * mu / (num_dims_);

      // check yield condition
      smag = Intrepid::norm(s);
      f = smag - sq23 * (Y + K * eqpsold(cell, pt)
      + sat_mod_ * (1. - std::exp(-sat_exp_ * eqpsold(cell, pt))));

      if (f > 1E-12) {
        // Use minimization equivalent to return mapping
        using ValueT = typename Sacado::ValueType<ScalarT>::type;

        J2NLS<ScalarT>
        j2nls(sat_mod_, sat_exp_, eqpsold(cell, pt), K, smag, mubar, Y);

        constexpr
        Intrepid::Index
        dimension{J2NLS<ScalarT>::DIMENSION};

        Intrepid::NewtonStep<ValueT, dimension>
        step;

        Intrepid::Minimizer<ValueT, dimension>
        minimizer;

        Intrepid::Vector<ScalarT, dimension>
        x;

        x(0) = 0.0;

        miniMinimize(minimizer, step, j2nls, x);

        ScalarT const
        alpha = eqpsold(cell, pt) + sq23 * x(0);

        ScalarT const
        H = K * alpha + sat_mod_ * (1.0 - exp(-sat_exp_ * alpha));

        dgam = x(0);

        // plastic direction
        N = (1 / smag) * s;

        // update s
        s -= 2 * mubar * dgam * N;

        // update eqps
        eqps(cell, pt) = alpha;

        // mechanical source
        if (have_temperature_ && delta_time(0) > 0) {
          source(cell, pt) = (sq23 * dgam / delta_time(0)
          * (Y + H + temperature_(cell,pt))) / (density_ * heat_capacity_);
        }

        // exponential map to get Fpnew
        A = dgam * N;
        expA = Intrepid::exp(A);
        Fpnew = expA * Fpn;
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            Fp(cell, pt, i, j) = Fpnew(i, j);
          }
        }
      } else {
        eqps(cell, pt) = eqpsold(cell, pt);
        if (have_temperature_) source(cell, pt) = 0.0;
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            Fp(cell, pt, i, j) = Fpn(i, j);
          }
        }
      }

      // update yield surface
      yieldSurf(cell, pt) = Y + K * eqps(cell, pt)
      + sat_mod_ * (1. - std::exp(-sat_exp_ * eqps(cell, pt)));

      // compute pressure
      p = 0.5 * kappa * (J(cell, pt) - 1. / (J(cell, pt)));

      // compute stress
      sigma = p * I + s / J(cell, pt);
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          stress(cell, pt, i, j) = sigma(i, j);
        }
      }
    }
  }

  if (have_temperature_) {
    for (int cell(0); cell < workset.numCells; ++cell) {
      for (int pt(0); pt < num_pts_; ++pt) {
        F.fill(def_grad,cell,pt,0,0);
        ScalarT J = Intrepid::det(F);
        sigma.fill(stress,cell,pt,0,0);
        sigma -= 3.0 * expansion_coeff_ * (1.0 + 1.0 / (J*J))
        * (temperature_(cell,pt) - ref_temperature_) * I;
        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            stress(cell, pt, i, j) = sigma(i, j);
          }
        }
      }
    }
  }
}
#ifdef ALBANY_ENSEMBLE
template<>
void J2MiniSolver<PHAL::AlbanyTraits::MPResidual, PHAL::AlbanyTraits>::
computeState(typename PHAL::AlbanyTraits::EvalData workset,
std::map<std::string, Teuchos::RCP<PHX::MDField<PHAL::AlbanyTraits::MPResidual::ScalarT>>> dep_fields,
std::map<std::string, Teuchos::RCP<PHX::MDField<PHAL::AlbanyTraits::MPResidual::ScalarT>>> eval_fields)
{
  assert(0);
}
template<>
void J2MiniSolver<PHAL::AlbanyTraits::MPJacobian, PHAL::AlbanyTraits>::
computeState(typename PHAL::AlbanyTraits::EvalData workset,
std::map<std::string, Teuchos::RCP<PHX::MDField<PHAL::AlbanyTraits::MPJacobian::ScalarT>>> dep_fields,
std::map<std::string, Teuchos::RCP<PHX::MDField<PHAL::AlbanyTraits::MPJacobian::ScalarT>>> eval_fields)
{
  assert(0);
}
template<>
void J2MiniSolver<PHAL::AlbanyTraits::MPTangent, PHAL::AlbanyTraits>::
computeState(typename PHAL::AlbanyTraits::EvalData workset,
std::map<std::string, Teuchos::RCP<PHX::MDField<PHAL::AlbanyTraits::MPTangent::ScalarT>>> dep_fields,
std::map<std::string, Teuchos::RCP<PHX::MDField<PHAL::AlbanyTraits::MPTangent::ScalarT>>> eval_fields)
{
  assert(0);
}
#endif
// computeState parallel function, which calls Kokkos::parallel_for
template<typename EvalT, typename Traits>
void J2MiniSolver<EvalT, Traits>::
computeStateParallel(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>>dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT>>> eval_fields)
    {
    }

}

