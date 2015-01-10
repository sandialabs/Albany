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
J2Model<EvalT, Traits>::
J2Model(Teuchos::ParameterList* p,
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
  this->eval_field_map_.insert(std::make_pair(yieldSurface_string, dl->qp_scalar));
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
  // yield surface
  this->num_state_variables_++;
  this->state_var_names_.push_back(yieldSurface_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Yield Surface", false));
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
void J2Model<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
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

  Intrepid::Tensor<ScalarT> F(num_dims_), be(num_dims_), s(num_dims_), sigma(
      num_dims_);
  Intrepid::Tensor<ScalarT> N(num_dims_), A(num_dims_), expA(num_dims_), Fpnew(
      num_dims_);
  Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));
  Intrepid::Tensor<ScalarT> Fpn(num_dims_), Fpinv(num_dims_), Cpinv(num_dims_);

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
        dFdX[0] = (-2. * mubar) * (1. + H / (3. * mubar));
        while (!converged && count <= 30)
        {
          count++;
          solver.solve(dFdX, X, F);
          alpha = eqpsold(cell, pt) + sq23 * X[0];
          H = K * alpha + sat_mod_ * (1. - exp(-sat_exp_ * alpha));
          dH = K + sat_exp_ * sat_mod_ * exp(-sat_exp_ * alpha);
          F[0] = smag - (2. * mubar * X[0] + sq23 * (Y + H));
          dFdX[0] = -2. * mubar * (1. + dH / (3. * mubar));

          res = std::abs(F[0]);
          if (res < 1.e-11 || res / Y < 1.E-11)
            converged = true;

          TEUCHOS_TEST_FOR_EXCEPTION(count == 30, std::runtime_error,
              std::endl <<
              "Error in return mapping, count = " <<
              count <<
              "\nres = " << res <<
              "\nrelres = " << res/f <<
              "\ng = " << F[0] <<
              "\ndg = " << dFdX[0] <<
              "\nalpha = " << alpha << std::endl);
        }
        solver.computeFadInfo(dFdX, X, F);
        dgam = X[0];

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
//------------------------------------------------------------------------------
#ifndef NO_KOKKOS_ALBANY

// computeState Kokkos functor
//template<typename EvalT, typename Traits>
template <class Array1, class Array2>
KOKKOS_INLINE_FUNCTION
//void J2Model<EvalT, Traits>::
void inverse(const Array1 &A, Array2  &Atrans) 
{
  int num_dims_=A.dimension(0);
  for (int i(0); i < num_dims_; ++i) 
        for (int j(0); j < num_dims_; ++j) 
            Atrans(i,j)=A(j,i);
}

template<typename EvalT, typename Traits>
template <class ArrayT>
KOKKOS_INLINE_FUNCTION
const typename J2Model<EvalT, Traits>::ScalarT
J2Model<EvalT, Traits>::
trace (const ArrayT &A) const
{

  ScalarT s = 0.0;

  switch (num_dims_) {

    default:
      for (int i = 0; i < num_dims_; ++i) {
        s += A(i,i);
      }
      break;

    case 3:
      s = A(0,0) + A(1,1) + A(2,2);
      break;

    case 2:
      s = A(0,0) + A(1,1);
      break;

  }

  return s;
}


template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void J2Model<EvalT, Traits>::computeStateKernel::
compute_common(const int cell) const{

  ScalarT kappa, mu, mubar, K, Y;
  ScalarT Jm23, trace, smag2, smag, f, p, dgam;
 
//  Kokkos::View <ScalarT**, PHX::Device> F ("F", dims_, dims_, derivative_dim);
//  Kokkos::View <ScalarT**, PHX::Device> Fpn ("Fpn", dims_, dims_, derivative_dim);
//  Kokkos::View <ScalarT**, PHX::Device> Fpinv ("Fpinv", dims_, dims_, derivative_dim);
//  Kokkos::View <ScalarT**, PHX::Device> Cpinv ("Cpinv", dims_, dims_, derivative_dim);
//  Kokkos::View <ScalarT**, PHX::Device> be ("be", dims_, dims_, derivative_dim);

  for (int pt(0); pt < num_pts; ++pt) {
      kappa = elastic_modulus(cell, pt)
          / (3. * (1. - 2. * poissons_ratio(cell, pt)));
      mu = elastic_modulus(cell, pt) / (2. * (1. + poissons_ratio(cell, pt)));
      K = hardeningModulus(cell, pt);
      Y = yieldStrength(cell, pt);
      Jm23 = std::pow(J(cell, pt), -2. / 3.);

      // fill local tensors
      for (int i(0); i < dims_; ++i) {
        for (int j(0); j < dims_; ++j) {
          F(i,j) = ScalarT(def_grad(cell, pt, i, j));
          Fpn(i, j) = ScalarT(Fpold(cell, pt, i, j));
        }
       }
      
     inverse(Fpn, Fpinv); 

     for (int i(0); i < dims_; ++i) {
        for (int j(0); j < dims_; ++j) {  
          Cpinv(i,j) = Fpinv(i,j) * Fpinv(j,i);
          be(i,j) = Jm23 * F(i,j) * Cpinv(i,j) * F(j,i);
   //   s = mu * Intrepid::dev(be);
  //    mubar = Intrepid::trace(be) * mu / (num_dims_);
        }
     }
  }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void J2Model<EvalT, Traits>::computeStateKernel::
compute_with_temperature(const int cell) const{

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void J2Model<EvalT, Traits>::computeStateKernel::
compute_with_no_temperature(const int cell) const{

}

template<typename EvalT, typename Traits> 
KOKKOS_INLINE_FUNCTION
void J2Model<EvalT, Traits>::computeStateKernel::
operator() (const have_temperature_Tag& tag, const int i) const
{
  compute_common(i);
  compute_with_temperature(i);
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void J2Model<EvalT, Traits>::computeStateKernel::
operator() (const dont_have_temperature_Tag& tag, const int i) const
{
  compute_common(i);
  compute_with_no_temperature(i);
}
// computeState parallel function, which calls Kokkos::parallel_for
template<typename EvalT, typename Traits>
void J2Model<EvalT, Traits>::
computeStateParallel(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
{

  //const int derivative_dim=25;
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

  //temporary data:
  
  typedef PHX::KokkosViewFactory<ScalarT,PHX::Device> ViewFactory;
  std::vector<PHX::index_size_type> ddims_;
  ddims_.push_back(24);

  PHX::MDField<ScalarT, Dim, Dim> F;
  PHX::MDField<ScalarT, Dim, Dim> be;
  PHX::MDField<ScalarT, Dim, Dim> s;
  PHX::MDField<ScalarT, Dim, Dim> sigma;
  PHX::MDField<ScalarT, Dim, Dim> N;
  PHX::MDField<ScalarT, Dim, Dim> A;
  PHX::MDField<ScalarT, Dim, Dim> expA; 
  PHX::MDField<ScalarT, Dim, Dim> Fpnew;
  PHX::MDField<ScalarT, Dim, Dim> I;
  PHX::MDField<ScalarT, Dim, Dim> Fpn;
  PHX::MDField<ScalarT, Dim, Dim> Fpinv;
  PHX::MDField<ScalarT, Dim, Dim> Cpinv;

  F     = PHX::MDField<ScalarT, Dim, Dim>("F",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(num_dims_,num_dims_)));
  s    = PHX::MDField<ScalarT, Dim, Dim>("s",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(num_dims_,num_dims_)));
  be    = PHX::MDField<ScalarT, Dim, Dim>("be",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(num_dims_,num_dims_)));
  sigma = PHX::MDField<ScalarT, Dim, Dim>("sigma",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(num_dims_,num_dims_)));
  N     = PHX::MDField<ScalarT, Dim, Dim>("N",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(num_dims_,num_dims_)));
  A     = PHX::MDField<ScalarT, Dim, Dim>("A",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(num_dims_,num_dims_)));
  expA  = PHX::MDField<ScalarT, Dim, Dim>("expA",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(num_dims_,num_dims_)));
  Fpnew = PHX::MDField<ScalarT, Dim, Dim>("Fpnew",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(num_dims_,num_dims_)));
  I     = PHX::MDField<ScalarT, Dim, Dim>("I",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(num_dims_,num_dims_)));
  Fpn   = PHX::MDField<ScalarT, Dim, Dim>("Fpn",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(num_dims_,num_dims_)));
  Fpinv = PHX::MDField<ScalarT, Dim, Dim>("Fpinv",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(num_dims_,num_dims_)));
  Cpinv = PHX::MDField<ScalarT, Dim, Dim>("Cpinv",Teuchos::rcp(new PHX::MDALayout<Dim,Dim>(num_dims_,num_dims_)));

  F.setFieldData(ViewFactory::buildView(F.fieldTag(),ddims_));
  be.setFieldData(ViewFactory::buildView(be.fieldTag(),ddims_));
  s.setFieldData(ViewFactory::buildView(s.fieldTag(),ddims_));
  sigma.setFieldData(ViewFactory::buildView(sigma.fieldTag(),ddims_));
  N.setFieldData(ViewFactory::buildView(N.fieldTag(),ddims_));
  A.setFieldData(ViewFactory::buildView(A.fieldTag(),ddims_));
  expA.setFieldData(ViewFactory::buildView(expA.fieldTag(),ddims_));
  Fpnew.setFieldData(ViewFactory::buildView(Fpnew.fieldTag(),ddims_));
  I.setFieldData(ViewFactory::buildView(I.fieldTag(),ddims_));
  Fpn.setFieldData(ViewFactory::buildView(Fpn.fieldTag(),ddims_));
  Fpinv.setFieldData(ViewFactory::buildView(Fpinv.fieldTag(),ddims_));
  Cpinv.setFieldData(ViewFactory::buildView(Cpinv.fieldTag(),ddims_));

  for (int i=0; i<num_dims_; i++){
     for (int j=0; j<num_dims_;j++){
        I(i,j)=ScalarT(0.0);
        if (j==j)I(i,j)=ScalarT(1.0);
     }
    }

   typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  computeStateKernel Kernel(num_dims_, num_pts_, def_grad, J, poissons_ratio, elastic_modulus, yieldStrength, hardeningModulus, delta_time, stress, Fp, eqps, yieldSurf, source, Fpold, eqpsold, have_temperature_, F, be, s, sigma, N, A, expA, Fpnew, I, Fpn, Fpinv, Cpinv);

  if (have_temperature_)
     Kokkos::parallel_for(have_temperature_Policy(0,workset.numCells),Kernel);
  else
     Kokkos::parallel_for(dont_have_temperature_Policy(0,workset.numCells),Kernel);

}
#endif
//-------------------------------------------------------------------------------
}

