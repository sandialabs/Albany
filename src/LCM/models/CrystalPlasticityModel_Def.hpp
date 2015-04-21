//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Utils.hpp"
#include <boost/math/special_functions/fpclassify.hpp>

//#define  PRINT_DEBUG
//#define  PRINT_OUTPUT
//#define  DECOUPLE

#include <typeinfo>
#include <iostream>
#include <Sacado_Traits.hpp>
namespace LCM
{

template<typename EvalT, typename Traits>
CrystalPlasticityModel<EvalT, Traits>::
CrystalPlasticityModel(Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
    num_slip_(p->get<int>("Number of Slip Systems", 0))
{
  slip_systems_.resize(num_slip_);

#ifdef PRINT_DEBUG
  std::cout << ">>> in cp constructor\n";
  std::cout << ">>> parameter list:\n" << *p << std::endl;
#endif

  Teuchos::ParameterList e_list = p->sublist("Crystal Elasticity");
  // assuming cubic symmetry
  c11_ = e_list.get<RealType>("C11");
  c12_ = e_list.get<RealType>("C12");
  c44_ = e_list.get<RealType>("C44");
  Intrepid::Tensor4<RealType> C(num_dims_);
  C.fill(Intrepid::ZEROS);
  for (int i = 0; i < num_dims_; ++i) {
    C(i, i, i, i) = c11_;
    for (int j = i + 1; j < num_dims_; ++j) {
      C(i, i, j, j) = C(j, j, i, i) = c12_;
      C(i, j, i, j) = C(j, i, j, i) = C(i, j, j, i) = C(j, i, i, j) = c44_;
    }
  }
// NOTE check if basis is given else default
// NOTE default to coordinate axes and also construct 3rd direction if only 2 given
  orientation_.set_dimension(num_dims_);
  for (int i = 0; i < num_dims_; ++i) {
    std::vector < RealType > b_temp = e_list.get<Teuchos::Array<RealType> >(
        Albany::strint("Basis Vector", i + 1)).toVector();
    RealType norm = 0.;
    for (int j = 0; j < num_dims_; ++j) {
      norm += b_temp[j] * b_temp[j];
    }
// NOTE check zero, rh system
// Filling columns of transformation with basis vectors
// We are forming R^{T} which is equivalent to the direction cosine matrix
    norm = 1. / std::sqrt(norm);
    for (int j = 0; j < num_dims_; ++j) {
      orientation_(j, i) = b_temp[j] * norm;
    }
  }

// print rotation tensor employed for transformations
#ifdef PRINT_DEBUG
  std::cout << ">>> orientation_ :\n" << orientation_ << std::endl;
#endif

  // rotate elastic tensor and slip systems to match given orientation
  C_ = Intrepid::kronecker(orientation_, C);
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    Teuchos::ParameterList ss_list = p->sublist(
        Albany::strint("Slip System", num_ss + 1));

    // Obtain and normalize slip directions. Miller indices need to be normalized.
    std::vector < RealType > s_temp = ss_list.get<Teuchos::Array<RealType> >(
        "Slip Direction").toVector();
    Intrepid::Vector<RealType> s_temp_normalized(num_dims_, &s_temp[0]);
    s_temp_normalized = Intrepid::unit(s_temp_normalized);
    slip_systems_[num_ss].s_ = orientation_ * s_temp_normalized;

    // Obtain and normal slip normals. Miller indices need to be normalized.
    std::vector < RealType > n_temp = ss_list.get<Teuchos::Array<RealType> >(
        "Slip Normal").toVector();
    Intrepid::Vector<RealType> n_temp_normalized(num_dims_, &n_temp[0]);
    n_temp_normalized = Intrepid::unit(n_temp_normalized);
    slip_systems_[num_ss].n_ = orientation_ * n_temp_normalized;

    // print each slip direction and slip normal after transformation
#ifdef PRINT_DEBUG
    std::cout << ">>> slip direction " << num_ss + 1 << ": " << slip_systems_[num_ss].s_ << std::endl;
    std::cout << ">>> slip normal " << num_ss + 1 << ": " << slip_systems_[num_ss].n_ << std::endl;
#endif

    slip_systems_[num_ss].projector_ = Intrepid::dyad(
        slip_systems_[num_ss].s_,
        slip_systems_[num_ss].n_);

    // print projector
#ifdef PRINT_DEBUG
    std::cout << ">>> projector_ " << num_ss + 1 << ": " << slip_systems_[num_ss].projector_ << std::endl;
#endif

    slip_systems_[num_ss].tau_critical_ = ss_list.get<RealType>("Tau Critical");
    slip_systems_[num_ss].gamma_dot_0_ = ss_list.get<RealType>("Gamma Dot");
    slip_systems_[num_ss].gamma_exp_ = ss_list.get<RealType>("Gamma Exponent");
    slip_systems_[num_ss].H_ = ss_list.get<RealType>("Hardening", 0);
  }
#ifdef PRINT_DEBUG
  std::cout << "<<< done with parameter list\n";
#endif

  // retrive appropriate field name strings (ref to problems/FieldNameMap)
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string = (*field_name_map_)["Fp"];
  std::string L_string = (*field_name_map_)["Velocity_Gradient"];
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];

  // define the dependent fields
  // required for calculation
  this->dep_field_map_.insert(std::make_pair(F_string, dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair(J_string, dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Delta Time", dl->workset_scalar));

  // define the evaluated fields
  // optional output
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(Fp_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(L_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(source_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Time", dl->workset_scalar));

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
  // L
  this->num_state_variables_++;
  this->state_var_names_.push_back(L_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("identity");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output L", false));
  //
  // mechanical source (body force)
  this->num_state_variables_++;
  this->state_var_names_.push_back(source_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Mechanical Source", false));
  //
  // gammas
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    std::string g = Albany::strint("gamma", num_ss + 1, '_');
    std::string gamma_string = (*field_name_map_)[g];
    std::string output_gamma_string = "Output " + gamma_string;
    this->eval_field_map_.insert(std::make_pair(gamma_string, dl->qp_scalar));
    this->num_state_variables_++;
    this->state_var_names_.push_back(gamma_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(
        p->get<bool>(output_gamma_string, false));
  }
  //
  // taus
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    std::string t = Albany::strint("tau", num_ss + 1, '_');
    std::string tau_string = (*field_name_map_)[t];
    std::string output_tau_string = "Output " + tau_string;
    this->eval_field_map_.insert(std::make_pair(tau_string, dl->qp_scalar));
    this->num_state_variables_++;
    this->state_var_names_.push_back(tau_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(
        p->get<bool>(output_tau_string, false));
  }

#ifdef PRINT_DEBUG
  std::cout << "<<< done in cp constructor\n";
#endif
}

//------------------------------------------------------------------------------

template<typename EvalT, typename Traits>
void CrystalPlasticityModel<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
{
#ifdef PRINT_DEBUG
  std::cout << ">>> in cp compute state\n";
#endif
  // retrive appropriate field name strings
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string = (*field_name_map_)["Fp"];
  std::string L_string = (*field_name_map_)["Velocity_Gradient"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];

  // extract dependent MDFields
  PHX::MDField<ScalarT> def_grad = *dep_fields[F_string];
  PHX::MDField<ScalarT> J = *dep_fields[J_string];
  PHX::MDField<ScalarT> delta_time = *dep_fields["Delta Time"];

  // extract evaluated MDFields
  PHX::MDField<ScalarT> stress = *eval_fields[cauchy_string];
  PHX::MDField<ScalarT> plastic_deformation = *eval_fields[Fp_string];
  PHX::MDField<ScalarT> velocity_gradient = *eval_fields[L_string];
  PHX::MDField<ScalarT> source = *eval_fields[source_string];
  PHX::MDField<ScalarT> time = *eval_fields["Time"];

  // extract slip on each slip system
  std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > slips;
  std::vector<Albany::MDArray *> previous_slips;
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    std::string g = Albany::strint("gamma", num_ss + 1, '_');
    std::string gamma_string = (*field_name_map_)[g];
    slips.push_back(eval_fields[gamma_string]);
    previous_slips.push_back(
        &((*workset.stateArrayPtr)[gamma_string + "_old"]));
  }

  // extract shear on each slip system
  std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > shears;
  std::vector<Albany::MDArray *> previous_shears;
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    std::string t = Albany::strint("tau", num_ss + 1, '_');
    std::string tau_string = (*field_name_map_)[t];
    shears.push_back(eval_fields[tau_string]);
    previous_shears.push_back(
        &((*workset.stateArrayPtr)[tau_string + "_old"]));
  }

  // get state variables
  Albany::MDArray previous_plastic_deformation =
      (*workset.stateArrayPtr)[Fp_string + "_old"];

  ScalarT tau, gamma, dgamma;
  ScalarT dt = delta_time(0);
  ScalarT tcurrent = time(0);
  Intrepid::Tensor<ScalarT> L(num_dims_);
  Intrepid::Tensor<ScalarT> F(num_dims_), Fp(num_dims_);
  Intrepid::Tensor<ScalarT> sigma(num_dims_), S(num_dims_);



  I_ = Intrepid::eye<RealType>(num_dims_);

#ifdef PRINT_OUTPUT
  std::ofstream out("output.dat", std::fstream::app);
#endif

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {

      // fill local tensors
      F.fill(def_grad, cell, pt, 0, 0);
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          Fp(i, j) = ScalarT(previous_plastic_deformation(cell, pt, i, j));
        }
      }

      // TODO get rid of cell and pt arguments and just pass a reference to the correct entries in
      //      slips and previous_slips (assuming this is possible with the Intrepid data structures).
      predictor(cell, pt, dt, slips, previous_slips, F, L, Fp);

      // IMPLICIT LOOP GOES HERE

      computeStress(F, Fp, sigma, S);

      // Project new stress onto each system for postprocessing
      Intrepid::Tensor<RealType> P(num_dims_);
      PHX::MDField<ScalarT> shear;
      for (int s(0); s < num_slip_; ++s) {
          P = slip_systems_[s].projector_;
          tau = Intrepid::dotdot(P, S);
          shear = *(shears[s]);
          shear(cell,pt) = tau;
      }

      // Load results into Albany data containers
      source(cell, pt) = 0.0;
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          plastic_deformation(cell, pt, i, j) = Fp(i, j);
          stress(cell, pt, i, j) = sigma(i, j);
          velocity_gradient(cell, pt, i, j) = L(i, j);
        }
      }

#ifdef PRINT_OUTPUT
      if (cell == 0 && pt == 0) {
        out << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(tcurrent) << " ";
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            out << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(F(i,j)) << " ";
          }
        }
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            out << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(Fp(i,j)) << " ";
          }
        }
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            out << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(sigma(i,j)) << " ";
          }
        }
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            out << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(L(i,j)) << " ";
          }
        }
        out << "\n";
      }
#endif

    }
  }
#ifdef PRINT_DEBUG
  std::cout << "<<< done in cp compute state\n" << std::flush;
#endif
}

//------------------------------------------------------------------------------

template<typename EvalT, typename Traits>
void CrystalPlasticityModel<EvalT, Traits>::
predictor(int cell,
    int pt,
    ScalarT dt,
    std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > & slips,
    std::vector<Albany::MDArray *> const & previous_slips,
    Intrepid::Tensor<ScalarT> const & F,
    Intrepid::Tensor<ScalarT> & L,
    Intrepid::Tensor<ScalarT> & Fp)
{
  ScalarT g0, tau, tauC, gamma, dgamma, m, H, t1;
  Intrepid::Tensor<RealType> P(num_dims_);
  Intrepid::Tensor<ScalarT> sigma(num_dims_), S(num_dims_), expL(num_dims_),
      Fp_temp(num_dims_);
  PHX::MDField<ScalarT> slip;
  Albany::MDArray previous_slip;

  computeStress(F, Fp, sigma, S);

  confirmTensorSanity(sigma, "first sigma calculation in predictor()");

  L.fill(Intrepid::ZEROS);
  for (int s(0); s < num_slip_; ++s) {
    P = slip_systems_[s].projector_;

    // compute resolved shear stresses
    tau = Intrepid::dotdot(P, S);
    int sign = tau < 0 ? -1 : 1;

    // compute  dgammas
    g0 = slip_systems_[s].gamma_dot_0_;
    tauC = slip_systems_[s].tau_critical_;
    m = slip_systems_[s].gamma_exp_;
    H = slip_systems_[s].H_;

    slip = *(slips[s]);
    previous_slip = *(previous_slips[s]);
    slip(cell, pt) = previous_slip(cell, pt);
    gamma = previous_slip(cell, pt);

    t1 = std::fabs(tau / (tauC + H * std::fabs(gamma)));
    dgamma = dt * g0 * std::fabs(std::pow(t1, m)) * sign;

    slip(cell, pt) += dgamma;

    L += (dgamma * P);
  }

  confirmTensorSanity(L, "L in predictor().");

  // update plastic deformation gradient
  expL = Intrepid::exp(L);
  Fp_temp = expL * Fp;
  Fp = Fp_temp;

  confirmTensorSanity(Fp, "Fp in predictor()");
}

//------------------------------------------------------------------------------

template<typename EvalT, typename Traits>
void CrystalPlasticityModel<EvalT, Traits>::
computeStress(Intrepid::Tensor<ScalarT> const & F,
    Intrepid::Tensor<ScalarT> const & Fp,
    Intrepid::Tensor<ScalarT> & sigma,
    Intrepid::Tensor<ScalarT> & S)

{
  // Saint Venant–Kirchhoff model
  Fpinv_ = Intrepid::inverse(Fp);
#ifdef DECOUPLE
  std::cout << "ELASTIC STRESS ONLY\n";
  Fe_ = F;
#else
  Fe_ = F * Fpinv_;
#endif
  E_ = 0.5 * (Intrepid::transpose(Fe_) * Fe_ - I_);
  S = Intrepid::dotdot(C_, E_);
  sigma = (1.0 / Intrepid::det(F)) * F * S * Intrepid::transpose(F);
}

//------------------------------------------------------------------------------

template<typename EvalT, typename Traits>
void CrystalPlasticityModel<EvalT, Traits>::
confirmTensorSanity(Intrepid::Tensor<ScalarT> const & input,
    std::string const & message)
{
  int dim = input.get_dimension();
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      if (!boost::math::isfinite(input(i, j))) {
        std::string msg =
            "**** Invalid data detected in CrystalPlasticityModel::confirmTensorSanity(): "
                + message;
        TEUCHOS_TEST_FOR_EXCEPTION(
            !boost::math::isfinite(input(i, j)),
            std::logic_error,
            msg);
      }
    }
  }
}

} // namespace LCM
