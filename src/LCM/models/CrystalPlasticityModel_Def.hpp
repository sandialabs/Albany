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
  std::string eqps_string = (*field_name_map_)["eqps"];
  std::string Re_string = (*field_name_map_)["Re"];
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string = (*field_name_map_)["Fp"];
  std::string L_string = (*field_name_map_)["Velocity_Gradient"];
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];
  std::string residual_string = (*field_name_map_)["CP_Residual"];

  // define the dependent fields
  // required for calculation
  this->dep_field_map_.insert(std::make_pair(F_string, dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair(J_string, dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Delta Time", dl->workset_scalar));

  // define the evaluated fields
  // optional output
  this->eval_field_map_.insert(std::make_pair(eqps_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(Re_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(Fp_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(L_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(source_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair(residual_string, dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Time", dl->workset_scalar));

  // define the state variables
  //
  // eqps
  this->num_state_variables_++;
  this->state_var_names_.push_back(eqps_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output EQPS", false));
  //
  // Re
  this->num_state_variables_++;
  this->state_var_names_.push_back(Re_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("identity");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Re", false));
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
  // gammas for each slip system
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
  // tau_hard - state variable for hardening on each slip system
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    std::string t_h = Albany::strint("tau_hard", num_ss + 1, '_');
    std::string tau_hard_string = (*field_name_map_)[t_h];
    std::string output_tau_hard_string = "Output " + tau_hard_string;
    this->eval_field_map_.insert(
        std::make_pair(tau_hard_string, dl->qp_scalar));
    this->num_state_variables_++;
    this->state_var_names_.push_back(tau_hard_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(
        p->get<bool>(output_tau_hard_string, false));
  }
  //
  // taus - output resolved shear stress for debugging - not stated
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
    this->state_var_old_state_flags_.push_back(false);
    this->state_var_output_flags_.push_back(
        p->get<bool>(output_tau_string, false));
  }
  // residual
  this->num_state_variables_++;
  this->state_var_names_.push_back(residual_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(
        p->get<bool>("Output CP_Residual", false));

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

bool print_debug = false;
#ifdef PRINT_DEBUG
  if (typeid(ScalarT) == typeid(RealType)) {
    print_debug = true;
  }
  std::cout.precision(15);
#endif

#ifdef PRINT_DEBUG
  std::cout << ">>> in cp compute state\n";
#endif
  // retrive appropriate field name strings
  std::string eqps_string = (*field_name_map_)["eqps"];
  std::string Re_string = (*field_name_map_)["Re"];
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string = (*field_name_map_)["Fp"];
  std::string L_string = (*field_name_map_)["Velocity_Gradient"];
  std::string residual_string = (*field_name_map_)["CP_Residual"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];
  std::string F_string = (*field_name_map_)["F"];
  std::string J_string = (*field_name_map_)["J"];

  // extract dependent MDFields
  PHX::MDField<ScalarT> def_grad = *dep_fields[F_string];
  PHX::MDField<ScalarT> J = *dep_fields[J_string];
  PHX::MDField<ScalarT> delta_time = *dep_fields["Delta Time"];

  // extract evaluated MDFields
  PHX::MDField<ScalarT> eqps = *eval_fields[eqps_string];
  PHX::MDField<ScalarT> xtal_rotation = *eval_fields[Re_string];
  PHX::MDField<ScalarT> stress = *eval_fields[cauchy_string];
  PHX::MDField<ScalarT> plastic_deformation = *eval_fields[Fp_string];
  PHX::MDField<ScalarT> velocity_gradient = *eval_fields[L_string];
  PHX::MDField<ScalarT> source = *eval_fields[source_string];
  PHX::MDField<ScalarT> cp_residual = *eval_fields[residual_string];

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
  // extract hardening on each slip system
  std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > hards;
  std::vector<Albany::MDArray *> previous_hards;
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    std::string t_h = Albany::strint("tau_hard", num_ss + 1, '_');
    std::string tau_hard_string = (*field_name_map_)[t_h];
    hards.push_back(eval_fields[tau_hard_string]);
    previous_hards.push_back(
        &((*workset.stateArrayPtr)[tau_hard_string + "_old"]));
  }

  // store shear on each slip system for output
  std::vector<Teuchos::RCP<PHX::MDField<ScalarT> > > shears;
  for (int num_ss = 0; num_ss < num_slip_; ++num_ss) {
    std::string t = Albany::strint("tau", num_ss + 1, '_');
    std::string tau_string = (*field_name_map_)[t];
    shears.push_back(eval_fields[tau_string]);
  }

  // get state variables
  Albany::MDArray previous_plastic_deformation =
      (*workset.stateArrayPtr)[Fp_string + "_old"];

  ScalarT tau, gamma, dgamma;
  ScalarT dt = delta_time(0);
  ScalarT tcurrent = time(0);
  Intrepid::Tensor<ScalarT> Lp_np1(num_dims_);
  Intrepid::Tensor<ScalarT> F_np1(num_dims_), Fp_n(num_dims_), Fp_np1(num_dims_);
  Intrepid::Tensor<ScalarT> sigma_np1(num_dims_), S_np1(num_dims_);
  Intrepid::Tensor<ScalarT> Re_np1(num_dims_);
  std::vector<ScalarT> slip_n(num_slip_), slip_np1(num_slip_), hardness_n(num_slip_), hardness_np1(num_slip_), shear_np1(num_slip_);
  ScalarT norm_slip_residual;
  I_ = Intrepid::eye<RealType>(num_dims_);

#ifdef PRINT_OUTPUT
  std::ofstream out("output.dat", std::fstream::app);
#endif

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {

      // -- Copy everything into local data structures --
      // Initial guesses for np1 quantities assume zero plastic slip over the load step
      // Tensors
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
	  F_np1(i, j) = def_grad(cell, pt, i, j);
          Fp_n(i, j) = previous_plastic_deformation(cell, pt, i, j);
	  Fp_np1(i, j) = Fp_n(i, j);
        }
      }
      // Slip-system quantities
      for (int s(0); s < num_slip_; ++s) {
	slip_n[s] = (*(previous_slips[s]))(cell, pt);
	slip_np1[s] = slip_n[s];
	hardness_n[s] = (*(previous_hards[s]))(cell, pt);
	hardness_np1[s] = hardness_n[s];
      }





      // Evaluate stress quantities and evaluate norm_slip_residual under the assumption that slip increments are zero
      residual(dt, slip_n, slip_np1, hardness_np1, F_np1, Fp_np1, sigma_np1, S_np1, shear_np1, norm_slip_residual);

      // Determine convergence tolerances for the nonlinear solver
      RealType residual_tolerance = 1.0e-6 * Sacado::ScalarValue<ScalarT>::eval(norm_slip_residual);
      int iteration(0), max_iterations(1);

      // Iterate until values for slip_np1 are found that drive norm_slip_residual below the tolerance, or until
      // the maximum number of iterations has been reached.
      while(iteration < max_iterations){

	// Update the guess for slip_np1
	// This is a work in progress.  Currently, we'll just use the explicit approach to compute slip_np1.
	// Soon, we'll replace this call with the proper machinery for the implicit Newton scheme.
	predictor(dt, slip_n, slip_np1, hardness_n, hardness_np1, F_np1, Lp_np1, Fp_np1);	

 	residual(dt, slip_n, slip_np1, hardness_np1, F_np1, Fp_np1, sigma_np1, S_np1, shear_np1, norm_slip_residual);
	iteration += 1;
      }






      // The EQPS can be computed (or can it?) from the Cauchy Green strain of Fp.
      Intrepid::Tensor<ScalarT> CGS_Fp(num_dims_);
      CGS_Fp = 0.5*(((Intrepid::transpose(Fp_np1))*Fp_np1) - (Intrepid::eye<ScalarT>(num_dims_)));
      eqps(cell, pt) = sqrt((Intrepid::dotdot(CGS_Fp, CGS_Fp))*2.0/3.0);
      // The xtal rotation from the polar decomp of Fe.
      // Saint Venant–Kirchhoff model
#ifdef DECOUPLE
      Fe_ = F_np1;
#else
      Fe_ = F_np1 * (Intrepid::inverse(Fp_np1));
#endif
      Re_np1 = Intrepid::polar_rotation(Fe_);

      // -- Fill Albany data containers --
      // Scalars
      source(cell, pt) = 0.0;
      // Tensors
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          xtal_rotation(cell, pt, i, j) = Re_np1(i, j);
          plastic_deformation(cell, pt, i, j) = Fp_np1(i, j);
          stress(cell, pt, i, j) = sigma_np1(i, j);
          velocity_gradient(cell, pt, i, j) = Lp_np1(i, j);
        }
      }
      // Slip-system quantities
      for (int s(0); s < num_slip_; ++s) {
	(*(slips[s]))(cell, pt) = slip_np1[s];
	(*(hards[s]))(cell, pt) = hardness_np1[s];
	(*(shears[s]))(cell, pt) = shear_np1[s];
      }

#ifdef PRINT_OUTPUT
      if (cell == 0 && pt == 0) {
        out << "\n" << "time: ";
        out << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(tcurrent) << " ";
        out << "\n";
        out << "\n" << "F: ";
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            out << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(F(i,j)) << " ";
          }
        }
        out << "\n" << "Fp: ";
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            out << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(Fp(i,j)) << " ";
          }
        }
        out << "\n" << "Sigma: ";
        for (int i(0); i < num_dims_; ++i) {
          for (int j(0); j < num_dims_; ++j) {
            out << std::setprecision(12) << Sacado::ScalarValue<ScalarT>::eval(sigma(i,j)) << " ";
          }
        }
        out << "\n" << "Lp: ";
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
predictor(ScalarT                            dt,
	  std::vector<ScalarT> const &       slip_n,
	  std::vector<ScalarT> &             slip_np1,
	  std::vector<ScalarT> const &       hardness_n,
	  std::vector<ScalarT> &             hardness_np1,
	  Intrepid::Tensor<ScalarT> const &  F_np1,
	  Intrepid::Tensor<ScalarT> &        Lp_np1,
	  Intrepid::Tensor<ScalarT> &        Fp_np1)
{
  ScalarT g0, tau, tauC, gamma, dgamma, m, H, t1;
  Intrepid::Tensor<RealType> P(num_dims_);
  Intrepid::Tensor<ScalarT> sigma(num_dims_), S(num_dims_), expL(num_dims_), Fp_temp(num_dims_);

  computeStress(F_np1, Fp_np1, sigma, S);
  confirmTensorSanity(sigma, "first sigma calculation in predictor()");

  Lp_np1.fill(Intrepid::ZEROS);
  for (int s(0); s < num_slip_; ++s) {

    // material parameters
    P = slip_systems_[s].projector_;
    tauC = slip_systems_[s].tau_critical_;
    m = slip_systems_[s].gamma_exp_;
    g0 = slip_systems_[s].gamma_dot_0_;
    H = slip_systems_[s].H_;

    // compute resolved shear stresses
    tau = Intrepid::dotdot(P, S);
    int sign = tau < 0 ? -1 : 1;

    // initialize slip, previous slip, and gamma (slip)
    slip_np1[s] = slip_n[s];
    gamma = slip_n[s];

    // initialize hardening and previous hardening
    hardness_np1[s] = hardness_n[s];

    // calculate additional hardening
    ScalarT tmp_hard = H * std::fabs(gamma);
    if (tmp_hard > hardness_np1[s]) {
      hardness_np1[s] = tmp_hard;
    }
    // calculate slip increment with additional hardening
    t1 = std::fabs(tau / (tauC + hardness_np1[s]));
    dgamma = dt * g0 * std::fabs(std::pow(t1, m)) * sign;

    // update slip
    slip_np1[s] += dgamma;

    // calculate plastic velocity gradient
    Lp_np1 += (dgamma * P);
  }

  confirmTensorSanity(Lp_np1, "Lp_np1 in predictor().");

  // update plastic deformation gradient
  expL = Intrepid::exp(Lp_np1);
  Fp_temp = expL * Fp_np1;
  Fp_np1 = Fp_temp;

  confirmTensorSanity(Fp_np1, "Fp_np1 in predictor()");
}
//------------------------------------------------------------------------------

template<typename EvalT, typename Traits>
void CrystalPlasticityModel<EvalT, Traits>::
residual(ScalarT                            dt,
	 std::vector<ScalarT> const &       slip_n,
	 std::vector<ScalarT> const &       slip_np1,
	 std::vector<ScalarT> const &       hardness_np1,
	 Intrepid::Tensor<ScalarT> const &  F_np1,
	 Intrepid::Tensor<ScalarT> const &  Fp_np1,
	 Intrepid::Tensor<ScalarT> &        sigma_np1,
	 Intrepid::Tensor<ScalarT> &        S_np1,
	 std::vector<ScalarT> &             shear_np1,
	 ScalarT &                          norm_slip_residual)
{
  ScalarT g0, tauC, m, temp;
  ScalarT dgamma_value1, dgamma_value2;
  Intrepid::Tensor<RealType> P(num_dims_);
  int sign;

  Intrepid::Vector<ScalarT> slip_residual(num_slip_);

  computeStress(F_np1, Fp_np1, sigma_np1, S_np1);
  confirmTensorSanity(sigma_np1, "sigma calculation in residual()");

  for (int s(0); s < num_slip_; ++s) {

    // Material properties
    P = slip_systems_[s].projector_;
    tauC = slip_systems_[s].tau_critical_;
    m = slip_systems_[s].gamma_exp_;

    // The current computed value of dgamma
    dgamma_value1 = slip_np1[s] - slip_n[s];

    // Compute slip increment using Fe_np1
    shear_np1[s] = Intrepid::dotdot(P, S_np1);
    sign = shear_np1[s] < 0 ? -1 : 1;
    temp = std::fabs(shear_np1[s] / (tauC + hardness_np1[s]));
    dgamma_value2 = dt * g0 * std::fabs(std::pow(temp, m)) * sign;

    // The difference between the slip increment calculations is the residual for this slip system
    slip_residual[s] = dgamma_value2 - dgamma_value1;
  }

  // Take norm of residual - protect sqrt (Saccado)
  norm_slip_residual = Intrepid::dot(slip_residual, slip_residual);
  if (norm_slip_residual > 0.0) {
    norm_slip_residual = std::sqrt(norm_slip_residual);
  }
}

//------------------------------------------------------------------------------

template<typename EvalT, typename Traits>
void CrystalPlasticityModel<EvalT, Traits>::
computeStress(Intrepid::Tensor<ScalarT> const & F,
	      Intrepid::Tensor<ScalarT> const & Fp,
	      Intrepid::Tensor<ScalarT> &       sigma,
	      Intrepid::Tensor<ScalarT> &       S)

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
