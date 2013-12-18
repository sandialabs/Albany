//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Utils.hpp"

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
CrystalPlasticityModel<EvalT, Traits>::
CrystalPlasticityModel(Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
    num_slip_(p->get<int>("Number of Slip Systems", 0))
{
  std::cout << ">>> in cp constructor\n";
  slip_systems_.resize(num_slip_);
  std::cout << ">>> parameter list:\n" << *p << std::endl;
  Teuchos::ParameterList e_list = p->sublist("Crystal Elasticity");
  // assuming cubic symmetry
  c11_ = e_list.get<RealType>("C11");
  c12_ = e_list.get<RealType>("C12");
  c44_ = e_list.get<RealType>("C44");
  Intrepid::Tensor4<RealType> C(num_dims_);
  C.fill(Intrepid::ZEROS);
  for (int i = 0; i < num_dims_; ++i) {
    C(i,i,i,i) = c11_;
    for (int j = i+1; j < num_dims_; ++j) {
      C(i,i,j,j) = C(j,j,i,i) = c12_;
      C(i,j,i,j) = C(j,i,j,i) = c44_;
    }
  }
  std::cout << "C\n" << C << "\n";
// NOTE check if basis is given else default
// NOTE default to coordinate axes and also construct 3rd direction if only 2 given
  orientation_.set_dimension(num_dims_);
  for (int i = 0; i < num_dims_; ++i) {
    std::vector<RealType> b_temp = e_list.get<Teuchos::Array<RealType> >(Albany::strint("Basis Vector", i+1)).toVector();
    RealType norm = 0.;
    for (int j = 0; j < num_dims_; ++j) {
      norm += b_temp[j]*b_temp[j];
    }
// NOTE check zero, rh system
    norm = 1./std::sqrt(norm);
    for (int j = 0; j < num_dims_; ++j) {
      orientation_(i,j) = b_temp[j]*norm;
    }
  }
  C_ = Intrepid::kronecker(orientation_,C);
  std::cout << "Q[x]C\n" << C_ << "\n";
  
  std::cout << "c " << c11_ << " " << c12_ << " " << c44_ << "\n";
  std::cout << "orientation\n" << orientation_ << "\n";
  for (int num_ss; num_ss < num_slip_; ++num_ss) {
    Teuchos::ParameterList ss_list = p->sublist(Albany::strint("Slip System", num_ss+1));

    std::vector<RealType> s_temp = ss_list.get<Teuchos::Array<RealType> >("Slip Direction").toVector();
    slip_systems_[num_ss].s_ = orientation_*(Intrepid::Vector<RealType>(num_dims_, &s_temp[0]));

    std::vector<RealType> n_temp = ss_list.get<Teuchos::Array<RealType> >("Slip Normal").toVector();
    slip_systems_[num_ss].n_ = orientation_*(Intrepid::Vector<RealType>(num_dims_, &n_temp[0]));

    slip_systems_[num_ss].projector_ = Intrepid::dyad(slip_systems_[num_ss].s_, slip_systems_[num_ss].n_);

    slip_systems_[num_ss].tau_critical_ = ss_list.get<RealType>("Tau Critical");
    slip_systems_[num_ss].gamma_dot_0_ = ss_list.get<RealType>("Gamma Dot");
    slip_systems_[num_ss].gamma_exp_ = ss_list.get<RealType>("Gamma Exponent");
  }
  std::cout << "<<< done with parameter list\n";

  // rotate elastic tensor and slip systems to match given orientation
  

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair("F", dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair("J", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Delta Time", dl->workset_scalar));

  // retrive appropriate field name strings
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string = (*field_name_map_)["Fp"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];
  std::string L_string = (*field_name_map_)["Velocity_Gradient"];

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(Fp_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(source_string, dl->qp_scalar));

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
  this->state_var_output_flags_.push_back(true);
  //
  // L
  this->num_state_variables_++;
  this->state_var_names_.push_back(L_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("identity");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  //
  // mechanical source
  this->num_state_variables_++;
  this->state_var_names_.push_back(source_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(true);

  std::cout << "<<< done in cp constructor\n";
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void CrystalPlasticityModel<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
{
  std::cout << ">>> in cp compute state\n";
  // extract dependent MDFields
  PHX::MDField<ScalarT> def_grad = *dep_fields["F"];
  PHX::MDField<ScalarT> J = *dep_fields["J"];
  PHX::MDField<ScalarT> delta_time = *dep_fields["Delta Time"];

  // retrive appropriate field name strings
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string = (*field_name_map_)["Fp"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];

  // extract evaluated MDFields
  PHX::MDField<ScalarT> stress = *eval_fields[cauchy_string];
  PHX::MDField<ScalarT> plastic_deformation = *eval_fields[Fp_string];
  PHX::MDField<ScalarT> source = *eval_fields[source_string];

  // get state variables
  Albany::MDArray previous_plastic_deformation = (*workset.stateArrayPtr)[Fp_string + "_old"];

  ScalarT tau, dgamma;
  ScalarT g0, tauC, m;
  ScalarT dt = delta_time(0);
  Intrepid::Tensor<ScalarT> Fp_temp(num_dims_),Fpinv(num_dims_);
  Intrepid::Tensor<ScalarT> F(num_dims_), Fp(num_dims_);
  Intrepid::Tensor<ScalarT> sigma(num_dims_);
  Intrepid::Tensor<ScalarT> L(num_dims_), expL(num_dims_);
  Intrepid::Tensor<RealType> P(num_dims_);
  I_=Intrepid::eye<RealType>(num_dims_);

  for (std::size_t cell(0); cell < workset.numCells; ++cell) {
    for (std::size_t pt(0); pt < num_pts_; ++pt) {

      std::cout << ">>> cell " << cell << " point " << pt << " <<<\n";
      // fill local tensors
      F.fill(&def_grad(cell, pt, 0, 0));
      for (std::size_t i(0); i < num_dims_; ++i) {
        for (std::size_t j(0); j < num_dims_; ++j) {
          Fp(i, j) = ScalarT(previous_plastic_deformation(cell, pt, i, j));
        }
      }

      // compute stress 
      computeStress(F,Fp,sigma);
      std::cout << "sigma-PRE\n" << sigma << "\n"; 
      std::cout << "number of slip systems " << num_slip_ << "\n"; 
      if (num_slip_ >0) { // crystal plasticity
        // compute velocity gradient
        L.fill(Intrepid::ZEROS);
        for (std::size_t s(0); s < num_slip_; ++s) {
          P  = slip_systems_[s].projector_; 
          // compute resolved shear stresses
          tau = Intrepid::dotdot(P,sigma);
          std::cout << s << " tau " << tau << "\n"; 
          // compute  dgammas
          g0   = slip_systems_[s].gamma_dot_0_;
          tauC = slip_systems_[s].tau_critical_;
          m    = slip_systems_[s].gamma_exp_;
          dgamma = dt*g0*std::pow(tau/tauC,m);
          L += (dgamma* P);
        }
        std::cout << "L\n" << L << "\n"; 

        // update plastic deformation gradient
        expL = Intrepid::exp(L);
        std::cout << "expL\n" << expL << "\n"; 
        Fp_temp = expL * Fp;
        Fp = Fp_temp;
        std::cout << "Fp-POST\n" << Fp << "\n"; 

        // recompute stress
        computeStress(F,Fp,sigma);
        // NOTE this is cut & paste
        std::cout << "sigma-POST\n" << sigma << "\n"; 
      }

      // history
      source(cell, pt) = 0.0;
      for (std::size_t i(0); i < num_dims_; ++i) {
        for (std::size_t j(0); j < num_dims_; ++j) {
          plastic_deformation(cell, pt, i, j) = Fp(i, j);
        }
      }
      std::cout << cell << " " << pt << " ";
      // store stress
      for (std::size_t i(0); i < num_dims_; ++i) {
        for (std::size_t j(0); j < num_dims_; ++j) {
          stress(cell, pt, i, j) = sigma(i, j);
          std::cout << F(i,j) << " " << Fp(i,j) << " " << sigma(i,j) << " ";
        }
      }
      std::cout << "#\n";
    }
  }
  std::cout << "<<< done in cp compute state\n";
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void CrystalPlasticityModel<EvalT, Traits>::
computeStress(Intrepid::Tensor<ScalarT> const & F,
              Intrepid::Tensor<ScalarT> const & Fp,
              Intrepid::Tensor<ScalarT>       & sigma) 

{
  // Saint Venant–Kirchhoff model
  std::cout << "F\n" << F << "\n";
  std::cout << "Fp\n" << Fp << "\n";
  Fpinv_ = Intrepid::inverse(Fp);
  std::cout << "Fp-1\n" << Fpinv_ << "\n";
  Fe_ = F * Fpinv_;
  std::cout << "Fe\n" << Fe_ << "\n";
  E_ = 0.5*( Intrepid::transpose(Fe_) * Fe_ - I_);
  std::cout << "E\n" << E_ << "\n";
  S_ = Intrepid::dotdot(C_,E_);
  std::cout << "S\n" << S_ << "\n";
  sigma = (1.0 / Intrepid::det(F) ) * F* S_ * Intrepid::transpose(F);
  
}
//------------------------------------------------------------------------------
}

