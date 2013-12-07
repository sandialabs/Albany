//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Utils.hpp"

#include "LocalNonlinearSolver.hpp"

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
CrystalPlasticityModel<EvalT, Traits>::
CrystalPlasticityModel(Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
    sat_mod_(p->get<RealType>("Saturation Modulus", 0.0)),
    sat_exp_(p->get<RealType>("Saturation Exponent", 0.0)),
    num_slip_(p->get<int>("Number of Slip Systems", 1))
{
  std::cout << ">>> in cp constructor\n";
  slip_systems_.resize(num_slip_);
  std::cout << ">>> parameter list:\n" << *p << std::endl;
  for (int num_ss; num_ss < num_slip_; ++num_ss) {
    Teuchos::ParameterList ss_list = p->sublist(Albany::strint("Slip System", num_ss+1));

    std::vector<RealType> s_temp = ss_list.get<Teuchos::Array<RealType> >("Slip Direction").toVector();
    slip_systems_[num_ss].s_ = Intrepid::Vector<RealType>(num_dims_, &s_temp[0]);

    std::vector<RealType> n_temp = ss_list.get<Teuchos::Array<RealType> >("Slip Normal").toVector();
    slip_systems_[num_ss].n_ = Intrepid::Vector<RealType>(num_dims_, &n_temp[0]);

    slip_systems_[num_ss].projectors_ = Intrepid::dyad(slip_systems_[num_ss].s_, slip_systems_[num_ss].n_);

    slip_systems_[num_ss].tau_critical_ = ss_list.get<RealType>("Tau Critical");
    slip_systems_[num_ss].gamma_dot_0_ = ss_list.get<RealType>("Gamma Dot");
    slip_systems_[num_ss].gamma_exp_ = ss_list.get<RealType>("Gamma Exponential");
  }
  std::cout << "<<< done with parameter list\n";
  

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair("F", dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair("J", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Poissons Ratio", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Elastic Modulus", dl->qp_scalar));
#ifndef REMOVE_THIS
  this->dep_field_map_.insert(std::make_pair("Yield Strength", dl->qp_scalar));
  this->dep_field_map_.insert(
      std::make_pair("Hardening Modulus", dl->qp_scalar));
#endif
  this->dep_field_map_.insert(std::make_pair("Delta Time", dl->workset_scalar));

  // retrive appropriate field name strings
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string = (*field_name_map_)["Fp"];
#ifndef REMOVE_THIS
  std::string eqps_string = (*field_name_map_)["eqps"];
#endif
  std::string source_string = (*field_name_map_)["Mechanical_Source"];

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(Fp_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(eqps_string, dl->qp_scalar));
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
  this->state_var_output_flags_.push_back(false);
  //
  // gammas
#ifndef REMOVE_THIS 
  //
  // eqps
  this->num_state_variables_++;
  this->state_var_names_.push_back(eqps_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
#endif
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
  PHX::MDField<ScalarT> poissons_ratio = *dep_fields["Poissons Ratio"];
  PHX::MDField<ScalarT> elastic_modulus = *dep_fields["Elastic Modulus"];
  PHX::MDField<ScalarT> delta_time = *dep_fields["Delta Time"];

  // retrive appropriate field name strings
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string = (*field_name_map_)["Fp"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];

  // extract evaluated MDFields
  PHX::MDField<ScalarT> stress = *eval_fields[cauchy_string];
  PHX::MDField<ScalarT> Fp = *eval_fields[Fp_string];
  PHX::MDField<ScalarT> source = *eval_fields[source_string];

  // get state variables
  Albany::MDArray Fpold =
      (*workset.stateArrayPtr)[Fp_string + "_old"];

  ScalarT c11,c12,c44;
  ScalarT tau, dgamma, dt;
  ScalarT g0;
  dt = 1.; // HACK

#ifndef REMOVE_THIS 
  ScalarT Y,nu;
  std::string eqps_string = (*field_name_map_)["eqps"];
  PHX::MDField<ScalarT> eqps = *eval_fields[eqps_string];
  Albany::MDArray eqpsold =
      (*workset.stateArrayPtr)[eqps_string + "_old"];
  ScalarT p;
#endif

  Intrepid::Tensor<ScalarT> F(num_dims_), Fe(num_dims_), Ee(num_dims_); 
  Intrepid::Tensor<ScalarT> Fpn(num_dims_), Fpinv(num_dims_);
  Intrepid::Tensor<ScalarT> L(num_dims_), expL(num_dims_), Fpnew(num_dims_);
  Intrepid::Tensor<ScalarT> s(num_dims_), sigma(num_dims_);
  Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));

  for (std::size_t cell(0); cell < workset.numCells; ++cell) {
    for (std::size_t pt(0); pt < num_pts_; ++pt) {

      // fill local tensors
      F.fill(&def_grad(cell, pt, 0, 0));
      for (std::size_t i(0); i < num_dims_; ++i) {
        for (std::size_t j(0); j < num_dims_; ++j) {
          Fpn(i, j) = ScalarT(Fpold(cell, pt, i, j));
        }
      }

      // compute trial state
      Fpinv = Intrepid::inverse(Fpn);
      Fe = F * Fpinv;
      Ee = 0.5*( Intrepid::transpose(Fe) * Fe - I);

      // compute stress 
      // elastic modulis NOTE make anisotropic
      Y = elastic_modulus(cell, pt);
      nu = poissons_ratio(cell, pt);
      Y *= 1./((1.+nu)*(1.-2.*nu));
      c11 = (1.   -nu)*Y;
      c12 =        nu *Y;
      c44 = (1.-2.*nu)*Y;
      sigma = c44*Ee;
#if 0
      sigma(0,0) = c11*Ee(0,0)+c12*(Ee(1,1)+Ee(2,2));
      sigma(1,1) = c11*Ee(1,1)+c12*(Ee(0,0)+Ee(2,2));
      sigma(2,2) = c11*Ee(2,2)+c12*(Ee(1,1)+Ee(0,0));

      //HACK L.initialize(0.);
      for (int i; i < num_slip_; ++i) {
        // compute resolved shear stresses
        tau = 1.;
         
        // compute  dgammas
        g0 = slip_systems_[i].gamma_dot_0_;
        dgamma = dt*g0*tau;

        // compute velocity gradient
        // HACK L += dgamma* (slip_systems_[i].projectors_); 
      }

      // update plastic deformation gradient
      expL = Intrepid::exp(L);
      Fpnew = expL * Fpn;
#endif

      // history
      eqps(cell, pt) = eqpsold(cell, pt);
      source(cell, pt) = 0.0;
      for (std::size_t i(0); i < num_dims_; ++i) {
        for (std::size_t j(0); j < num_dims_; ++j) {
          Fp(cell, pt, i, j) = Fpnew(i, j);
        }
      }

      // store stress
      for (std::size_t i(0); i < num_dims_; ++i) {
        for (std::size_t j(0); j < num_dims_; ++j) {
          stress(cell, pt, i, j) = sigma(i, j);
        }
      }
    }
  }
  std::cout << "<<< done in cp compute state\n";
}
//------------------------------------------------------------------------------
}

