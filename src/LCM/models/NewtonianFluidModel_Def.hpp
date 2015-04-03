//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
NewtonianFluidModel<EvalT, Traits>::
NewtonianFluidModel(Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
    mu_( p->get<RealType>("Shear Viscosity", 1.0) )
{
  // retrive appropriate field name strings
  std::string F_string = (*field_name_map_)["F"];
  std::string Fp_string = (*field_name_map_)["Fp"];
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair(F_string, dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair("Delta Time", dl->workset_scalar));

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(Fp_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));

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

}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void NewtonianFluidModel<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
{

  std::string F_string      = (*field_name_map_)["F"];
  std::string Fp_string     = (*field_name_map_)["Fp"];
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];

  // extract dependent MDFields
  PHX::MDField<ScalarT> def_grad         = *dep_fields[F_string];
  PHX::MDField<ScalarT> delta_time       = *dep_fields["Delta Time"];

  // extract evaluated MDFields
  PHX::MDField<ScalarT> stress = *eval_fields[cauchy_string];
  PHX::MDField<ScalarT> Fp     = *eval_fields[Fp_string];

  // get State Variables
  Albany::MDArray Fpold = (*workset.stateArrayPtr)[Fp_string + "_old"];

  // pressure is hard coded as 1 for now
  // this is likely not general enough :)
  ScalarT p = 1;

  // time increment
  ScalarT dt = delta_time(0);

  std::cout << "DT " << dt << std::endl;

  // containers
  Intrepid::Tensor<ScalarT> Fnew(num_dims_);
  Intrepid::Tensor<ScalarT> Fold(num_dims_);
  Intrepid::Tensor<ScalarT> Finc(num_dims_);
  Intrepid::Tensor<ScalarT> logFinc(num_dims_);
  Intrepid::Tensor<ScalarT> V(num_dims_);
  Intrepid::Tensor<ScalarT> Vinc(num_dims_);
  Intrepid::Tensor<ScalarT> logVinc(num_dims_);
  Intrepid::Tensor<ScalarT> R(num_dims_);
  Intrepid::Tensor<ScalarT> Rinc(num_dims_);
  Intrepid::Tensor<ScalarT> logRinc(num_dims_);
  Intrepid::Tensor<ScalarT> L(num_dims_);
  Intrepid::Tensor<ScalarT> D(num_dims_);
  Intrepid::Tensor<ScalarT> sigma(num_dims_);
  Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {

      // should only be the first time step
      if ( dt == 0 ) {
        for (int i=0; i < num_dims_; ++i) {
        for (int j=0; j < num_dims_; ++j) {
          Fp(cell,pt,i,j) = 0.0;
          stress(cell,pt,i,j) = 0.0;
        }}
      }
      else {

        // old deformation gradient
        for (int i=0; i < num_dims_; ++i)
        for (int j=0; j < num_dims_; ++j)
          Fold(i,j) = ScalarT(Fpold(cell,pt,i,j));

        // current deformation gradient
        Fnew.fill(def_grad,cell,pt,0,0);

        // left stretch V, and rotation R, from left polar decomposition of
        // new deformation gradient
        boost::tie(V,R) = Intrepid::polar_left_eig(Fnew);

        // incremental left stretch Vinc, incremental rotation Rinc, and log
        // of incremental left stretch, logVinc
        boost::tie(Vinc,Rinc,logVinc) = Intrepid::polar_left_logV_lame(Finc);

        // log of incremental rotation
        logRinc = Intrepid::log_rotation(Rinc);

        // log of incremental deformation gradient
        logFinc = Intrepid::bch(logVinc, logRinc);

        // velocity gradient
        L = (1.0/dt)*logFinc;

        // strain rate (a.k.a rate of deformation)
        D = Intrepid::sym(L);

        // stress tensor
        sigma = -p*I +  mu_*( D - (2.0/3.0)*Intrepid::trace(D)*I);

        // update state
        for (int i=0; i < num_dims_; ++i) {
        for (int j=0; j < num_dims_; ++j) {
          Fp(cell,pt,i,j) = def_grad(cell,pt,i,j);
          stress(cell,pt,i,j) = sigma(i,j);
        }}

      }
    }
  }
}
//------------------------------------------------------------------------------
}
