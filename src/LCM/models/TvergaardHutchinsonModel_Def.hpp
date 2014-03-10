//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include <Teuchos_TestForException.hpp>
#include <Phalanx_DataLayout.hpp>

namespace LCM
{

//------------------------------------------------------------------------------
// See Klein, Theoretical and Applied Fracture Mechanics (2001)
// for details regarding implementation
// NOTE: beta_0, beta_1 and beta_2 are parameters that enable one to favor
// tension or shear. The default tensor should be identity.
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
TvergaardHutchinsonModel<EvalT, Traits>::
TvergaardHutchinsonModel(Teuchos::ParameterList* p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
  delta_1(p->get<RealType>("delta_1", 0.5)),
  delta_2(p->get<RealType>("delta_2", 0.5)),
  delta_c(p->get<RealType>("delta_c", 1.0)),
  sigma_c(p->get<RealType>("sigma_c", 1.0)),
  beta_0(p->get<RealType>("beta_0", 1.0)),
  beta_1(p->get<RealType>("beta_1", 1.0)),
  beta_2(p->get<RealType>("beta_2", 1.0))
{

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair("Vector Jump", dl->qp_vector));
  this->dep_field_map_.insert(std::make_pair("Current Basis", dl->qp_tensor));

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair("Cohesive_Traction", dl->qp_vector));
  this->eval_field_map_.insert(std::make_pair("Normal_Traction", dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Shear_Traction", dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Normal_Jump", dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Shear_Jump", dl->qp_scalar));

  // define the state variables
  //
  // cohesive traction
  this->num_state_variables_++;
  this->state_var_names_.push_back("Cohesive_Traction");
  this->state_var_layouts_.push_back(dl->qp_vector);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Cohesive Traction", false));
  //
  // normal traction
  this->num_state_variables_++;
  this->state_var_names_.push_back("Normal_Traction");
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Normal Traction", false));
  //
  // shear traction
  this->num_state_variables_++;
  this->state_var_names_.push_back("Shear_Traction");
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Shear Traction", false));
  //
  // normal jump
  this->num_state_variables_++;
  this->state_var_names_.push_back("Normal_Jump");
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Normal Jump", false));
  //
  // shear jump
  this->num_state_variables_++;
  this->state_var_names_.push_back("Shear_Jump");
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Shear Jump", false));
  //
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void TvergaardHutchinsonModel<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
{

  // extract dependent MDFields
  PHX::MDField<ScalarT> jump = *dep_fields["Vector Jump"];
  PHX::MDField<ScalarT> basis = *dep_fields["Current Basis"];

  // extract evaluated MDFields
  PHX::MDField<ScalarT> traction = *eval_fields["Cohesive_Traction"];
  PHX::MDField<ScalarT> traction_normal = *eval_fields["Normal_Traction"];
  PHX::MDField<ScalarT> traction_shear = *eval_fields["Shear_Traction"];
  PHX::MDField<ScalarT> jump_normal = *eval_fields["Normal_Jump"];
  PHX::MDField<ScalarT> jump_shear = *eval_fields["Shear_Jump"];

  for (std::size_t cell(0); cell < workset.numCells; ++cell) {
    for (std::size_t pt(0); pt < num_pts_; ++pt) {
      
      //current basis vector
      Intrepid::Vector<ScalarT> g_0(3, &basis(cell, pt, 0, 0));
      Intrepid::Vector<ScalarT> g_1(3, &basis(cell, pt, 1, 0));
      Intrepid::Vector<ScalarT> n(3, &basis(cell, pt, 2, 0));

      //construct orthogonal unit basis
      Intrepid::Vector<ScalarT> t_0(0.0,0.0,0.0), t_1(0.0,0.0,0.0);
      t_0 = g_0 / norm(g_0);
      t_1 = cross(n,t_0);

      //construct transformation matrix Q (2nd order tensor)
      Intrepid::Tensor<ScalarT> Q(3, Intrepid::ZEROS);
      // manually fill Q = [t_0; t_1; n];
      Q(0,0) = t_0(0); Q(1,0) = t_0(1);  Q(2,0) = t_0(2);
      Q(0,1) = t_1(0); Q(1,1) = t_1(1);  Q(2,1) = t_1(2);
      Q(0,2) = n(0);   Q(1,2) = n(1);    Q(2,2) = n(2);

      //global and local jump
      Intrepid::Vector<ScalarT> jump_global(3, &jump(cell, pt, 0));
      Intrepid::Vector<ScalarT> jump_local(3);
      jump_local = Intrepid::dot(Intrepid::transpose(Q), jump_global);

      // define shear and normal components of jump
      // needed for interpenetration
      ScalarT JumpNormal, JumpShear;
      JumpNormal = jump_local(2);
      JumpShear = sqrt(jump_local(0)*jump_local(0) + jump_local(1)*jump_local(1));

      // matrix beta that controls relative effect of shear and normal opening
      Intrepid::Tensor<ScalarT> beta(3, Intrepid::ZEROS);
      beta(0,0) = beta_0; beta(1,1) = beta_1; beta(2,2) = beta_2;

      // compute scalar effective jump
      ScalarT jump_eff;
      jump_eff =
        std::sqrt(Intrepid::dot(jump_local,Intrepid::dot(beta,jump_local)));

      // traction-separation law from Tvergaard-Hutchinson 1992
      ScalarT sigma_eff;
      //Sacado::ScalarValue<ScalarT>::eval
      if(jump_eff <= delta_1)
        sigma_eff = sigma_c * jump_eff / delta_1;
      else if(jump_eff > delta_1 && jump_eff <= delta_2)
        sigma_eff = sigma_c;
      else if(jump_eff > delta_2 && jump_eff <= delta_c)
        sigma_eff = sigma_c * (delta_c - jump_eff) / (delta_c - delta_2);
      else
        sigma_eff = 0.0;

      // construct traction vector
      Intrepid::Vector<ScalarT> traction_local(3);
      traction_local.clear();
      if(jump_eff != 0)
        traction_local = Intrepid::dot(beta,jump_local) * sigma_eff / jump_eff;

      // norm of the local shear components of the traction
      ScalarT TractionShear;
      TractionShear = sqrt(traction_local(0)*traction_local(0) + traction_local(1)*traction_local(1));

      // global traction vector
      Intrepid::Vector<ScalarT> traction_global(3);
      traction_global = Intrepid::dot(Q, traction_local);

      traction(cell,pt,0) = traction_global(0);
      traction(cell,pt,1) = traction_global(1);
      traction(cell,pt,2) = traction_global(2);

      // update state variables
      traction_normal(cell,pt) = traction_local(2);
      traction_shear(cell,pt) = TractionShear;
      jump_normal(cell,pt) = JumpNormal;
      jump_shear(cell,pt) = JumpShear;



    }
  }
}
//------------------------------------------------------------------------------
}

