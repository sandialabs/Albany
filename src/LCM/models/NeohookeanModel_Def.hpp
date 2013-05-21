//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  NeohookeanModel<EvalT, Traits>::
  NeohookeanModel(Teuchos::ParameterList* p,
                  const Teuchos::RCP<Albany::Layouts>& dl):
    LCM::ConstitutiveModel<EvalT,Traits>(p,dl)
  {
    // define the dependent fields
    this->dep_field_map_.insert( std::make_pair("F", dl->qp_tensor) );
    this->dep_field_map_.insert( std::make_pair("J", dl->qp_scalar) );
    this->dep_field_map_.insert( std::make_pair("Poissons Ratio", dl->qp_scalar) );
    this->dep_field_map_.insert( std::make_pair("Elastic Modulus", dl->qp_scalar) );

    // define the evaluated fields
    std::string cauchy = (*field_name_map_)["Cauchy_Stress"];
    this->eval_field_map_.insert( std::make_pair(cauchy, dl->qp_tensor) );

    // define the state variables
    this->num_state_variables_++;
    this->state_var_names_.push_back(cauchy);
    this->state_var_layouts_.push_back(dl->qp_tensor);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(false);
    this->state_var_output_flags_.push_back(true);
  }
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void NeohookeanModel<EvalT, Traits>::
  computeState(typename Traits::EvalData workset,
               std::map<std::string,Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
               std::map<std::string,Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
  {
    // extract dependent MDFields
    PHX::MDField<ScalarT> defGrad        = *dep_fields["F"];
    PHX::MDField<ScalarT> J              = *dep_fields["J"];
    PHX::MDField<ScalarT> poissonsRatio  = *dep_fields["Poissons Ratio"];
    PHX::MDField<ScalarT> elasticModulus = *dep_fields["Elastic Modulus"];
    // extract evaluated MDFields
    std::string cauchy = (*field_name_map_)["Cauchy_Stress"];
    PHX::MDField<ScalarT> stress = *eval_fields[cauchy];
    ScalarT kappa;
    ScalarT mu;
    ScalarT Jm53;

    Intrepid::Tensor<ScalarT> F(num_dims_), b(num_dims_), sigma(num_dims_);
    Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));

    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t pt(0); pt < num_pts_; ++pt) {
        kappa = 
          elasticModulus(cell,pt) / ( 3. * ( 1. - 2. * poissonsRatio(cell,pt) ) );
        mu = 
          elasticModulus(cell,pt) / ( 2. * ( 1. + poissonsRatio(cell,pt) ) );
        Jm53 = std::pow(J(cell,pt), -5./3.);

        F.fill(&defGrad(cell,pt,0,0));
        b = F*transpose(F);
        sigma = 0.5 * kappa * ( J(cell,pt) - 1. / J(cell,pt) ) * I
          + mu * Jm53 * Intrepid::dev(b);

        for (std::size_t i=0; i < num_dims_; ++i)
          for (std::size_t j=0; j < num_dims_; ++j)
            stress(cell,pt,i,j) = sigma(i,j);
      }
    }
  }
  //----------------------------------------------------------------------------
} 

