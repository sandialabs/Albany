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
  NeohookeanModel(const Teuchos::ParameterList* p,
                  const Teuchos::RCP<Albany::Layouts>& dl):
    LCM::ConstitutiveModel<EvalT,Traits>(p,dl)
  {
    // define the dependent fields
    this->dep_field_map_.insert( std::make_pair("F", dl->qp_tensor) );
    this->dep_field_map_.insert( std::make_pair("J", dl->qp_scalar) );
    this->dep_field_map_.insert( std::make_pair("Poissons Ratio", dl->qp_scalar) );
    this->dep_field_map_.insert( std::make_pair("Elastic Modulus", dl->qp_scalar) );

    // define the evaluated fields
    this->eval_field_map_.insert( std::make_pair("Cauchy_Stress", dl->qp_tensor) );

    // define the state variables
    this->num_state_variables_++;
    this->state_var_names_.push_back("Cauchy_Stress");
    this->state_var_layouts_.push_back(dl->qp_tensor);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(false);
    this->state_var_output_flags_.push_back(true);
  }
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void NeohookeanModel<EvalT, Traits>::
  computeEnergy(typename Traits::EvalData workset,
                std::map<std::string,Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
                std::map<std::string,Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
  {
    // not implemented
  }
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  void NeohookeanModel<EvalT, Traits>::
  computeState(typename Traits::EvalData workset,
               std::map<std::string,Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
               std::map<std::string,Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
  {
    std::cout << "In NeohookeanModel::computeState" << std::endl;
    // extract dependent MDFields
    std::cout << "  grab defGrad" << std::endl;
    PHX::MDField<ScalarT> defGrad        = *dep_fields["F"];
    std::cout << "  grab J" << std::endl;
    PHX::MDField<ScalarT> J              = *dep_fields["J"];
    std::cout << "  grab nu" << std::endl;
    PHX::MDField<ScalarT> poissonsRatio  = *dep_fields["Poissons Ratio"];
    std::cout << "  grab E" << std::endl;
    PHX::MDField<ScalarT> elasticModulus = *dep_fields["Elastic Modulus"];
    // extract evaluated MDFields
    std::cout << "  grab stress" << std::endl;
    PHX::MDField<ScalarT> stress = *eval_fields["Cauchy_Stress"];
    ScalarT kappa;
    ScalarT mu;
    ScalarT Jm53;

    std::cout << "  initialize tensors" << std::endl;
    Intrepid::Tensor<ScalarT> F(num_dims_), b(num_dims_), sigma(num_dims_);
    Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(num_dims_));

    std::cout << "  start loop over cells" << std::endl;
    for (std::size_t cell(0); cell < workset.numCells; ++cell) {
      for (std::size_t pt(0); pt < num_pts_; ++pt) {

        std::cout << "  Print fields " << std::endl;
        std::cout << "   E    :  " << elasticModulus(cell,pt) << std::endl;
        std::cout << "   nu   :  " << poissonsRatio(cell,pt) << std::endl;        
        std::cout << "   J    :  " << J(cell,pt) << std::endl;

        kappa = 
          elasticModulus(cell,pt) / ( 3. * ( 1. - 2. * poissonsRatio(cell,pt) ) );
        mu = 
          elasticModulus(cell,pt) / ( 2. * ( 1. + poissonsRatio(cell,pt) ) );
        Jm53 = std::pow(J(cell,pt), -5./3.);

        std::cout << "   kappa: " << kappa << std::endl;
        std::cout << "   mu   : " << mu << std::endl;
        std::cout << "   Jm53 : " << Jm53 << std::endl;

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
  template<typename EvalT, typename Traits>
  void NeohookeanModel<EvalT, Traits>::
  computeTangent(typename Traits::EvalData workset,
                 std::map<std::string,Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
                 std::map<std::string,Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
  {
    // not implemented
  }
  //----------------------------------------------------------------------------
} 

