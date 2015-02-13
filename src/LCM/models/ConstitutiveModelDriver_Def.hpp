//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include <Intrepid_MiniTensor.h>

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
ConstitutiveModelDriver<EvalT, Traits>::
ConstitutiveModelDriver(Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl):
  residual_(p.get<std::string>("Residual Name"),dl->node_tensor),
  def_grad_(p.get<std::string>("F Name"),dl->qp_tensor),
  stress_(p.get<std::string>("Stress Name"),dl->qp_tensor)
{
  this->addDependentField(def_grad_);
  this->addDependentField(stress_);
  this->addEvaluatedField(residual_);
  this->setName("ConstitutiveModelDriver" + PHX::TypeString<EvalT>::value);
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ConstitutiveModelDriver<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(def_grad_, fm);
  this->utils.setFieldData(stress_, fm);
  this->utils.setFieldData(residual_, fm);
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ConstitutiveModelDriver<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::cout << "ConstitutiveModelDriver<EvalT, Traits>::evaluateFields" << std::endl;
  Intrepid::Tensor<ScalarT> F(num_dims_), P(num_dims_), sig(num_dims_);

  Intrepid::Tensor<ScalarT> F0(num_dims_), P0(num_dims_);

  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t pt = 0; pt < num_pts_; ++pt) {
      F.fill(&def_grad_(cell,pt,0,0));
      sig.fill(&stress_(cell,pt,0,0));
      P = Intrepid::piola(F,sig);
      for (std::size_t node = 0; node < num_nodes_; ++node) {
        for (std::size_t dim1 = 0; dim1 < num_dims_; ++dim1) {
          for (std::size_t dim2 = 0; dim2 < num_dims_; ++dim2) {
            residual_(cell,node,dim1,dim2) = 
              (F(dim1,dim2) - F0(dim1,dim2)) * 
              (P(dim1,dim2) - P0(dim1,dim2));
          }
        }
      }
    }
  }

}

//------------------------------------------------------------------------------

}

