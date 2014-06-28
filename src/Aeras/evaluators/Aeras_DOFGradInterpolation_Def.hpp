//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
DOFGradInterpolation<EvalT, Traits>::
DOFGradInterpolation(Teuchos::ParameterList& p,
                     const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"),         dl->node_scalar),
  GradBF      (p.get<std::string>   ("Gradient BF Name"),      dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"),dl->qp_gradient), 
  numNodes   (dl->node_scalar             ->dimension(1)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numQPs     (dl->node_qp_scalar          ->dimension(2))
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("Aeras::DOFGradInterpolation"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolation<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolation<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int dim=0; dim<numDims; dim++) {
        ScalarT& gvqp = grad_val_qp(cell,qp,dim) = 0;
        for (int node=0 ; node < numNodes; ++node) {
          gvqp += val_node(cell, node) * GradBF(cell, node, qp, dim);
        }
      }
    }
  }
}

//**********************************************************************
//**********************************************************************

template<typename EvalT, typename Traits>
DOFGradInterpolation_noDeriv<EvalT, Traits>::
DOFGradInterpolation_noDeriv(Teuchos::ParameterList& p,
                             const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"),          dl->node_scalar),
  GradBF      (p.get<std::string>   ("Gradient BF Name"),       dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient), 
  numNodes   (dl->node_scalar             ->dimension(1)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numQPs     (dl->node_qp_scalar          ->dimension(2))
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("Aeras::DOFGradInterpolation_noDeriv"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolation_noDeriv<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolation_noDeriv<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int dim=0; dim<numDims; dim++) {
        MeshScalarT& gvqp = grad_val_qp(cell,qp,dim) = 0;
        for (int node=0 ; node < numNodes; ++node) {
          gvqp += val_node(cell, node) * GradBF(cell, node, qp, dim);
        }
      }
    }
  }
}

//**********************************************************************

}

