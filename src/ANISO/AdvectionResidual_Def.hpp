//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

namespace ANISO {

template<typename EvalT, typename Traits>
AdvectionResidual<EvalT, Traits>::
AdvectionResidual(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
  w_bf      (p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
  w_grad_bf (p.get<std::string>("Weighted Gradient BF Name"), dl->node_qp_vector),
  phi       (p.get<std::string>("Concentration Name"), dl->qp_scalar),
  grad_phi  (p.get<std::string>("Concentration Gradient Name"), dl->qp_vector),
  residual  (p.get<std::string>("Residual Name"), dl->node_scalar) {

  num_nodes = dl->node_qp_vector->dimension(1);
  num_qps = dl->node_qp_vector->dimension(2);
  num_dims = dl->node_qp_vector->dimension(3);

  kappa = p.get<double>("Kappa");
  alpha = p.get<Teuchos::Array<double> >("Alpha");
  assert(alpha.size() == num_dims);

  this->addDependentField(w_bf);
  this->addDependentField(w_grad_bf);
  this->addDependentField(phi);
  this->addDependentField(grad_phi);
  this->addEvaluatedField(residual);
  this->setName("Advection Residual");
}

template<typename EvalT, typename Traits>
void AdvectionResidual<EvalT, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm) {

  this->utils.setFieldData(w_bf, fm);
  this->utils.setFieldData(w_grad_bf, fm);
  this->utils.setFieldData(phi, fm);
  this->utils.setFieldData(grad_phi, fm);
  this->utils.setFieldData(residual, fm);
}

template<typename EvalT, typename Traits>
void AdvectionResidual<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset) {

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < num_nodes; ++node)
      residual(cell, node) = 0.0;
  }

}

}
