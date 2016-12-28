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
AdvectionTau<EvalT, Traits>::
AdvectionTau(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
  grad_bf   (p.get<std::string>("Gradient BF Name"), dl->node_qp_vector),
  tau       (p.get<std::string>("Tau Name"), dl->qp_scalar) {

  num_nodes = dl->node_qp_vector->dimension(1);
  num_qps = dl->node_qp_vector->dimension(2);
  num_dims = dl->node_qp_vector->dimension(3);

  kappa = p.get<double>("Kappa");
  alpha = p.get<Teuchos::Array<double> >("Alpha");
  assert(alpha.size() == num_dims);

  this->addDependentField(grad_bf);
  this->addEvaluatedField(tau);
  this->setName("Advection Residual");
}

template<typename EvalT, typename Traits>
void AdvectionTau<EvalT, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm) {

  this->utils.setFieldData(grad_bf, fm);
  this->utils.setFieldData(tau, fm);
}

template<typename EvalT, typename Traits>
void AdvectionTau<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset) {

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < num_qps; ++qp) {

      // compute the element length
      ScalarT h = 0.0;
      for (int node=0; node < num_nodes; ++node)
      for (int dim=0; dim < num_dims; ++dim)
        h += std::abs(grad_bf(cell,node,qp,dim)/std::sqrt(num_dims));
      h = 2.0/h;

      // get the magnitude of the alpha vector
      ScalarT am = 0.0;
      for (int dim=0; dim < num_dims; ++dim)
        am += alpha[dim]*alpha[dim];
      am = std::sqrt(am);

      // compute tau
      ScalarT a = h*am/(2.0*kappa);
      tau(cell, qp) = h/(2.0*am)*(1.0/std::tanh(a)-1.0/a);
    }
  }

}

}
