//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
TPSALaplaceResid<EvalT, Traits>::
TPSALaplaceResid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  GradBF        (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient),
  wGradBF       (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  solnVec(p.get<std::string> ("Solution Vector Name"), dl->node_vector),
  solnResidual(p.get<std::string> ("Residual Name"), dl->node_vector) {


  this->addDependentField(GradBF);
  this->addDependentField(wGradBF);
  this->addDependentField(solnVec);
  this->addEvaluatedField(solnResidual);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_vector->dimensions(dims);
  worksetSize = dims[0];
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  this->setName("LaplaceResid" + PHX::typeAsString<EvalT>());

}

//**********************************************************************
template<typename EvalT, typename Traits>
void TPSALaplaceResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm) {
  this->utils.setFieldData(GradBF, fm);
  this->utils.setFieldData(wGradBF, fm);
  this->utils.setFieldData(solnVec, fm);
  this->utils.setFieldData(solnResidual, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void TPSALaplaceResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset) {

   // Straight Laplace's equation evaluation for the nodal coord solution

    for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for(std::size_t node_a = 0; node_a < numNodes; ++node_a) {

        for(std::size_t eq = 0; eq < numDims; eq++)  {
          solnResidual(cell, node_a, eq) = 0.0;
        }

        for(std::size_t qp = 0; qp < numQPs; ++qp) {
          for(std::size_t node_b = 0; node_b < numNodes; ++node_b) {

            ScalarT kk = 0.0;

            for(std::size_t i = 0; i < numDims; i++) {

              kk += GradBF(cell, node_a, qp, i) * wGradBF(cell, node_b, qp, i);

            }

            for(std::size_t eq = 0; eq < numDims; eq++) {

              solnResidual(cell, node_a, eq) += kk * solnVec(cell, node_b, eq);

            }
          }
        }
      }
    }
}

//**********************************************************************
}

