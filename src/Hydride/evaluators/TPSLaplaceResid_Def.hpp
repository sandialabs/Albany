//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Sacado.hpp"
#include "Sacado_Traits.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
TPSLaplaceResid<EvalT, Traits>::
TPSLaplaceResid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
  solnVec(p.get<std::string> ("Solution Vector Name"), dl->node_vector),
  cubature(p.get<Teuchos::RCP <Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > >("Cubature")),
  cellType(p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type")),
  intrepidBasis(p.get<Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > > ("Intrepid2 Basis")),
  solnResidual(p.get<std::string> ("Residual Name"), dl->node_vector) {


  this->addDependentField(solnVec);
  this->addEvaluatedField(solnResidual);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_vector->dimensions(dims);
  worksetSize = dims[0];
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  // Allocate Temporary FieldContainers
  grad_at_cub_points.resize(numNodes, numQPs, numDims);
  refPoints.resize(numQPs, numDims);
  refWeights.resize(numQPs);
  jacobian.resize(worksetSize, numQPs, numDims, numDims);
  jacobian_det.resize(worksetSize, numQPs);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);
  intrepidBasis->getValues(grad_at_cub_points, refPoints, Intrepid2::OPERATOR_GRAD);

  this->setName("LaplaceResid" + PHX::typeAsString<EvalT>());

}

//**********************************************************************
template<typename EvalT, typename Traits>
void TPSLaplaceResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm) {
  this->utils.setFieldData(solnVec, fm);
  this->utils.setFieldData(solnResidual, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void TPSLaplaceResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset) {

  // Note that all the integration operations are ScalarT! We need the partials of the Jacobian to show up in the
  // system Jacobian as the solution is a function of coordinates (it IS the coordinates!).

  // This adds significant time to the compile

  Intrepid2::CellTools<ScalarT>::setJacobian(jacobian, refPoints, solnVec, *cellType);

  // Since Intrepid2 will perform calculations on the entire workset size and not
  // just the used portion, we must fill the excess with reasonable values.
  // Leaving this out leads to a floating point exception in
  //   Intrepid2::RealSpaceTools<Scalar>::det(ArrayDet & detArray,
  //                                         const ArrayIn & inMats).
  for (std::size_t cell = workset.numCells; cell < worksetSize; ++cell)
    for (std::size_t qp = 0; qp < numQPs; ++qp)
      for (std::size_t i = 0; i < numDims; ++i)
        jacobian(cell, qp, i, i) = 1.0;

  Intrepid2::CellTools<ScalarT>::setJacobianDet(jacobian_det, jacobian);

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

              kk += grad_at_cub_points(node_a, qp, i) * grad_at_cub_points(node_b, qp, i);

            }

            for(std::size_t eq = 0; eq < numDims; eq++) {

              solnResidual(cell, node_a, eq) +=
                kk * solnVec(cell, node_b, eq) * jacobian_det(cell, qp) * refWeights(qp);

            }
          }
        }
      }
    }
}

//**********************************************************************
}

