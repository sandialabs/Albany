//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
ContravariantTargetMetricTensor<EvalT, Traits>::
ContravariantTargetMetricTensor(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :

  solnVec       (p.get<std::string> ("Solution Vector Name"), dl->node_vector),
  cubature      (p.get<Teuchos::RCP <Intrepid2::Cubature<PHX::Device> > >("Cubature")),
  intrepidBasis(p.get<Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > > ("Intrepid2 Basis")),
  Gc            (p.get<std::string> ("Contravariant Metric Tensor Name"), dl->qp_tensor)

{

  this->addDependentField(solnVec);
  this->addEvaluatedField(Gc);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_vector->dimensions(dim);
  containerSize = dim[0];
  numQPs = dim[1];
  numDims = dim[2];

  // Allocate Temporary FieldContainers
  refPoints = Kokkos::DynRankView<RealType, PHX::Device>("refPoints", numQPs, numDims);
  refWeights = Kokkos::DynRankView<RealType, PHX::Device>("refWeights", numQPs);

  // Pre-Calculate reference element quantitites
  cubature->getCubature(refPoints, refWeights);

  this->setName("ContravariantTargetMetricTensor"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ContravariantTargetMetricTensor<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(solnVec, fm);
  this->utils.setFieldData(Gc, fm);
  
  jacobian = Kokkos::createDynRankView(solnVec.get_view(), "jacobian", containerSize, numQPs, numDims, numDims);
  jacobian_inv = Kokkos::createDynRankView(solnVec.get_view(), "jacobian_inv", containerSize, numQPs, numDims, numDims);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ContravariantTargetMetricTensor<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  // Need to be ScalarT!
  Intrepid2::CellTools<PHX::Device>::setJacobian(jacobian, refPoints, solnVec.get_view(), intrepidBasis);
  // Since Intrepid2 will perform calculations on the entire workset size and not
  // just the used portion, we must fill the excess with reasonable values.
  // Leaving this out leads to a floating point exception.
  for (std::size_t cell = workset.numCells; cell < jacobian.dimension(0); ++cell)
    for (std::size_t qp = 0; qp < numQPs; ++qp)
      for (std::size_t i = 0; i < numDims; ++i)
        jacobian(cell, qp, i, i) = 1.0;
  Intrepid2::CellTools<PHX::Device>::setJacobianInv(jacobian_inv, jacobian);

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {      
      for (std::size_t i=0; i < numDims; ++i) {        
        for (std::size_t j=0; j < numDims; ++j) {
          Gc(cell, qp, i, j) = 0.0;
          for (std::size_t alpha=0; alpha < numDims; ++alpha) {  
            Gc(cell, qp, i, j) += jacobian_inv(cell, qp, alpha, i) * jacobian_inv(cell, qp, alpha, j); 
          }
        } 
      } 
    }
  }
  
}

//**********************************************************************
}
