//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
DOFInterpolationLevels<EvalT, Traits>::
DOFInterpolationLevels(Teuchos::ParameterList& p,
                 const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar_level),
  BF          (p.get<std::string>   ("BF Name"),       dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), dl->qp_scalar_level),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("Aeras::DOFInterpolationLevels"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFInterpolationLevels<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void DOFInterpolationLevels<EvalT, Traits>::
operator() (const int cell, const int qp, const int level) const{
 ScalarT vqp = 0;
 for (int node=0; node < numNodes; ++node) {
     vqp += val_node(cell, node, level) * BF(cell, node, qp);
 } 
 val_qp(cell,qp,level)=vqp;    
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFInterpolationLevels<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  //Intrepid2 version:
  // for (int i=0; i < val_qp.size() ; i++) val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(val_qp, val_node, BF);
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) {
        typename PHAL::Ref<ScalarT>::type vqp = val_qp(cell,qp,level) = 0;
        for (int node=0; node < numNodes; ++node) {
          vqp += val_node(cell, node, level) * BF(cell, node, qp);
        }
      }
    }
  }

#else
#if defined(PHX_KOKKOS_DEVICE_TYPE_CUDA)
  DOFInterpolationLevels_Policy range(
                {0,0,0}, {workset.numCells,numNodes,numLevels}, {256,1,1} );
#else
  DOFInterpolationLevels_Policy  range ({(int)workset.numCells,(int)numQPs,(int)numLevels});
#endif
  Kokkos::Experimental::md_parallel_for(range,*this);
#endif
}
}
