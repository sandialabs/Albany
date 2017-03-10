//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
DOFVecInterpolationLevels<EvalT, Traits>::
DOFVecInterpolationLevels(Teuchos::ParameterList& p,
                 const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_vector_level),
  BF          (p.get<std::string>   ("BF Name"),       dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), dl->qp_vector_level),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("Aeras::DOFVecInterpolationLevels"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFVecInterpolationLevels<EvalT, Traits>::
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
void DOFVecInterpolationLevels<EvalT, Traits>::
operator() (const int cell, const int qp, const int level) const{
  if (numDims==2){
    ScalarT val_qp_0 =0;
    ScalarT val_qp_1=0;
    for (int node=0; node < numNodes; ++node) {
          val_qp_0 += val_node(cell,node,level,0) * BF(cell,node,qp);
          val_qp_1 += val_node(cell,node,level,1) * BF(cell,node,qp);
    }
    val_qp(cell,qp,level,0)=val_qp_0; 
    val_qp(cell,qp,level,1)=val_qp_1;

  }else{
      for (int dim=0; dim < numDims; ++dim) {
        ScalarT temp=0;
        for (int node=0; node < numNodes; ++node) {
          temp += val_node(cell,node,level,dim) * BF(cell,node,qp);
        }
        val_qp(cell,qp,level,dim)=temp;
      }
  }//endif
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFVecInterpolationLevels<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) {
        for (int dim=0; dim < numDims; ++dim) {
          val_qp(cell,qp,level,dim) = 0;
          for (int node=0; node < numNodes; ++node) {
            val_qp(cell,qp,level,dim) += val_node(cell,node,level,dim) * BF(cell,node,qp);
          }
        }
      }
    }
  }

#else
  DOFVecInterpolationLevels_Policy range(
                {0,0,0}, {(int)workset.numCells,(int)numQPs,(int)numLevels}, DOFVecInterpolationLevels_TileSize);
  Kokkos::Experimental::md_parallel_for(range,*this);

#endif
}
}
