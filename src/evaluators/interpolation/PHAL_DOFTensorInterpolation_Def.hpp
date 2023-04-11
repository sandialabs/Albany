//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_AbstractDiscretization.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
DOFTensorInterpolationBase<EvalT, Traits, ScalarT>::
DOFTensorInterpolationBase(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>  ("Variable Name"), dl->node_tensor),
  BF          (p.get<std::string>  ("BF Name"),  dl->node_qp_scalar),
  val_qp      (p.get<std::string>  ("Variable Name"), dl->qp_tensor)
{
  this->addDependentField(val_node.fieldTag());
  this->addDependentField(BF.fieldTag());
  this->addEvaluatedField(val_qp);

  this->setName("DOFTensorInterpolationBase"+PHX::print<EvalT>());
  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

  val_node.fieldTag().dataLayout().dimensions(dims);
  vecDim   = dims[2];
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFTensorInterpolationBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFTensorInterpolationBase<EvalT, Traits, ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t i=0; i<vecDim; i++) {
        for (std::size_t j=0; j<vecDim; j++) {
          // Zero out for node==0; then += for node = 1 to numNodes
          typename PHAL::Ref<ScalarT>::type vqp = val_qp(cell,qp,i,j);
          vqp = val_node(cell, 0, i, j) * BF(cell, 0, qp);
          for (std::size_t node=1; node < numNodes; ++node) {
            vqp += val_node(cell, node, i, j) * BF(cell, node, qp);
          }
        }
      }
    }
  }
}

//**********************************************************************
//! Specialization for Jacobian evaluation taking advantage of known sparsity

template<typename Traits>
void FastSolutionTensorInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  const int num_dof = this->val_node(0,0,0,0).size();
  const int neq = workset.disc->getDOFManager()->getNumFields();
  const auto vecDim = this->vecDim;
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < this->numQPs; ++qp) {
      for (std::size_t i=0; i< vecDim; i++) {
        for (std::size_t j=0; j< vecDim; j++) {
          // Zero out for node==0; then += for node = 1 to numNodes
          typename PHAL::Ref<ScalarT>::type vqp = this->val_qp(cell,qp,i,j);

          vqp = this->val_node(cell, 0, i, j) * this->BF(cell, 0, qp);
          vqp = ScalarT(num_dof, this->val_node(cell, 0, i, j).val() * this->BF(cell, 0, qp));
          vqp.fastAccessDx(offset+i*vecDim+j) = this->val_node(cell, 0, i, j).fastAccessDx(offset+i*vecDim+j) * this->BF(cell, 0, qp);
          for (std::size_t node=1; node < this->numNodes; ++node) {
            vqp.val() += this->val_node(cell, node, i, j).val() * this->BF(cell, node, qp);
            vqp.fastAccessDx(neq*node+offset+i*this->vecDim+j)
              += this->val_node(cell, node, i, j).fastAccessDx(neq*node+offset+i*vecDim+j) * this->BF(cell, node, qp);
          }
        }
      }
    }
  }
}
//**********************************************************************

} // Namespace PHAL
