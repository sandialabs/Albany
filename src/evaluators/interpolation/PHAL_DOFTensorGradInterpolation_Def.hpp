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
  DOFTensorGradInterpolationBase<EvalT, Traits, ScalarT>::
  DOFTensorGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_tensor),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_tensorgradient )
  {
    this->addDependentField(val_node.fieldTag());
    this->addDependentField(GradBF.fieldTag());
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFTensorGradInterpolationBase"+PHX::print<EvalT>());

    std::vector<PHX::DataLayout::size_type> dims;
    GradBF.fieldTag().dataLayout().dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    numDims  = dims[3];

    val_node.fieldTag().dataLayout().dimensions(dims);
    vecDim  = dims[2];
  }

  //**********************************************************************
  template<typename EvalT, typename Traits, typename ScalarT>
  void DOFTensorGradInterpolationBase<EvalT, Traits, ScalarT>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }

  //**********************************************************************
  template<typename EvalT, typename Traits, typename ScalarT>
  void DOFTensorGradInterpolationBase<EvalT, Traits, ScalarT>::
  evaluateFields(typename Traits::EvalData workset)
  {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        for (std::size_t i=0; i<vecDim; i++) {
          for (std::size_t j=0; j<vecDim; j++) {
            for (std::size_t dim=0; dim<numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
              //ScalarT& gvqp = grad_val_qp(cell,qp,i,j,dim);
              grad_val_qp(cell,qp,i,j,dim) = val_node(cell, 0, i, j) * GradBF(cell, 0, qp, dim);
              for (std::size_t node= 1 ; node < numNodes; ++node) {
                grad_val_qp(cell,qp,i,j,dim) += val_node(cell, node, i, j) * GradBF(cell, node, qp, dim);
              }
            }
          }
        }
      }
    }
  }

//Specialization for Jacobian evaluation taking advantage of the sparsity of the derivatives

  //**********************************************************************
  template<typename Traits>
  void FastSolutionTensorGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
  evaluateFields(typename Traits::EvalData workset)
  {
    const int num_dof = this->val_node(0,0,0,0).size();
    const int neq = workset.disc->getDOFManager()->getNumFields();
    const auto vecDim = this->vecDim;
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < this->numQPs; ++qp) {
        for (std::size_t i=0; i<vecDim; i++) {
          for (std::size_t j=0; j<vecDim; j++) {
            for (std::size_t dim=0; dim<this->numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
              typename PHAL::Ref<ScalarT>::type gvqp = this->grad_val_qp(cell,qp,i,j,dim);
              gvqp = ScalarT(num_dof, this->val_node(cell, 0, i, j).val() * this->GradBF(cell, 0, qp, dim));
              gvqp.fastAccessDx(offset+i*vecDim+j) = this->val_node(cell, 0, i, j).fastAccessDx(offset+i*vecDim+j) * this->GradBF(cell, 0, qp, dim);
              for (std::size_t node= 1 ; node < this->numNodes; ++node) {
                gvqp.val() += this->val_node(cell, node, i, j).val() * this->GradBF(cell, node, qp, dim);
                gvqp.fastAccessDx(neq*node+offset+i*vecDim+j)
                  += this->val_node(cell, node, i, j).fastAccessDx(neq*node+offset+i*vecDim+j) * this->GradBF(cell, node, qp, dim);
              }
            }
          }
        }
      }
    }
  }

} // Namespace PHAL
