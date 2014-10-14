//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

  //**********************************************************************
  template<typename EvalT, typename Traits>
  DOFTensorGradInterpolation<EvalT, Traits>::
  DOFTensorGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_tensor),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_tensorgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFTensorGradInterpolation"+PHX::TypeString<EvalT>::value);

    std::vector<PHX::DataLayout::size_type> dims;
    GradBF.fieldTag().dataLayout().dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    numDims  = dims[3];

    val_node.fieldTag().dataLayout().dimensions(dims);
    vecDim  = dims[2];
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void DOFTensorGradInterpolation<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void DOFTensorGradInterpolation<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        for (std::size_t i=0; i<vecDim; i++) {
          for (std::size_t j=0; j<vecDim; j++) {
            for (std::size_t dim=0; dim<numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
              ScalarT& gvqp = grad_val_qp(cell,qp,i,j,dim);
              gvqp = val_node(cell, 0, i, j) * GradBF(cell, 0, qp, dim);
              for (std::size_t node= 1 ; node < numNodes; ++node) {
                gvqp += val_node(cell, node, i, j) * GradBF(cell, node, qp, dim);
              }
            } 
          } 
        } 
      } 
    }
  }
  
  //**********************************************************************
  template<typename Traits>
  DOFTensorGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
  DOFTensorGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_tensor),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_tensorgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFTensorGradInterpolation"+PHX::TypeString<PHAL::AlbanyTraits::Jacobian>::value);

    std::vector<PHX::DataLayout::size_type> dims;
    GradBF.fieldTag().dataLayout().dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    numDims  = dims[3];

    val_node.fieldTag().dataLayout().dimensions(dims);
    vecDim  = dims[2];

    offset = p.get<int>("Offset of First DOF");
  }

  //**********************************************************************
  template<typename Traits>
  void DOFTensorGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }

  //**********************************************************************
  template<typename Traits>
  void DOFTensorGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
  int num_dof = val_node(0,0,0,0).size();
  int neq = num_dof / numNodes;

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        for (std::size_t i=0; i<vecDim; i++) {
          for (std::size_t j=0; j<vecDim; j++) {
            for (std::size_t dim=0; dim<numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
              ScalarT& gvqp = grad_val_qp(cell,qp,i,j,dim);
              gvqp = FadType(num_dof, val_node(cell, 0, i, j).val() * GradBF(cell, 0, qp, dim));
              gvqp.fastAccessDx(offset+i*vecDim+j) = val_node(cell, 0, i, j).fastAccessDx(offset+i*vecDim+j) * GradBF(cell, 0, qp, dim);
              for (std::size_t node= 1 ; node < numNodes; ++node) {
                gvqp.val() += val_node(cell, node, i, j).val() * GradBF(cell, node, qp, dim);
                gvqp.fastAccessDx(neq*node+offset+i*vecDim+j) 
                  += val_node(cell, node, i, j).fastAccessDx(neq*node+offset+i*vecDim+j) * GradBF(cell, node, qp, dim);
              }
            } 
          } 
        } 
      } 
    }
  }
  
  //**********************************************************************
}
