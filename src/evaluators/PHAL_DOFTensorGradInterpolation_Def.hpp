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
  template<typename EvalT, typename Traits, typename ScalarT>
  DOFTensorGradInterpolationBase<EvalT, Traits, ScalarT>::
  DOFTensorGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_tensor),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_tensorgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFTensorGradInterpolationBase"+PHX::typeAsString<EvalT>());

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

  //**********************************************************************
  template<typename Traits>
  DOFTensorGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
  DOFTensorGradInterpolationBase(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_tensor),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_tensorgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFTensorGradInterpolationBase"+PHX::typeAsString<PHAL::AlbanyTraits::Jacobian>());

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
  void DOFTensorGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }

  //**********************************************************************
  template<typename Traits>
  void DOFTensorGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
  evaluateFields(typename Traits::EvalData workset)
  {
    const int num_dof = val_node(0,0,0,0).size();
    const int neq = workset.wsElNodeEqID[0][0].size();
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        for (std::size_t i=0; i<vecDim; i++) {
          for (std::size_t j=0; j<vecDim; j++) {
            for (std::size_t dim=0; dim<numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
              typename PHAL::Ref<ScalarT>::type gvqp = grad_val_qp(cell,qp,i,j,dim);
#ifdef ALBANY_MESH_DEPENDS_ON_SOLUTION
              gvqp = val_node(cell, 0, i, j) * GradBF(cell, 0, qp, dim);
#else
              gvqp = ScalarT(num_dof, val_node(cell, 0, i, j).val() * GradBF(cell, 0, qp, dim));
              gvqp.fastAccessDx(offset+i*vecDim+j) = val_node(cell, 0, i, j).fastAccessDx(offset+i*vecDim+j) * GradBF(cell, 0, qp, dim);
#endif
              for (std::size_t node= 1 ; node < numNodes; ++node) {
#ifdef ALBANY_MESH_DEPENDS_ON_SOLUTION
                gvqp += val_node(cell, node, i, j) * GradBF(cell, node, qp, dim);
#else
                gvqp.val() += val_node(cell, node, i, j).val() * GradBF(cell, node, qp, dim);
                gvqp.fastAccessDx(neq*node+offset+i*vecDim+j)
                  += val_node(cell, node, i, j).fastAccessDx(neq*node+offset+i*vecDim+j) * GradBF(cell, node, qp, dim);
#endif
              }
            }
          }
        }
      }
    }
  }

#ifdef ALBANY_SG
  //**********************************************************************
  template<typename Traits>
  DOFTensorGradInterpolationBase<PHAL::AlbanyTraits::SGJacobian, Traits, typename PHAL::AlbanyTraits::SGJacobian::ScalarT>::
  DOFTensorGradInterpolationBase(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_tensor),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_tensorgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFTensorGradInterpolationBase"+PHX::typeAsString<PHAL::AlbanyTraits::SGJacobian>());

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
  void DOFTensorGradInterpolationBase<PHAL::AlbanyTraits::SGJacobian, Traits, typename PHAL::AlbanyTraits::SGJacobian::ScalarT>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }

  //**********************************************************************
  template<typename Traits>
  void DOFTensorGradInterpolationBase<PHAL::AlbanyTraits::SGJacobian, Traits, typename PHAL::AlbanyTraits::SGJacobian::ScalarT>::
  evaluateFields(typename Traits::EvalData workset)
  {
    const int num_dof = val_node(0,0,0,0).size();
    const int neq = workset.wsElNodeEqID[0][0].size();
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        for (std::size_t i=0; i<vecDim; i++) {
          for (std::size_t j=0; j<vecDim; j++) {
            for (std::size_t dim=0; dim<numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
              typename PHAL::Ref<ScalarT>::type gvqp = grad_val_qp(cell,qp,i,j,dim);
              gvqp = ScalarT(num_dof, val_node(cell, 0, i, j).val() * GradBF(cell, 0, qp, dim));
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
#endif

#ifdef ALBANY_ENSEMBLE
  //**********************************************************************
  template<typename Traits>
  DOFTensorGradInterpolationBase<PHAL::AlbanyTraits::MPJacobian, Traits, typename PHAL::AlbanyTraits::MPJacobian::ScalarT>::
  DOFTensorGradInterpolationBase(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl) :
    val_node    (p.get<std::string>  ("Variable Name"), dl->node_tensor),
    GradBF      (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient ),
    grad_val_qp (p.get<std::string>  ("Gradient Variable Name"), dl->qp_tensorgradient )
  {
    this->addDependentField(val_node);
    this->addDependentField(GradBF);
    this->addEvaluatedField(grad_val_qp);

    this->setName("DOFTensorGradInterpolationBase"+PHX::typeAsString<PHAL::AlbanyTraits::MPJacobian>());

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
  void DOFTensorGradInterpolationBase<PHAL::AlbanyTraits::MPJacobian, Traits, typename PHAL::AlbanyTraits::MPJacobian::ScalarT>::
  postRegistrationSetup(typename Traits::SetupData d,
                        PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(val_node,fm);
    this->utils.setFieldData(GradBF,fm);
    this->utils.setFieldData(grad_val_qp,fm);
  }

  //**********************************************************************
  template<typename Traits>
  void DOFTensorGradInterpolationBase<PHAL::AlbanyTraits::MPJacobian, Traits, typename PHAL::AlbanyTraits::MPJacobian::ScalarT>::
  evaluateFields(typename Traits::EvalData workset)
  {
    const int num_dof = val_node(0,0,0,0).size();
    const int neq = workset.wsElNodeEqID[0][0].size();
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        for (std::size_t i=0; i<vecDim; i++) {
          for (std::size_t j=0; j<vecDim; j++) {
            for (std::size_t dim=0; dim<numDims; dim++) {
              // For node==0, overwrite. Then += for 1 to numNodes.
              typename PHAL::Ref<ScalarT>::type gvqp = grad_val_qp(cell,qp,i,j,dim);
              gvqp = ScalarT(num_dof, val_node(cell, 0, i, j).val() * GradBF(cell, 0, qp, dim));
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
#endif

} // Namespace PHAL
