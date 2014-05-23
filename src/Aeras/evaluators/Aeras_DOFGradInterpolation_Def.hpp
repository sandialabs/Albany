//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
DOFGradInterpolation<EvalT, Traits>::
DOFGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar_level),
  GradBF      (p.get<std::string>   ("Gradient BF Name"), dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient_level)
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolation"+PHX::TypeString<EvalT>::value);

  std::vector<PHX::DataLayout::size_type> dims;
  GradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
  numLevels = p.get< int >("Number of Vertical Levels");
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolation<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolation<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

    for (int cell=0; cell < workset.numCells; ++cell) {
        for (int qp=0; qp < numQPs; ++qp) {
          for (int level=0; level < numLevels; ++level) {
            for (int dim=0; dim<numDims; dim++) {
              ScalarT& gvqp = grad_val_qp(cell,qp,level,dim);
              gvqp = val_node(cell, 0, level) * GradBF(cell, 0, qp, dim);
              for (int node= 1 ; node < numNodes; ++node) {
                gvqp += val_node(cell, node, level) * GradBF(cell, node, qp, dim);
            }
          }
        }
      }
    }
}

//**********************************************************************
template<typename Traits>
DOFGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
DOFGradInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar_level),
  GradBF      (p.get<std::string>   ("Gradient BF Name"), dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient_level)
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolation"+PHX::TypeString<PHAL::AlbanyTraits::Jacobian>::value);

  std::vector<PHX::DataLayout::size_type> dims;
  GradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
  numLevels = p.get< int >("Number of Vertical Levels");
}

//**********************************************************************
template<typename Traits>
void DOFGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}

//**********************************************************************
template<typename Traits>
void DOFGradInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

  int num_dof = val_node(0,0,0).size();
  int neq = num_dof / numNodes;

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int level=0; level < numLevels; ++level) {
        for (int dim=0; dim<numDims; dim++) {
          ScalarT& gvqp = grad_val_qp(cell,qp,level,dim);
          gvqp = FadType(num_dof, val_node(cell, 0, level).val() * GradBF(cell, 0, qp, dim));
          for (int node= 1 ; node < numNodes; ++node) 
            gvqp.val() += val_node(cell, node, level).val() * GradBF(cell, node, qp, dim);
          if (gvqp.hasFastAccess()) {
            gvqp.fastAccessDx(0) = val_node(cell, 0, level).fastAccessDx(0) * GradBF(cell, 0, qp, dim);
            for (int node= 1 ; node < numNodes; ++node) {
              gvqp.fastAccessDx(neq*node) += val_node(cell, node, level).fastAccessDx(neq*node) * GradBF(cell, node, qp, dim);
            }
          }
        }
      }
    }
  }
}

//**********************************************************************
//**********************************************************************

template<typename EvalT, typename Traits>
DOFGradInterpolation_noDeriv<EvalT, Traits>::
DOFGradInterpolation_noDeriv(const Teuchos::ParameterList& p,
                             const Teuchos::RCP<Aeras::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar_level),
  GradBF      (p.get<std::string>   ("Gradient BF Name"), dl->node_qp_gradient),
  grad_val_qp (p.get<std::string>   ("Gradient Variable Name"), dl->qp_gradient_level)
{
  this->addDependentField(val_node);
  this->addDependentField(GradBF);
  this->addEvaluatedField(grad_val_qp);

  this->setName("DOFGradInterpolation_noDeriv"+PHX::TypeString<EvalT>::value);

  std::vector<PHX::DataLayout::size_type> dims;
  GradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
  numLevels = p.get< int >("Number of Vertical Levels");
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolation_noDeriv<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(grad_val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFGradInterpolation_noDeriv<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid Version:
  // for (int i=0; i < grad_val_qp.size() ; i++) grad_val_qp[i] = 0.0;
  // Intrepid::FunctionSpaceTools:: evaluate<ScalarT>(grad_val_qp, val_node, GradBF);

    for (int cell=0; cell < workset.numCells; ++cell) {
        for (int qp=0; qp < numQPs; ++qp) {
          for (int level=0; level < numLevels; ++level) {
            for (int dim=0; dim<numDims; dim++) {
              MeshScalarT& gvqp = grad_val_qp(cell,qp,level,dim);
              gvqp = val_node(cell, 0, level) * GradBF(cell, 0, qp, dim);
              for (int node= 1 ; node < numNodes; ++node) {
                gvqp += val_node(cell, node, level) * GradBF(cell, node, qp, dim);
              }
          }
        }
      }
    }
}

//**********************************************************************

}

