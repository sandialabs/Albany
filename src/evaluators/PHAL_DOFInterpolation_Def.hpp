//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
DOFInterpolation<EvalT, Traits>::
DOFInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  BF          (p.get<std::string>   ("BF Name"), dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), dl->qp_scalar )
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("DOFInterpolation" );

  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFInterpolation<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFInterpolation<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid2 version:
  // for (int i=0; i < val_qp.size() ; i++) val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(val_qp, val_node, BF);

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      //ScalarT& vqp = val_qp(cell,qp);
      val_qp(cell,qp) = val_node(cell, 0) * BF(cell, 0, qp);
      for (std::size_t node=1; node < numNodes; ++node) {
        val_qp(cell,qp) += val_node(cell, node) * BF(cell, node, qp);
      }
    }
  }
}

//**********************************************************************
template<typename Traits>
DOFInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
DOFInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  BF          (p.get<std::string>   ("BF Name"), dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), dl->qp_scalar )
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("DOFInterpolation Jacobian");

  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename Traits>
void DOFInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid2 version:
  // for (int i=0; i < val_qp.size() ; i++) val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(val_qp, val_node, BF);

  const int num_dof = val_node(0,0).size();
  const int neq = workset.wsElNodeEqID[0][0].size();

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      //ScalarT& vqp = val_qp(cell,qp);
      val_qp(cell,qp) = ScalarT(num_dof, val_node(cell, 0).val() * BF(cell, 0, qp));
      if (num_dof) (val_qp(cell,qp)).fastAccessDx(offset) = val_node(cell, 0).fastAccessDx(offset) * BF(cell, 0, qp);
      for (std::size_t node=1; node < numNodes; ++node) {
        (val_qp(cell,qp)).val() += val_node(cell, node).val() * BF(cell, node, qp);
        if (num_dof) (val_qp(cell,qp)).fastAccessDx(neq*node+offset) += val_node(cell, node).fastAccessDx(neq*node+offset) * BF(cell, node, qp);
      }
    }
  }
}

#ifdef ALBANY_SG
//**********************************************************************
template<typename Traits>
DOFInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits>::
DOFInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  BF          (p.get<std::string>   ("BF Name"), dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), dl->qp_scalar )
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("DOFInterpolation SGJacobian");

  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename Traits>
void DOFInterpolation<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid2 version:
  // for (int i=0; i < val_qp.size() ; i++) val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(val_qp, val_node, BF);

  const int num_dof = val_node(0,0).size();
  const int neq = workset.wsElNodeEqID[0][0].size();

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      //ScalarT& vqp = val_qp(cell,qp);
      val_qp(cell,qp) = ScalarT(num_dof, val_node(cell, 0).val() * BF(cell, 0, qp));
      if (num_dof) (val_qp(cell,qp)).fastAccessDx(offset) = val_node(cell, 0).fastAccessDx(offset) * BF(cell, 0, qp);
      for (std::size_t node=1; node < numNodes; ++node) {
        (val_qp(cell,qp)).val() += val_node(cell, node).val() * BF(cell, node, qp);
        if (num_dof) (val_qp(cell,qp)).fastAccessDx(neq*node+offset) += val_node(cell, node).fastAccessDx(neq*node+offset) * BF(cell, node, qp);
      }
    }
  }
}
#endif

#ifdef ALBANY_ENSEMBLE
//**********************************************************************
template<typename Traits>
DOFInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits>::
DOFInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  BF          (p.get<std::string>   ("BF Name"), dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), dl->qp_scalar )
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("DOFInterpolation MPJacobian");

  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename Traits>
void DOFInterpolation<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid2 version:
  // for (int i=0; i < val_qp.size() ; i++) val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(val_qp, val_node, BF);

  const int num_dof = val_node(0,0).size();
  const int neq = workset.wsElNodeEqID[0][0].size();

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      //ScalarT& vqp = val_qp(cell,qp);
      val_qp(cell,qp) = ScalarT(num_dof, val_node(cell, 0).val() * BF(cell, 0, qp));
      if (num_dof) (val_qp(cell,qp)).fastAccessDx(offset) = val_node(cell, 0).fastAccessDx(offset) * BF(cell, 0, qp);
      for (std::size_t node=1; node < numNodes; ++node) {
        (val_qp(cell,qp)).val() += val_node(cell, node).val() * BF(cell, node, qp);
        if (num_dof) (val_qp(cell,qp)).fastAccessDx(neq*node+offset) += val_node(cell, node).fastAccessDx(neq*node+offset) * BF(cell, node, qp);
      }
    }
  }
}
#endif

//**********************************************************************

}

