//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

#include "PHAL_Workset.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
DOFInterpolationBase<EvalT, Traits, ScalarT>::
DOFInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  BF          (p.get<std::string>   ("BF Name"), dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), dl->qp_scalar )
{
  this->addDependentField(val_node.fieldTag());
  this->addDependentField(BF.fieldTag());
  this->addEvaluatedField(val_qp);

  this->setName("DOFInterpolationBase" );

  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFInterpolationBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void DOFInterpolationBase<EvalT, Traits, ScalarT>::
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
/*
//**********************************************************************
template<typename Traits>
DOFInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
DOFInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  BF          (p.get<std::string>   ("BF Name"), dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), dl->qp_scalar )
{
  this->addDependentField(val_node.fieldTag());
  this->addDependentField(BF.fieldTag());
  this->addEvaluatedField(val_qp);

  this->setName("DOFInterpolationBase Jacobian");

  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename Traits>
void DOFInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>::
evaluateFields(typename Traits::EvalData workset)
{
  //Intrepid2 version:
  // for (int i=0; i < val_qp.size() ; i++) val_qp[i] = 0.0;
  // Intrepid2::FunctionSpaceTools:: evaluate<ScalarT>(val_qp, val_node, BF);

  const int num_dof = val_node(0,0).size();
  const int neq = workset.wsElNodeEqID[0][0].size();
std::cout << val_node.fieldTag().name() << ", ws " << workset.wsIndex << ":\n";

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      //ScalarT& vqp = val_qp(cell,qp);
      val_qp(cell,qp) = ScalarT(num_dof, val_node(cell, 0).val() * BF(cell, 0, qp));
      if (num_dof) (val_qp(cell,qp)).fastAccessDx(offset) = val_node(cell, 0).fastAccessDx(offset) * BF(cell, 0, qp);
std::cout << "  val_node(" << cell << "," << 0 << ") = " << val_node(cell,0) << "\n";
      for (std::size_t node=1; node < numNodes; ++node) {
        (val_qp(cell,qp)).val() += val_node(cell, node).val() * BF(cell, node, qp);
        if (num_dof) (val_qp(cell,qp)).fastAccessDx(neq*node+offset) += val_node(cell, node).fastAccessDx(neq*node+offset) * BF(cell, node, qp);
std::cout << "  val_node(" << cell << "," << node << ") = " << val_node(cell,node) << "\n";
      }
std::cout << "  val_qp(" << cell << "," << qp << ") = " << val_qp(cell,qp) << "\n";
    }
  }
}
*/
#ifdef ALBANY_SG
//**********************************************************************
template<typename Traits>
DOFInterpolationBase<PHAL::AlbanyTraits::SGJacobian, Traits, typename PHAL::AlbanyTraits::SGJacobian::ScalarT>::
DOFInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  BF          (p.get<std::string>   ("BF Name"), dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), dl->qp_scalar )
{
  this->addDependentField(val_node.fieldTag());
  this->addDependentField(BF.fieldTag());
  this->addEvaluatedField(val_qp);

  this->setName("DOFInterpolationBase SGJacobian");

  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFInterpolationBase<PHAL::AlbanyTraits::SGJacobian, Traits, typename PHAL::AlbanyTraits::SGJacobian::ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename Traits>
void DOFInterpolationBase<PHAL::AlbanyTraits::SGJacobian, Traits, typename PHAL::AlbanyTraits::SGJacobian::ScalarT>::
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
DOFInterpolationBase<PHAL::AlbanyTraits::MPJacobian, Traits, typename PHAL::AlbanyTraits::MPJacobian::ScalarT>::
DOFInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  BF          (p.get<std::string>   ("BF Name"), dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), dl->qp_scalar )
{
  this->addDependentField(val_node.fieldTag());
  this->addDependentField(BF.fieldTag());
  this->addEvaluatedField(val_qp);

  this->setName("DOFInterpolationBase MPJacobian");

  std::vector<PHX::DataLayout::size_type> dims;
  BF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void DOFInterpolationBase<PHAL::AlbanyTraits::MPJacobian, Traits, typename PHAL::AlbanyTraits::MPJacobian::ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename Traits>
void DOFInterpolationBase<PHAL::AlbanyTraits::MPJacobian, Traits, typename PHAL::AlbanyTraits::MPJacobian::ScalarT>::
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

