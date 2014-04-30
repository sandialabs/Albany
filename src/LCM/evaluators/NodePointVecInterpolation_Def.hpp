//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
NodePointVecInterpolation<EvalT, Traits>::
NodePointVecInterpolation(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
  nodal_value    (p.get<std::string>  ("Variable Name"), dl->node_vector),
  shape_fn          (p.get<std::string>  ("BF Name"),  dl->node_qp_scalar),
  val_qp      (p.get<std::string>  ("Variable Name"), dl->qp_vector)
{
  this->addDependentField(nodal_value);
  this->addDependentField(shape_fn);
  this->addEvaluatedField(val_qp);

  this->setName("NodePointVecInterpolation"+PHX::TypeString<EvalT>::value);
  std::vector<PHX::DataLayout::size_type> dims;
  shape_fn.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

  nodal_value.fieldTag().dataLayout().dimensions(dims);
  vecDim   = dims[2];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NodePointVecInterpolation<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(nodal_value,fm);
  this->utils.setFieldData(shape_fn,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NodePointVecInterpolation<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t i=0; i<vecDim; i++) {
        // Zero out for node==0; then += for node = 1 to numNodes
        ScalarT& vqp = val_qp(cell,qp,i);
        vqp = nodal_value(cell, 0, i) * shape_fn(cell, 0, qp);
        for (std::size_t node=1; node < numNodes; ++node) {
          vqp += nodal_value(cell, node, i) * shape_fn(cell, node, qp);
        }
      }
    }
  }
//  Intrepid::FunctionSpaceTools::evaluate<ScalarT>(val_qp, nodal_value, shape_fn);
}

//**********************************************************************
template<typename Traits>
NodePointVecInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
NodePointVecInterpolation(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
  nodal_value    (p.get<std::string>  ("Variable Name"), dl->node_vector),
  shape_fn          (p.get<std::string>  ("BF Name"),  dl->node_qp_scalar),
  val_qp      (p.get<std::string>  ("Variable Name"), dl->qp_vector)
{
  this->addDependentField(nodal_value);
  this->addDependentField(shape_fn);
  this->addEvaluatedField(val_qp);

  this->setName("NodePointVecInterpolation"+PHX::TypeString<PHAL::AlbanyTraits::Jacobian>::value);
  std::vector<PHX::DataLayout::size_type> dims;
  shape_fn.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];

  nodal_value.fieldTag().dataLayout().dimensions(dims);
  vecDim   = dims[2];

  offset = p.get<int>("Offset of First DOF");
}

//**********************************************************************
template<typename Traits>
void NodePointVecInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(nodal_value,fm);
  this->utils.setFieldData(shape_fn,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename Traits>
void NodePointVecInterpolation<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int num_dof = nodal_value(0,0,0).size();
  int neq = num_dof / numNodes;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t i=0; i<vecDim; i++) {
        // Zero out for node==0; then += for node = 1 to numNodes
        ScalarT& vqp = val_qp(cell,qp,i);
	vqp = FadType(num_dof, nodal_value(cell, 0, i).val() * shape_fn(cell, 0, qp));
        vqp.fastAccessDx(offset+i) = nodal_value(cell, 0, i).fastAccessDx(offset+i) * shape_fn(cell, 0, qp);
        for (std::size_t node=1; node < numNodes; ++node) {
          vqp.val() += nodal_value(cell, node, i).val() * shape_fn(cell, node, qp);
          vqp.fastAccessDx(neq*node+offset+i) += nodal_value(cell, node, i).fastAccessDx(neq*node+offset+i) * shape_fn(cell, node, qp);
        }
      }
    }
  }
//  Intrepid::FunctionSpaceTools::evaluate<ScalarT>(val_qp, nodal_value, shape_fn);
}
}
