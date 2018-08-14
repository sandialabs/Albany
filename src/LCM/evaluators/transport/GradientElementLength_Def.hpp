//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Phalanx_DataLayout.hpp>
#include <Teuchos_TestForException.hpp>

namespace LCM {

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
GradientElementLength<EvalT, Traits>::GradientElementLength(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : grad_bf_(p.get<std::string>("Gradient BF Name"), dl->node_qp_vector),
      unit_grad_(
          p.get<std::string>("Unit Gradient QP Variable Name"),
          dl->qp_vector),
      element_length_(p.get<std::string>("Element Length Name"), dl->qp_scalar)
{
  this->addDependentField(unit_grad_);
  this->addDependentField(grad_bf_);
  this->addEvaluatedField(element_length_);

  this->setName("Element Length" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_vector->dimensions(dims);
  num_nodes_ = dims[1];
  num_pts_   = dims[2];
  num_dims_  = dims[3];
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
GradientElementLength<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(grad_bf_, fm);
  this->utils.setFieldData(unit_grad_, fm);
  this->utils.setFieldData(element_length_, fm);
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
GradientElementLength<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  ScalarT scalar_h(0.0);

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      scalar_h = 0.0;
      for (int j(0); j < num_dims_; ++j) {
        for (int node(0); node < num_nodes_; ++node) {
          scalar_h +=
              std::abs(grad_bf_(cell, node, pt, j) / std::sqrt(num_dims_));
        }
      }
      element_length_(cell, pt) = 2.0 / scalar_h;
    }
  }
}
//----------------------------------------------------------------------------
}  // namespace LCM
