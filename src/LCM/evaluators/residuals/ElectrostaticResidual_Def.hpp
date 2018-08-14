//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include <Phalanx_DataLayout.hpp>
#include <Sacado_ParameterRegistration.hpp>
#include <Teuchos_TestForException.hpp>

namespace LCM {

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
ElectrostaticResidual<EvalT, Traits>::ElectrostaticResidual(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : edisp_(p.get<std::string>("Electric Displacement Name"), dl->qp_vector),
      w_grad_bf_(
          p.get<std::string>("Weighted Gradient BF Name"),
          dl->node_qp_vector),
      residual_(p.get<std::string>("Residual Name"), dl->node_scalar)
{
  this->addDependentField(edisp_);
  this->addDependentField(w_grad_bf_);

  this->addEvaluatedField(residual_);

  this->setName("ElectrostaticResidual" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  w_grad_bf_.fieldTag().dataLayout().dimensions(dims);
  num_nodes_ = dims[1];
  num_pts_   = dims[2];
  num_dims_  = dims[3];
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
ElectrostaticResidual<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(edisp_, fm);
  this->utils.setFieldData(w_grad_bf_, fm);
  this->utils.setFieldData(residual_, fm);
}

// ***************************************************************************
template <typename EvalT, typename Traits>
void
ElectrostaticResidual<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < num_nodes_; ++node)
      residual_(cell, node) = ScalarT(0);
    for (int pt = 0; pt < num_pts_; ++pt)
      for (int node = 0; node < num_nodes_; ++node)
        for (int i = 0; i < num_dims_; ++i)
          residual_(cell, node) +=
              edisp_(cell, pt, i) * w_grad_bf_(cell, node, pt, i);
  }
}
//------------------------------------------------------------------------------
}  // namespace LCM
