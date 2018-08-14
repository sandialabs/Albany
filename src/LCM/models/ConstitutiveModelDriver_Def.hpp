//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
ConstitutiveModelDriver<EvalT, Traits>::ConstitutiveModelDriver(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : residual_(p.get<std::string>("Residual Name"), dl->node_tensor),
      def_grad_(p.get<std::string>("F Name"), dl->qp_tensor),
      stress_(p.get<std::string>("Stress Name"), dl->qp_tensor),
      prescribed_def_grad_(
          p.get<std::string>("Prescribed F Name"),
          dl->qp_tensor)
{
  this->addDependentField(def_grad_);
  this->addDependentField(prescribed_def_grad_);
  this->addDependentField(stress_);
  this->addEvaluatedField(residual_);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_vector->dimensions(dims);
  num_nodes_ = dims[1];
  num_pts_   = dims[2];
  num_dims_  = dims[3];
  this->setName("ConstitutiveModelDriver" + PHX::typeAsString<EvalT>());
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
ConstitutiveModelDriver<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(def_grad_, fm);
  this->utils.setFieldData(prescribed_def_grad_, fm);
  this->utils.setFieldData(stress_, fm);
  this->utils.setFieldData(residual_, fm);
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
ConstitutiveModelDriver<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  bool print = false;
  if (typeid(ScalarT) == typeid(RealType)) print = true;
  std::cout.precision(15);

  std::cout << "ConstitutiveModelDriver<EvalT, Traits>::evaluateFields"
            << std::endl;
  minitensor::Tensor<ScalarT> F(num_dims_), P(num_dims_), sig(num_dims_);

  minitensor::Tensor<ScalarT> F0(num_dims_), P0(num_dims_);

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < num_pts_; ++pt) {
      F0.fill(prescribed_def_grad_, cell, pt, 0, 0);
      F.fill(def_grad_, cell, pt, 0, 0);
      sig.fill(stress_, cell, pt, 0, 0);
      P = minitensor::piola(F, sig);
      if (print) {
        std::cout << "F: \n" << F << std::endl;
        std::cout << "P: \n" << P << std::endl;
        std::cout << "sig: \n" << sig << std::endl;
      }
      for (int node = 0; node < num_nodes_; ++node) {
        for (int dim1 = 0; dim1 < num_dims_; ++dim1) {
          for (int dim2 = 0; dim2 < num_dims_; ++dim2) {
            residual_(cell, node, dim1, dim2) =
                (F(dim1, dim2) - F0(dim1, dim2));
            //* (P(dim1,dim2) - P0(dim1,dim2));
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------

}  // namespace LCM
