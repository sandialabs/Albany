//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
ConstitutiveModelDriverPre<EvalT, Traits>::ConstitutiveModelDriverPre(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : solution_(p.get<std::string>("Solution Name"), dl->node_tensor),
      def_grad_(p.get<std::string>("F Name"), dl->qp_tensor),
      j_(p.get<std::string>("J Name"), dl->qp_scalar),
      prescribed_def_grad_(
          p.get<std::string>("Prescribed F Name"),
          dl->qp_tensor),
      time_(p.get<std::string>("Time Name"), dl->workset_scalar)
{
  std::cout << "ConstitutiveModelDriverPre<EvalT, Traits>::constructor"
            << std::endl;
  Teuchos::ParameterList driver_params =
      p.get<Teuchos::ParameterList>("Driver Params");
  std::string loading_case =
      driver_params.get<std::string>("loading case", "uniaxial-strain");
  double increment      = driver_params.get<double>("increment", 0.1);
  final_time_           = driver_params.get<double>("final time", 1.0);
  std::string component = driver_params.get<std::string>("component", "00");

  this->addDependentField(solution_);
  this->addDependentField(time_);
  this->addEvaluatedField(def_grad_);
  this->addEvaluatedField(j_);
  this->addEvaluatedField(prescribed_def_grad_);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->node_qp_vector->dimensions(dims);
  num_nodes_ = dims[1];
  num_pts_   = dims[2];
  num_dims_  = dims[3];
  this->setName("ConstitutiveModelDriverPre" + PHX::typeAsString<EvalT>());

  // F0 is the total prescribed deformation gradient
  F0_ = computeLoading(loading_case, increment);
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
ConstitutiveModelDriverPre<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(solution_, fm);
  this->utils.setFieldData(time_, fm);
  this->utils.setFieldData(def_grad_, fm);
  this->utils.setFieldData(j_, fm);
  this->utils.setFieldData(prescribed_def_grad_, fm);
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
ConstitutiveModelDriverPre<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  std::cout << "ConstitutiveModelDriverPre<EvalT, Traits>::evaluateFields"
            << std::endl;
  minitensor::Tensor<ScalarT> F(num_dims_);

  RealType alpha = Sacado::ScalarValue<ScalarT>::eval(time_(0)) / final_time_;
  minitensor::Tensor<ScalarT> log_F_tensor        = minitensor::log(F0_);
  minitensor::Tensor<ScalarT> scaled_log_F_tensor = alpha * log_F_tensor;
  minitensor::Tensor<ScalarT> current_F = minitensor::exp(scaled_log_F_tensor);

  // FIXME, I really need to figure out which components are prescribed, and
  // which will be traction free, and assign the def_grad_ only for the
  // prescribed components
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int pt = 0; pt < num_pts_; ++pt) {
      for (int node = 0; node < num_nodes_; ++node) {
        for (int dim1 = 0; dim1 < num_dims_; ++dim1) {
          for (int dim2 = 0; dim2 < num_dims_; ++dim2) {
            def_grad_(cell, pt, dim1, dim2) = solution_(cell, node, dim1, dim2);
            prescribed_def_grad_(cell, pt, dim1, dim2) = current_F(dim1, dim2);
            F.fill(def_grad_, cell, pt, 0, 0);
            j_(cell, pt) = minitensor::det(F);
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
minitensor::Tensor<typename EvalT::ScalarT>
ConstitutiveModelDriverPre<EvalT, Traits>::computeLoading(
    std::string load_case,
    double      inc)
{
  minitensor::Tensor<ScalarT> F0(num_dims_);
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));

  F0 = I;
  if (load_case == "uniaxial-strain") {
    F0(0, 0) += inc;
  } else if (load_case == "simple-shear") {
    F0(0, 1) += inc;
  } else if (load_case == "hydrostatic") {
    F0 = F0 * inc;
  }

  return F0;
}
//------------------------------------------------------------------------------
}  // namespace LCM
