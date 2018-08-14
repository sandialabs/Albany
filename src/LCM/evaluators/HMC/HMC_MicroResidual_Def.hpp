//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace HMC {

//**********************************************************************
template <typename EvalT, typename Traits>
MicroResidual<EvalT, Traits>::MicroResidual(const Teuchos::ParameterList& p)
    : microStress(
          p.get<std::string>("Micro Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      doubleStress(
          p.get<std::string>("Double Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP 3Tensor Data Layout")),
      wGradBF(
          p.get<std::string>("Weighted Gradient BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout")),
      wBF(p.get<std::string>("Weighted BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout")),
      ExResidual(
          p.get<std::string>("Residual Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node Tensor Data Layout"))
{
  this->addDependentField(microStress);
  this->addDependentField(doubleStress);
  this->addDependentField(wGradBF);
  this->addDependentField(wBF);

  this->addEvaluatedField(ExResidual);

  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else
    enableTransient = true;

  if (enableTransient) {
    // One more field is required for transient capability
    Teuchos::RCP<PHX::DataLayout> tensor_dl =
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout");
    epsDotDot = decltype(epsDotDot)(
        p.get<std::string>("Time Dependent Variable Name"), tensor_dl);
    this->addDependentField(epsDotDot);
  }

  this->setName("MicroResidual" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
MicroResidual<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(microStress, fm);
  this->utils.setFieldData(doubleStress, fm);
  this->utils.setFieldData(wGradBF, fm);
  this->utils.setFieldData(wBF, fm);

  this->utils.setFieldData(ExResidual, fm);

  if (enableTransient) this->utils.setFieldData(epsDotDot, fm);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
MicroResidual<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  // typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t node = 0; node < numNodes; ++node) {
      for (std::size_t idim = 0; idim < numDims; idim++)
        for (std::size_t jdim = 0; jdim < numDims; jdim++)
          ExResidual(cell, node, idim, jdim) = 0.0;
      for (std::size_t qp = 0; qp < numQPs; ++qp) {
        for (std::size_t i = 0; i < numDims; i++) {
          for (std::size_t j = 0; j < numDims; j++) {
            for (std::size_t dim = 0; dim < numDims; dim++) {
              ExResidual(cell, node, i, j) +=
                  doubleStress(cell, qp, i, j, dim) *
                      wGradBF(cell, node, qp, dim) +
                  microStress(cell, qp, i, j) * wBF(cell, node, qp);
            }
          }
        }
      }
    }
  }

  if (workset.transientTerms && enableTransient)
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t node = 0; node < numNodes; ++node) {
        for (std::size_t qp = 0; qp < numQPs; ++qp) {
          for (std::size_t i = 0; i < numDims; i++) {
            for (std::size_t j = 0; j < numDims; j++) {
              ExResidual(cell, node, i, j) +=
                  epsDotDot(cell, qp, i, j) * wBF(cell, node, qp);
            }
          }
        }
      }
    }
}

//**********************************************************************
}  // namespace HMC
