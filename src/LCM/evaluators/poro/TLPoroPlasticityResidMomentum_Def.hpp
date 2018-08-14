//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
TLPoroPlasticityResidMomentum<EvalT, Traits>::TLPoroPlasticityResidMomentum(
    const Teuchos::ParameterList& p)
    : TotalStress(
          p.get<std::string>("Total Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      J(p.get<std::string>("DetDefGrad Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      defgrad(
          p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      wGradBF(
          p.get<std::string>("Weighted Gradient BF Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout")),
      ExResidual(
          p.get<std::string>("Residual Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Node Vector Data Layout"))
{
  this->addDependentField(TotalStress);
  this->addDependentField(wGradBF);
  this->addDependentField(J);
  this->addDependentField(defgrad);
  this->addEvaluatedField(ExResidual);

  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else
    enableTransient = true;

  if (enableTransient) {
    // Two more fields are required for transient capability
    Teuchos::RCP<PHX::DataLayout> node_qp_scalar_dl =
        p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout");
    Teuchos::RCP<PHX::DataLayout> vector_dl =
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");

    wBF = decltype(wBF)(
        p.get<std::string>("Weighted BF Name"), node_qp_scalar_dl);
    uDotDot = decltype(uDotDot)(
        p.get<std::string>("Time Dependent Variable Name"), vector_dl);

    this->addDependentField(wBF);
    this->addDependentField(uDotDot);
  }

  this->setName("TLPoroPlasticityResidMomentum" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes    = dims[1];
  numQPs      = dims[2];
  numDims     = dims[3];
  worksetSize = dims[0];
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
TLPoroPlasticityResidMomentum<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(TotalStress, fm);
  this->utils.setFieldData(wGradBF, fm);
  this->utils.setFieldData(J, fm);
  this->utils.setFieldData(defgrad, fm);
  this->utils.setFieldData(ExResidual, fm);

  if (enableTransient) this->utils.setFieldData(uDotDot, fm);
  if (enableTransient) this->utils.setFieldData(wBF, fm);

  // Works space FCs
  F_inv = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  F_invT = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  JF_invT = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
TLPoroPlasticityResidMomentum<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;
  typedef Intrepid2::RealSpaceTools<PHX::Device>     RST;
  RST::inverse(F_inv, defgrad.get_view());
  RST::transpose(F_invT, F_inv);
  FST::scalarMultiplyDataData(JF_invT, J.get_view(), F_invT);
  // FST::tensorMultiplyDataData(P.get_view(), TotalStress.get_view(), JF_invT);

  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < numNodes; ++node) {
      for (int dim = 0; dim < numDims; dim++) ExResidual(cell, node, dim) = 0.0;
      for (int qp = 0; qp < numQPs; ++qp) {
        for (int i = 0; i < numDims; i++) {
          for (int dim = 0; dim < numDims; dim++) {
            ExResidual(cell, node, i) +=
                TotalStress(cell, qp, i, dim) * wGradBF(cell, node, qp, dim);
          }
        }
      }
    }
  }

  if (workset.transientTerms && enableTransient)
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int node = 0; node < numNodes; ++node) {
        for (int qp = 0; qp < numQPs; ++qp) {
          for (int i = 0; i < numDims; i++) {
            ExResidual(cell, node, i) +=
                uDotDot(cell, qp, i) * wBF(cell, node, qp);
          }
        }
      }
    }

  //   FST::integrate(ExResidual.get_view(), TotalStress.get_view(),
  //   wGradBF.get_view(), false); // "false" overwrites
}

//**********************************************************************
}  // namespace LCM
