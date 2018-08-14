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
TLPoroStress<EvalT, Traits>::TLPoroStress(const Teuchos::ParameterList& p)
    : stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      defGrad(
          p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      J(p.get<std::string>("DetDefGrad Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      biotCoefficient(
          p.get<std::string>("Biot Coefficient Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      porePressure(
          p.get<std::string>("QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      totstress(
          p.get<std::string>("Total Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout"))
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs      = dims[1];
  numDims     = dims[2];
  worksetSize = dims[0];

  this->addDependentField(stress);
  this->addDependentField(defGrad);
  this->addDependentField(J);
  this->addDependentField(biotCoefficient);
  // this->addDependentField(porePressure);

  this->addEvaluatedField(porePressure);
  this->addEvaluatedField(totstress);

  this->setName("TLPoroStress" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
TLPoroStress<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress, fm);
  this->utils.setFieldData(defGrad, fm);
  this->utils.setFieldData(J, fm);
  this->utils.setFieldData(biotCoefficient, fm);
  this->utils.setFieldData(porePressure, fm);
  this->utils.setFieldData(totstress, fm);

  // Works space FCs
  F_inv = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  F_invT = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  JF_invT = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  JpF_invT = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
  JBpF_invT = Kokkos::createDynRankView(
      J.get_view(), "XXX", worksetSize, numQPs, numDims, numDims);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
TLPoroStress<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;
  typedef Intrepid2::RealSpaceTools<PHX::Device>     RST;

  if (numDims == 1) {
    Intrepid2::FunctionSpaceTools<PHX::Device>::scalarMultiplyDataData(
        totstress.get_view(), J.get_view(), stress.get_view());
    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        for (int dim = 0; dim < numDims; ++dim) {
          for (int j = 0; j < numDims; ++j) {
            totstress(cell, qp, dim, j) =
                stress(cell, qp, dim, j) -
                biotCoefficient(cell, qp) * porePressure(cell, qp);
          }
        }
      }
    }
  } else {
    RST::inverse(F_inv, defGrad.get_view());
    RST::transpose(F_invT, F_inv);
    FST::scalarMultiplyDataData<ScalarT>(JF_invT, J.get_view(), F_invT);
    FST::scalarMultiplyDataData<ScalarT>(
        JpF_invT, porePressure.get_view(), JF_invT);
    FST::scalarMultiplyDataData<ScalarT>(
        JBpF_invT, biotCoefficient.get_view(), JpF_invT);
    FST::tensorMultiplyDataData<ScalarT>(
        totstress.get_view(), stress.get_view(), JF_invT);  // Cauchy to 1st PK

    // Compute Stress

    for (int cell = 0; cell < workset.numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        for (int dim = 0; dim < numDims; ++dim) {
          for (int j = 0; j < numDims; ++j) {
            totstress(cell, qp, dim, j) -= JBpF_invT(cell, qp, dim, j);
          }
        }
      }
    }
  }
}

//**********************************************************************
}  // namespace LCM
