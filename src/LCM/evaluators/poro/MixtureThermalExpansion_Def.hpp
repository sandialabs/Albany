//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
MixtureThermalExpansion<EvalT, Traits>::MixtureThermalExpansion(
    const Teuchos::ParameterList& p)
    : biotCoefficient(
          p.get<std::string>("Biot Coefficient Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      porosity(
          p.get<std::string>("Porosity Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      alphaSkeleton(
          p.get<std::string>("Skeleton Thermal Expansion Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      alphaPoreFluid(
          p.get<std::string>("Pore-Fluid Thermal Expansion Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      J(p.get<std::string>("DetDefGrad Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      mixtureThermalExpansion(
          p.get<std::string>("Mixture Thermal Expansion Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  this->addDependentField(biotCoefficient);
  this->addDependentField(porosity);
  this->addDependentField(J);
  this->addDependentField(alphaPoreFluid);
  this->addDependentField(alphaSkeleton);

  this->addEvaluatedField(mixtureThermalExpansion);

  this->setName("Mixture Thermal Expansion" + PHX::typeAsString<EvalT>());

  Teuchos::RCP<PHX::DataLayout> scalar_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  scalar_dl->dimensions(dims);
  numQPs = dims[1];
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
MixtureThermalExpansion<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(mixtureThermalExpansion, fm);
  this->utils.setFieldData(biotCoefficient, fm);
  this->utils.setFieldData(porosity, fm);
  this->utils.setFieldData(J, fm);
  this->utils.setFieldData(alphaSkeleton, fm);
  this->utils.setFieldData(alphaPoreFluid, fm);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
MixtureThermalExpansion<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // Compute Strain tensor from displacement gradient
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < numQPs; ++qp) {
      mixtureThermalExpansion(cell, qp) =
          (biotCoefficient(cell, qp) * J(cell, qp) - porosity(cell, qp)) *
              alphaSkeleton(cell, qp) +
          porosity(cell, qp) * alphaPoreFluid(cell, qp);
    }
  }
}

//**********************************************************************
}  // namespace LCM
