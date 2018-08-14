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
MixtureSpecificHeat<EvalT, Traits>::MixtureSpecificHeat(
    const Teuchos::ParameterList& p)
    : porosity(
          p.get<std::string>("Porosity Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      gammaSkeleton(
          p.get<std::string>("Skeleton Specific Heat Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      gammaPoreFluid(
          p.get<std::string>("Pore-Fluid Specific Heat Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      densitySkeleton(
          p.get<std::string>("Skeleton Density Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      densityPoreFluid(
          p.get<std::string>("Pore-Fluid Density Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      J(p.get<std::string>("DetDefGrad Name"),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      mixtureSpecificHeat(
          p.get<std::string>("Mixture Specific Heat Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  this->addDependentField(porosity);
  this->addDependentField(J);
  this->addDependentField(gammaPoreFluid);
  this->addDependentField(gammaSkeleton);
  this->addDependentField(densityPoreFluid);
  this->addDependentField(densitySkeleton);

  this->addEvaluatedField(mixtureSpecificHeat);

  this->setName("Mixture Specific Heat" + PHX::typeAsString<EvalT>());

  Teuchos::RCP<PHX::DataLayout> scalar_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  scalar_dl->dimensions(dims);
  numQPs = dims[1];
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
MixtureSpecificHeat<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(mixtureSpecificHeat, fm);
  this->utils.setFieldData(porosity, fm);
  this->utils.setFieldData(J, fm);
  this->utils.setFieldData(gammaSkeleton, fm);
  this->utils.setFieldData(gammaPoreFluid, fm);
  this->utils.setFieldData(densitySkeleton, fm);
  this->utils.setFieldData(densityPoreFluid, fm);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
MixtureSpecificHeat<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // Compute Strain tensor from displacement gradient
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < numQPs; ++qp) {
      mixtureSpecificHeat(cell, qp) =
          (J(cell, qp) - porosity(cell, qp)) * gammaSkeleton(cell, qp) *
              densitySkeleton(cell, qp) +
          porosity(cell, qp) * gammaPoreFluid(cell, qp) *
              densityPoreFluid(cell, qp);
    }
  }
}

//**********************************************************************
}  // namespace LCM
