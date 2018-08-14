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
Stress<EvalT, Traits>::Stress(const Teuchos::ParameterList& p)
    : strain(
          p.get<std::string>("Strain Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      elasticModulus(
          p.get<std::string>("Elastic Modulus Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      poissonsRatio(
          p.get<std::string>("Poissons Ratio Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout")),
      stress(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout"))
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(strain);
  this->addDependentField(elasticModulus);
  // PoissonRatio not used in 1D stress calc
  if (numDims > 1) this->addDependentField(poissonsRatio);

  this->addEvaluatedField(stress);

  this->setName("Stress" + PHX::typeAsString<EvalT>());
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
Stress<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress, fm);
  this->utils.setFieldData(strain, fm);
  this->utils.setFieldData(elasticModulus, fm);
  if (numDims > 1) this->utils.setFieldData(poissonsRatio, fm);
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
Stress<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  ScalarT lambda, mu;

  switch (numDims) {
    case 1:
      Intrepid2::FunctionSpaceTools<PHX::Device>::tensorMultiplyDataData(
          stress.get_view(), elasticModulus.get_view(), strain.get_view());
      break;
    case 2:
      // Compute Stress (with the plane strain assumption for now)
      for (int cell = 0; cell < workset.numCells; ++cell) {
        for (int qp = 0; qp < numQPs; ++qp) {
          lambda = (elasticModulus(cell, qp) * poissonsRatio(cell, qp)) /
                   ((1 + poissonsRatio(cell, qp)) *
                    (1 - 2 * poissonsRatio(cell, qp)));
          mu = elasticModulus(cell, qp) / (2 * (1 + poissonsRatio(cell, qp)));
          stress(cell, qp, 0, 0) =
              2.0 * mu * (strain(cell, qp, 0, 0)) +
              lambda * (strain(cell, qp, 0, 0) + strain(cell, qp, 1, 1));
          stress(cell, qp, 1, 1) =
              2.0 * mu * (strain(cell, qp, 1, 1)) +
              lambda * (strain(cell, qp, 0, 0) + strain(cell, qp, 1, 1));
          stress(cell, qp, 0, 1) = 2.0 * mu * (strain(cell, qp, 0, 1));
          stress(cell, qp, 1, 0) = stress(cell, qp, 0, 1);
        }
      }
      break;
    case 3:
      // Compute Stress
      for (int cell = 0; cell < workset.numCells; ++cell) {
        for (int qp = 0; qp < numQPs; ++qp) {
          lambda = (elasticModulus(cell, qp) * poissonsRatio(cell, qp)) /
                   ((1 + poissonsRatio(cell, qp)) *
                    (1 - 2 * poissonsRatio(cell, qp)));
          mu = elasticModulus(cell, qp) / (2 * (1 + poissonsRatio(cell, qp)));
          stress(cell, qp, 0, 0) =
              2.0 * mu * (strain(cell, qp, 0, 0)) +
              lambda * (strain(cell, qp, 0, 0) + strain(cell, qp, 1, 1) +
                        strain(cell, qp, 2, 2));
          stress(cell, qp, 1, 1) =
              2.0 * mu * (strain(cell, qp, 1, 1)) +
              lambda * (strain(cell, qp, 0, 0) + strain(cell, qp, 1, 1) +
                        strain(cell, qp, 2, 2));
          stress(cell, qp, 2, 2) =
              2.0 * mu * (strain(cell, qp, 2, 2)) +
              lambda * (strain(cell, qp, 0, 0) + strain(cell, qp, 1, 1) +
                        strain(cell, qp, 2, 2));
          stress(cell, qp, 0, 1) = 2.0 * mu * (strain(cell, qp, 0, 1));
          stress(cell, qp, 1, 2) = 2.0 * mu * (strain(cell, qp, 1, 2));
          stress(cell, qp, 2, 0) = 2.0 * mu * (strain(cell, qp, 2, 0));
          stress(cell, qp, 1, 0) = stress(cell, qp, 0, 1);
          stress(cell, qp, 2, 1) = stress(cell, qp, 1, 2);
          stress(cell, qp, 0, 2) = stress(cell, qp, 2, 0);
        }
      }
      break;
  }
}

//**********************************************************************
}  // namespace LCM
