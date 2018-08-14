//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace HMC {

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
StrainDifference<EvalT, Traits>::StrainDifference(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : macroStrain(p.get<std::string>("Macro Strain Name"), dl->qp_tensor),
      microStrain(p.get<std::string>("Micro Strain Name"), dl->qp_tensor),
      strainDifference(
          p.get<std::string>("Strain Difference Name"),
          dl->qp_tensor)
{
  this->addDependentField(microStrain);
  this->addDependentField(macroStrain);

  this->addEvaluatedField(strainDifference);

  this->setName("StrainDifference" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_tensor->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
StrainDifference<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(microStrain, fm);
  this->utils.setFieldData(macroStrain, fm);
  this->utils.setFieldData(strainDifference, fm);
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
StrainDifference<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // Compute strain difference tensor
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t qp = 0; qp < numQPs; ++qp) {
      for (std::size_t i = 0; i < numDims; ++i) {
        for (std::size_t j = 0; j < numDims; ++j) {
          strainDifference(cell, qp, i, j) =
              macroStrain(cell, qp, i, j) - microStrain(cell, qp, i, j);
        }
      }
    }
  }
}
//----------------------------------------------------------------------------
}  // namespace HMC
