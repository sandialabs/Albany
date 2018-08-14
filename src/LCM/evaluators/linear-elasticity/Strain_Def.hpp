//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace LCM {

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
Strain<EvalT, Traits>::Strain(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : GradU(p.get<std::string>("Gradient QP Variable Name"), dl->qp_tensor),
      strain(p.get<std::string>("Strain Name"), dl->qp_tensor)
{
  this->addDependentField(GradU);

  this->addEvaluatedField(strain);

  this->setName("Strain" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_tensor->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
Strain<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(strain, fm);
  this->utils.setFieldData(GradU, fm);
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
Strain<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  // Compute Strain tensor from displacement gradient
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < numQPs; ++qp) {
      for (int i = 0; i < numDims; ++i) {
        for (int j = 0; j < numDims; ++j) {
          strain(cell, qp, i, j) =
              0.5 * (GradU(cell, qp, i, j) + GradU(cell, qp, j, i));
        }
      }
    }
  }
}
//----------------------------------------------------------------------------
}  // namespace LCM
