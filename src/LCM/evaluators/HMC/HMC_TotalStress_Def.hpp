//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

#include "Albany_Utils.hpp"

namespace HMC {

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
TotalStress<EvalT, Traits>::TotalStress(
    const Teuchos::ParameterList&        p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : macroStress(p.get<std::string>("Macro Stress Name"), dl->qp_tensor),
      totalStress(p.get<std::string>("Total Stress Name"), dl->qp_tensor),
      numMicroScales(p.get<int>("Additional Scales"))
{
  this->addDependentField(macroStress);

  microStress.resize(numMicroScales);
  for (int i = 0; i < numMicroScales; i++) {
    std::string ms = Albany::strint("Micro Stress", i);
    std::string msname(ms);
    msname += " Name";
    microStress[i] = Teuchos::rcp(new cHMC2Tensor(
        p.get<std::string>(msname),
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP 2Tensor Data Layout")));
    this->addDependentField(*(microStress[i]));
  }

  this->addEvaluatedField(totalStress);

  this->setName("TotalStress" + PHX::typeAsString<EvalT>());

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_tensor->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
TotalStress<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(macroStress, fm);
  this->utils.setFieldData(totalStress, fm);
  int n = this->numMicroScales;
  for (int i = 0; i < n; i++) this->utils.setFieldData(*(microStress[i]), fm);
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
TotalStress<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  // Compute strain difference tensor
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t qp = 0; qp < numQPs; ++qp) {
      for (std::size_t i = 0; i < numDims; ++i) {
        for (std::size_t j = 0; j < numDims; ++j) {
          totalStress(cell, qp, i, j) = macroStress(cell, qp, i, j);
          for (std::size_t k = 0; k < numMicroScales; ++k)
            totalStress(cell, qp, i, j) += (*microStress[k])(cell, qp, i, j);
        }
      }
    }
  }
}
//----------------------------------------------------------------------------
}  // namespace HMC
