//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_PUMIMeshStruct.hpp"
#include "Albany_PUMIDiscretization.hpp"
#include "PHAL_Workset.hpp"

namespace GOAL {

template<typename EvalT, typename Traits>
LpStress<EvalT, Traits>::
LpStress(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
  order    (p.get<int>         ("Order")),
  weight   (p.get<std::string> ("Weights Name"), dl->qp_scalar),
  stress   (p.get<std::string> ("Stress Name"), dl->qp_tensor),
  lpStress (p.get<std::string> ("Lp Stress Name"), dl->cell_scalar2)
{

  this->addDependentField(stress);
  this->addDependentField(weight);
  this->addEvaluatedField(lpStress);

  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_gradient->dimensions(dim);

  numNodes = dim[1];
  numQPs = dim[2];
  numDims = dim[3];
  
  this->setName("LpStress"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LpStress<EvalT, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress, fm);
  this->utils.setFieldData(weight, fm);
  this->utils.setFieldData(lpStress, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LpStress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  for (int cell=0; cell < workset.numCells; ++cell) {

    // initialize the integrated cell value to zero
    lpStress(cell) = 0.0;

    for (int qp=0; qp < numQPs; ++qp) {

      // get the Frobenius norm of stress
      ScalarT n = 0.0;
      for (int i=0; i < numDims; ++i)
        for (int j=0; j < numDims; ++j)
          n += stress(cell,qp,i,j) * stress(cell,qp,i,j);
      n = sqrt(n);

      // numerical integration
      lpStress(cell) += std::pow(n,order) * weight(cell,qp);

    }
  }

}

}
