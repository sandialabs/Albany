//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_Application.hpp"
#include "Albany_PUMIMeshStruct.hpp"
#include "Albany_PUMIDiscretization.hpp"
#include "PHAL_Workset.hpp"

namespace GOAL {

//**********************************************************************
template<typename EvalT, typename Traits>
ScatterQoI<EvalT, Traits>::
ScatterQoI(
    const Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
  qoi (p.get<std::string>("QoI Name"), dl->cell_scalar2)
{
  operation = Teuchos::rcp(new PHX::Tag<ScalarT>("Scatter QoI", dl->dummy));

  this->addDependentField(qoi);
  this->addEvaluatedField(*operation);
  this->setName("ScatterQoI"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ScatterQoI<EvalT, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(qoi, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ScatterQoI<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
}

}
