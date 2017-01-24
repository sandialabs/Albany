//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
DummyResidual<EvalT, Traits>::DummyResidual (const Teuchos::ParameterList& p,
                                             const Teuchos::RCP<Albany::Layouts>& dl) :
  solution (p.get<std::string> ("Solution Variable Name"), dl->node_scalar),
  residual (p.get<std::string> ("Residual Variable Name"),dl->node_scalar)
{
  this->addDependentField(solution.fieldTag());
  this->addEvaluatedField(residual);

  this->setName("DummyResidual"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DummyResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(solution,fm);
  this->utils.setFieldData(residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DummyResidual<EvalT, Traits>::evaluateFields (typename Traits::EvalData /*workset*/)
{
  residual.deep_copy(solution);
}

} // Namespace FELIX
