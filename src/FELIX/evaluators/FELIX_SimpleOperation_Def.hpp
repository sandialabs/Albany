//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

#include "PHAL_Utilities.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT, typename UnaryOperation>
SimpleOperationBase<EvalT, Traits, ScalarT, UnaryOperation>::
SimpleOperationBase (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl)
{
  std::string fieldInName  = p.get<std::string> ("Input Field Name");
  std::string fieldOutName = p.get<std::string> ("Output Field Name");

  Teuchos::RCP<PHX::DataLayout> layout = p.get<Teuchos::RCP<PHX::DataLayout>>("Field Layout");

  field_in  = PHX::MDField<ScalarT> (fieldInName,  layout);
  field_out = PHX::MDField<ScalarT> (fieldOutName, layout);

  this->addDependentField(field_in.fieldTag());
  this->addEvaluatedField(field_out);

  this->setName("SimpleOperationBase"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT, typename UnaryOperation>
void SimpleOperationBase<EvalT, Traits, ScalarT, UnaryOperation>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field_in,fm);
  this->utils.setFieldData(field_out,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT, typename UnaryOperation>
void SimpleOperationBase<EvalT, Traits, ScalarT, UnaryOperation>::
evaluateFields (typename Traits::EvalData workset)
{
  PHAL::MDFieldIterator<ScalarT> in(field_in);
  PHAL::MDFieldIterator<ScalarT> out(field_out);
  for (; !in.done(); ++in, ++out)
    *out = this->op(*in);
}

} // Namespace FELIX
