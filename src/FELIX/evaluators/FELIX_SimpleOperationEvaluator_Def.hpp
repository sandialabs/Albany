//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

#include "PHAL_Utilities.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, typename Operation>
SimpleOperationBase<EvalT, Traits, InOutScalarT, Operation>::
SimpleOperationBase (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl)
{
  std::string fieldInName  = p.get<std::string> ("Input Field Name");
  std::string fieldOutName = p.get<std::string> ("Output Field Name");

  Teuchos::RCP<PHX::DataLayout> layout = p.get<Teuchos::RCP<PHX::DataLayout>>("Field Layout");
  TEUCHOS_TEST_FOR_EXCEPTION (layout.is_null(), std::runtime_error, "Error! Input layout is null.\n");

  field_in  = PHX::MDField<InOutScalarT> (fieldInName,  layout);
  field_out = PHX::MDField<InOutScalarT> (fieldOutName, layout);

  this->addDependentField(field_in);
  this->addEvaluatedField(field_out);

  this->setName("SimpleOperationBase"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, typename Operation>
void SimpleOperationBase<EvalT, Traits, InOutScalarT, Operation>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field_in,fm);
  this->utils.setFieldData(field_out,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, template<typename> typename UnaryOperation>
void SimpleUnaryOperation<EvalT, Traits, InOutScalarT, UnaryOperation>::
evaluateFields (typename Traits::EvalData workset)
{
  PHAL::MDFieldIterator<const InOutScalarT> in(this->field_in);
  PHAL::MDFieldIterator<InOutScalarT> out(this->field_out);
  for (; !in.done(); ++in, ++out) {
    *out = this->op(*in);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT, template<typename> typename BinaryOperation>
void SimpleBinaryOperation<EvalT, Traits, InOutScalarT, FieldScalarT, BinaryOperation>::
evaluateFields (typename Traits::EvalData workset)
{
  PHAL::MDFieldIterator<const InOutScalarT> in(this->field_in);
  PHAL::MDFieldIterator<const FieldScalarT> param1(field1);
  PHAL::MDFieldIterator<InOutScalarT> out(this->field_out);
  for (; !in.done(); ++in, ++out, ++param1) {
    *out = this->op(*in,Albany::convertScalar<const InOutScalarT>(*param1));
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT, template<typename> typename TernaryOperation>
void SimpleTernaryOperation<EvalT, Traits, InOutScalarT, FieldScalarT, TernaryOperation>::
evaluateFields (typename Traits::EvalData workset)
{
  PHAL::MDFieldIterator<const InOutScalarT> in(this->field_in);
  PHAL::MDFieldIterator<const FieldScalarT> param1(field1);
  PHAL::MDFieldIterator<const FieldScalarT> param2(field2);
  PHAL::MDFieldIterator<InOutScalarT> out(this->field_out);
  for (; !in.done(); ++in, ++out, ++param1, ++param2) {
    *out = this->op(*in,Albany::convertScalar<const InOutScalarT>(*param1),Albany::convertScalar<const InOutScalarT>(*param2));
  }
}

} // Namespace FELIX
