//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

#include "PHAL_Utilities.hpp"

namespace LandIce {

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
template<typename EvalT, typename Traits, typename InOutScalarT, template<typename> class UnaryOperation>
SimpleUnaryOperation<EvalT, Traits, InOutScalarT, UnaryOperation>::
SimpleUnaryOperation (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleOperationBase<EvalT,Traits,InOutScalarT,UnaryOperation<InOutScalarT>> (p,dl)
{
  this->op.setup(p);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, template<typename> class UnaryOperation>
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
template<typename EvalT, typename Traits, typename InOutScalarT>
UnaryScaleOp<EvalT, Traits, InOutScalarT>::
UnaryScaleOp (const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Scale>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT>
UnaryLogOp<EvalT, Traits, InOutScalarT>::
UnaryLogOp (const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Log>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT>
UnaryExpOp<EvalT, Traits, InOutScalarT>::
UnaryExpOp (const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Exp>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT>
UnaryLowPassOp<EvalT, Traits, InOutScalarT>::
UnaryLowPassOp (const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::LowPass>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT>
UnaryHighPassOp<EvalT, Traits, InOutScalarT>::
UnaryHighPassOp (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::HighPass>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT>
UnaryBandPassOp<EvalT, Traits, InOutScalarT>::
UnaryBandPassOp (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::BandPass>(p,dl) {}

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT, template<typename> class BinaryOperation>
SimpleBinaryOperation<EvalT, Traits, InOutScalarT, FieldScalarT, BinaryOperation>::
SimpleBinaryOperation (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleOperationBase<EvalT,Traits,InOutScalarT,BinaryOperation<InOutScalarT>> (p,dl)
{
  field1 = PHX::MDField<const FieldScalarT> (p.get<std::string>("Parameter Field 1"), 
      p.get<Teuchos::RCP<PHX::DataLayout>>("Field Layout"));
  this->addDependentField(field1);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT, template<typename> class BinaryOperation>
void SimpleBinaryOperation<EvalT, Traits, InOutScalarT, FieldScalarT, BinaryOperation>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  SimpleOperationBase<EvalT,Traits,InOutScalarT,BinaryOperation<InOutScalarT>>::postRegistrationSetup(d,fm);
  this->utils.setFieldData(field1,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT, template<typename> class BinaryOperation>
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
template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
BinaryScaleOp<EvalT, Traits, InOutScalarT, FieldScalarT>::
BinaryScaleOp (const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::Scale>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
BinaryLogOp<EvalT, Traits, InOutScalarT, FieldScalarT>::
BinaryLogOp (const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::Log>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
BinaryExpOp<EvalT, Traits, InOutScalarT, FieldScalarT>::
BinaryExpOp (const Teuchos::ParameterList& p,
             const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::Exp>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
BinaryLowPassOp<EvalT, Traits, InOutScalarT, FieldScalarT>::
BinaryLowPassOp (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::LowPass>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
BinaryHighPassOp<EvalT, Traits, InOutScalarT, FieldScalarT>::
BinaryHighPassOp (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::HighPass>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
BinaryBandPassFixedLowerOp<EvalT, Traits, InOutScalarT, FieldScalarT>::
BinaryBandPassFixedLowerOp (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::BandPassFixedLower>(p,dl) {}

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
BinaryBandPassFixedUpperOp<EvalT, Traits, InOutScalarT, FieldScalarT>::
BinaryBandPassFixedUpperOp (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::BandPassFixedUpper>(p,dl) {}

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT, template<typename> class TernaryOperation>
SimpleTernaryOperation<EvalT, Traits, InOutScalarT, FieldScalarT, TernaryOperation>::
SimpleTernaryOperation (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleOperationBase<EvalT,Traits,InOutScalarT,TernaryOperation<InOutScalarT>> (p,dl)
{
  field1 = PHX::MDField<const FieldScalarT> (p.get<std::string>("Parameter Field 1"),
      p.get<Teuchos::RCP<PHX::DataLayout>>("Field Layout"));
  field2 = PHX::MDField<const FieldScalarT> (p.get<std::string>("Parameter Field 2"),
      p.get<Teuchos::RCP<PHX::DataLayout>>("Field Layout"));
  this->addDependentField(field1);
  this->addDependentField(field2);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT, template<typename> class TernaryOperation>
void SimpleTernaryOperation<EvalT, Traits, InOutScalarT, FieldScalarT, TernaryOperation>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  SimpleOperationBase<EvalT,Traits,InOutScalarT,TernaryOperation<InOutScalarT>>::postRegistrationSetup(d,fm);
  this->utils.setFieldData(field1,fm);
  this->utils.setFieldData(field2,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT, template<typename> class TernaryOperation>
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

//**********************************************************************
template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
TernaryBandPassOp<EvalT, Traits, InOutScalarT, FieldScalarT>::
TernaryBandPassOp (const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl) :
  SimpleTernaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,TernaryOps::BandPass>(p,dl) {}

} // Namespace LandIce

