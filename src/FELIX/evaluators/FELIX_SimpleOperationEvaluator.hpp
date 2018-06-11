//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_SIMPLE_OPERATION_EVALUATOR_HPP
#define FELIX_SIMPLE_OPERATION_EVALUATOR_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

#include "FELIX_SimpleOperation.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits, typename InOutScalarT, typename Operation>
class SimpleOperationBase: public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  SimpleOperationBase (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);
protected:

  // Input:
  PHX::MDField<const InOutScalarT> field_in;

  // Output:
  PHX::MDField<InOutScalarT> field_out;

  // The operation
  Operation             op;
};

// =================== Specializations For Unary Operations ================= //

template<typename EvalT, typename Traits, typename InOutScalarT, template<typename> typename UnaryOperation>
class SimpleUnaryOperation : public SimpleOperationBase<EvalT,Traits,InOutScalarT,UnaryOperation<InOutScalarT>>
{
public:
  SimpleUnaryOperation  (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleOperationBase<EvalT,Traits,InOutScalarT,UnaryOperation<InOutScalarT>> (p,dl)
  {
    this->op.setup(p);
  }

  void evaluateFields(typename Traits::EvalData d);
};

template<typename EvalT, typename Traits, typename InOutScalarT>
class UnaryScaleOp : public SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Scale>
{
public:
  UnaryScaleOp (const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Scale>(p,dl) {}
};

template<typename EvalT, typename Traits, typename InOutScalarT>
class UnaryLogOp : public SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Log>
{
public:
  UnaryLogOp (const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Log>(p,dl) {}
};

template<typename EvalT, typename Traits, typename InOutScalarT>
class UnaryExpOp : public SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Exp>
{
public:
  UnaryExpOp (const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Exp>(p,dl) {}
};

template<typename EvalT, typename Traits, typename InOutScalarT>
class UnaryLowPassOp : public SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::LowPass>
{
public:
  UnaryLowPassOp (const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::LowPass>(p,dl) {}
};

template<typename EvalT, typename Traits, typename InOutScalarT>
class UnaryHighPassOp : public SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::HighPass>
{
public:
  UnaryHighPassOp (const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::HighPass>(p,dl) {}
};

template<typename EvalT, typename Traits, typename InOutScalarT>
class UnaryBandPassOp : public SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::BandPass>
{
public:
  UnaryBandPassOp (const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::BandPass>(p,dl) {}
};

// =================== Specializations For Binary Operations ================= //

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT, template<typename> typename BinaryOperation>
class SimpleBinaryOperation : public SimpleOperationBase<EvalT,Traits,InOutScalarT,BinaryOperation<InOutScalarT>>
{
public:

  SimpleBinaryOperation  (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleOperationBase<EvalT,Traits,InOutScalarT,BinaryOperation<InOutScalarT>> (p,dl)
  {
    field1 = PHX::MDField<const FieldScalarT> (p.get<std::string>("Parameter Field 1"), p.get<Teuchos::RCP<PHX::DataLayout>>("Field Layout"));
    this->addDependentField(field1);
  }

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm)
  {
    SimpleOperationBase<EvalT,Traits,InOutScalarT,BinaryOperation<InOutScalarT>>::postRegistrationSetup(d,fm);
    this->utils.setFieldData(field1,fm);
  }
  void evaluateFields(typename Traits::EvalData d);

private:
  PHX::MDField<const FieldScalarT> field1;
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class BinaryScaleOp : public SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::Scale>
{
public:
  BinaryScaleOp (const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::Scale>(p,dl) {}
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class BinaryLogOp : public SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::Log>
{
public:
  BinaryLogOp (const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::Log>(p,dl) {}
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class BinaryExpOp : public SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::Exp>
{
public:
  BinaryExpOp (const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::Exp>(p,dl) {}
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class BinaryLowPassOp : public SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::LowPass>
{
public:
  BinaryLowPassOp (const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::LowPass>(p,dl) {}
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class BinaryHighPassOp : public SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::HighPass>
{
public:
  BinaryHighPassOp (const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::HighPass>(p,dl) {}
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class BinaryBandPassFixedLowerOp : public SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::BandPassFixedLower>
{
public:
  BinaryBandPassFixedLowerOp (const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::BandPassFixedLower>(p,dl) {}
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class BinaryBandPassFixedUpperOp : public SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::BandPassFixedUpper>
{
public:
  BinaryBandPassFixedUpperOp (const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::BandPassFixedUpper>(p,dl) {}
};

// =================== Specializations For Ternary Operations ================= //

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT, template<typename> typename TernaryOperation>
class SimpleTernaryOperation : public SimpleOperationBase<EvalT,Traits,InOutScalarT,TernaryOperation<InOutScalarT>>
{
public:

  SimpleTernaryOperation (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleOperationBase<EvalT,Traits,InOutScalarT,TernaryOperation<InOutScalarT>> (p,dl)
  {
    field1 = PHX::MDField<const FieldScalarT> (p.get<std::string>("Parameter Field 1"), p.get<Teuchos::RCP<PHX::DataLayout>>("Field Layout"));
    field2 = PHX::MDField<const FieldScalarT> (p.get<std::string>("Parameter Field 2"), p.get<Teuchos::RCP<PHX::DataLayout>>("Field Layout"));
    this->addDependentField(field1);
    this->addDependentField(field2);
  }

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm)
  {
    SimpleOperationBase<EvalT,Traits,InOutScalarT,TernaryOperation<InOutScalarT>>::postRegistrationSetup(d,fm);
    this->utils.setFieldData(field1,fm);
    this->utils.setFieldData(field2,fm);
  }
  void evaluateFields(typename Traits::EvalData d);

private:

  PHX::MDField<const FieldScalarT> field1;
  PHX::MDField<const FieldScalarT> field2;
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class TernaryBandPassOp : public SimpleTernaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,TernaryOps::BandPass>
{
public:
  TernaryBandPassOp (const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleTernaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,TernaryOps::BandPass>(p,dl) {}
};

} // Namespace FELIX

#endif // FELIX_SIMPLE_OPERATION_EVALUATOR_HPP
