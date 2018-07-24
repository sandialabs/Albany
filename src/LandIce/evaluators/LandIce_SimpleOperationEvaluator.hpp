//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_SIMPLE_OPERATION_EVALUATOR_HPP
#define LANDICE_SIMPLE_OPERATION_EVALUATOR_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

#include "LandIce_SimpleOperation.hpp"

namespace LandIce
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

template<typename EvalT, typename Traits, typename InOutScalarT, template<typename> class UnaryOperation>
class SimpleUnaryOperation : public SimpleOperationBase<EvalT,Traits,InOutScalarT,UnaryOperation<InOutScalarT>>
{
public:
  SimpleUnaryOperation (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);
};

template<typename EvalT, typename Traits, typename InOutScalarT>
class UnaryScaleOp : public SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Scale>
{
public:
  UnaryScaleOp (const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);
};

template<typename EvalT, typename Traits, typename InOutScalarT>
class UnaryLogOp : public SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Log>
{
public:
  UnaryLogOp (const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);
};

template<typename EvalT, typename Traits, typename InOutScalarT>
class UnaryExpOp : public SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::Exp>
{
public:
  UnaryExpOp (const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);
};

template<typename EvalT, typename Traits, typename InOutScalarT>
class UnaryLowPassOp : public SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::LowPass>
{
public:
  UnaryLowPassOp (const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
};

template<typename EvalT, typename Traits, typename InOutScalarT>
class UnaryHighPassOp : public SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::HighPass>
{
public:
  UnaryHighPassOp (const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl);
};

template<typename EvalT, typename Traits, typename InOutScalarT>
class UnaryBandPassOp : public SimpleUnaryOperation<EvalT,Traits,InOutScalarT,UnaryOps::BandPass>
{
public:
  UnaryBandPassOp (const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl);
};

// =================== Specializations For Binary Operations ================= //

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT, template<typename> class BinaryOperation>
class SimpleBinaryOperation : public SimpleOperationBase<EvalT,Traits,InOutScalarT,BinaryOperation<InOutScalarT>>
{
public:
  SimpleBinaryOperation (const Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  PHX::MDField<const FieldScalarT> field1;
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class BinaryScaleOp : public SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::Scale>
{
public:
  BinaryScaleOp (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class BinaryLogOp : public SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::Log>
{
public:
  BinaryLogOp (const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl);
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class BinaryExpOp : public SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::Exp>
{
public:
  BinaryExpOp (const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl);
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class BinaryLowPassOp : public SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::LowPass>
{
public:
  BinaryLowPassOp (const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl);
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class BinaryHighPassOp : public SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::HighPass>
{
public:
  BinaryHighPassOp (const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl);
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class BinaryBandPassFixedLowerOp : public SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::BandPassFixedLower>
{
public:
  BinaryBandPassFixedLowerOp (const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT>
class BinaryBandPassFixedUpperOp : public SimpleBinaryOperation<EvalT,Traits,InOutScalarT,FieldScalarT,BinaryOps::BandPassFixedUpper>
{
public:
  BinaryBandPassFixedUpperOp (const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
};

// =================== Specializations For Ternary Operations ================= //

template<typename EvalT, typename Traits, typename InOutScalarT, typename FieldScalarT, template<typename> class TernaryOperation>
class SimpleTernaryOperation : public SimpleOperationBase<EvalT,Traits,InOutScalarT,TernaryOperation<InOutScalarT>>
{
public:
  SimpleTernaryOperation (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

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
                     const Teuchos::RCP<Albany::Layouts>& dl);
};

} // Namespace LandIce

#endif // LANDICE_SIMPLE_OPERATION_EVALUATOR_HPP
