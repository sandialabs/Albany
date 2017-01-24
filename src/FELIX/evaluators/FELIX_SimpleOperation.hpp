//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_SIMPLE_OPERATION_HPP
#define FELIX_SIMPLE_OPERATION_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX
{

/** \brief Field Norm Evaluator

    This evaluator evaluates the norm of a field
*/

namespace SimpleOps
{

template<typename ScalarT>
struct Log
{
  ScalarT operator() (const ScalarT& x) const {return std::log(x);}
};

template<typename ScalarT>
struct Exp
{
  ScalarT operator() (const ScalarT& x) const {return std::exp(x);}
};

}

template<typename EvalT, typename Traits, typename ScalarT, typename UnaryOperation>
class SimpleOperationBase: public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  SimpleOperationBase (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:

  // Input:
  PHX::MDField<ScalarT> field_in;

  // Output:
  PHX::MDField<ScalarT> field_out;

  // The operation
  UnaryOperation    op;
};

template<typename EvalT, typename Traits, typename ScalarT>
class SimpleOperationExp : public SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::Exp<ScalarT> >
{
public:
  SimpleOperationExp (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::Exp<ScalarT> >(p,dl) {}
};

template<typename EvalT, typename Traits, typename ScalarT>
class SimpleOperationLog : public SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::Log<ScalarT> >
{
public:
  SimpleOperationLog (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::Log<ScalarT> > (p,dl) {}
};

} // Namespace FELIX

#endif // FELIX_SIMPLE_OPERATION_HPP
