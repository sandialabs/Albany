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

namespace SimpleOps
{

template<typename ScalarT>
struct Scale
{
  ScalarT operator() (const ScalarT& x) const {return factor*x;}

  ScalarT factor;
};

template<typename ScalarT>
struct Log
{
  ScalarT operator() (const ScalarT& x) const {return std::log(x);}
};

template<typename ScalarT>
struct Exp
{
  ScalarT operator() (const ScalarT& x) const {return std::exp(tau*x);}

  ScalarT tau;
};

template<typename ScalarT>
struct LowPass
{
  ScalarT operator() (const ScalarT& x) const {return std::min(x,threshold_up);}

  ScalarT threshold_up;
};

template<typename ScalarT>
struct HighPass
{
  ScalarT operator() (const ScalarT& x) const {return std::max(x,threshold_lo);}

  ScalarT threshold_lo;
};

template<typename ScalarT>
struct BandPass
{
  ScalarT operator() (const ScalarT& x) const {return std::max(std::min(x,threshold_up),threshold_lo);}

  ScalarT threshold_lo;
  ScalarT threshold_up;
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

protected:

  // Input:
  PHX::MDField<const ScalarT> field_in;

  // Output:
  PHX::MDField<ScalarT> field_out;

  // The operation
  UnaryOperation    op;
};

// ======================= Derived Specializations ================= //

template<typename EvalT, typename Traits, typename ScalarT>
class SimpleOperationScale : public SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::Scale<ScalarT> >
{
public:
  SimpleOperationScale (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::Scale<ScalarT> >(p,dl) {
      this->op.factor = ScalarT(p.get<double>("Scaling Factor"));
    }
};

template<typename EvalT, typename Traits, typename ScalarT>
class SimpleOperationExp : public SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::Exp<ScalarT> >
{
public:
  SimpleOperationExp (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::Exp<ScalarT> >(p,dl) {
      this->op.tau = ScalarT(p.get<double>("Tau"));
    }
};

template<typename EvalT, typename Traits, typename ScalarT>
class SimpleOperationLog : public SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::Log<ScalarT> >
{
public:
  SimpleOperationLog (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::Log<ScalarT> > (p,dl) {}
};

template<typename EvalT, typename Traits, typename ScalarT>
class SimpleOperationLowPass : public SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::LowPass<ScalarT> >
{
public:
  SimpleOperationLowPass (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::LowPass<ScalarT> > (p,dl) {
      this->op.threshold_up = ScalarT(p.get<double>("Upper Threshold"));
    }
};

template<typename EvalT, typename Traits, typename ScalarT>
class SimpleOperationHighPass : public SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::HighPass<ScalarT> >
{
public:
  SimpleOperationHighPass (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::HighPass<ScalarT> > (p,dl) {
      this->op.threshold_lo = ScalarT(p.get<double>("Lower Threshold"));
    }
};

template<typename EvalT, typename Traits, typename ScalarT>
class SimpleOperationBandPass : public SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::BandPass<ScalarT> >
{
public:
  SimpleOperationBandPass (const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl) :
    SimpleOperationBase<EvalT,Traits,ScalarT,SimpleOps::BandPass<ScalarT> > (p,dl) {
      this->op.threshold_lo = ScalarT(p.get<double>("Lower Threshold"));
      this->op.threshold_up = ScalarT(p.get<double>("Upper Threshold"));
    }
};

} // Namespace FELIX

#endif // FELIX_SIMPLE_OPERATION_HPP
