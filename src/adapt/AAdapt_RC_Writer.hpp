//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_RC_WRITER
#define AADAPT_RC_WRITER

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace AAdapt {
namespace rc {

class Manager;

/*! This evaluator writes data from the upstream evaluators to the rc::Manager.
 */
template<typename EvalT, typename Traits>
class WriterBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits> {
public:
  WriterBase();
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm) = 0;
  void evaluateFields(typename Traits::EvalData d) = 0;
  const Teuchos::RCP<const PHX::FieldTag>& getNoOutputTag();
private:
  Teuchos::RCP<const PHX::FieldTag> nooutput_tag_;
};

template<typename EvalT, typename Traits>
class Writer : public WriterBase<EvalT, Traits> {
public:
  Writer(const Teuchos::RCP<Manager>& rc_mgr) {}
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm) {}
  void evaluateFields(typename Traits::EvalData d) {}
};

template<typename Traits>
class Writer<PHAL::AlbanyTraits::Residual, Traits>
  : public WriterBase<PHAL::AlbanyTraits::Residual, Traits> {
public:
  Writer(const Teuchos::RCP<Manager>& rc_mgr);
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename std::vector< PHX::MDField<RealType> > FieldsVector;
  typedef typename FieldsVector::iterator FieldsIterator;

  Teuchos::RCP<Manager> rc_mgr_;
  FieldsVector fields_;
};

} // namespace rc
} // namespace AAdapt

#endif // AADAPT_RC_WRITER
