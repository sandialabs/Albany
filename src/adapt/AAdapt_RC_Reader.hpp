//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_RC_READER
#define AADAPT_RC_READER

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace AAdapt {
namespace rc {

class Manager;

/*! This evaluator reads data from the rc::Manager to provide to downstream
 *! evaluators.
 */

template<typename EvalT, typename Traits>
class Reader : public PHX::EvaluatorWithBaseImpl<Traits>,
               public PHX::EvaluatorDerived<EvalT, Traits> {
public:
  Reader(const Teuchos::RCP<Manager>& rc_mgr);
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

#endif // AADAPT_RC_READER
