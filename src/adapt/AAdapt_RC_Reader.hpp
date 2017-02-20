//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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

/*! \brief This evaluator reads data from the rc::Manager to provide to downstream
 *! evaluators.
 *
 *  All specializations evaluate RealType state values. Additionally, the
 *  Residual specialization optionally implements part of the projection.
 */

template<typename EvalT, typename Traits>
class ReaderBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                   public PHX::EvaluatorDerived<EvalT, Traits> {
public:
  ReaderBase(const Teuchos::RCP<Manager>& rc_mgr);
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& fm);
  void evaluateFields(typename Traits::EvalData d);
  const Teuchos::RCP<const PHX::FieldTag>& getNoOutputTag();
protected:
  typedef typename std::vector< PHX::MDField<RealType> > FieldsVector;
  typedef typename FieldsVector::iterator FieldsIterator;
  Teuchos::RCP<Manager> rc_mgr_;
  FieldsVector fields_;  
};

template<typename EvalT, typename Traits>
class Reader : public ReaderBase<EvalT, Traits> {
public:
  Reader(const Teuchos::RCP<Manager>& rc_mgr);
};

template<typename Traits>
class Reader<PHAL::AlbanyTraits::Residual, Traits>
  : public ReaderBase<PHAL::AlbanyTraits::Residual, Traits> {
public:
  Reader(const Teuchos::RCP<Manager>& rc_mgr,
         const Teuchos::RCP<Albany::Layouts>& dl);
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& fm);
  void evaluateFields(typename Traits::EvalData d);
private:
  PHX::MDField<RealType,Cell,Node,QuadPoint> bf_, wbf_;
  // For correctness testing the projector.
  struct InterpTestData;
  Teuchos::RCP<InterpTestData> itd_;
};

} // namespace rc
} // namespace AAdapt

#endif // AADAPT_RC_READER
