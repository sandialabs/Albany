//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PERIDIGMPARTIALSTRESS_HPP
#define PERIDIGMPARTIALSTRESS_HPP

#include "PeridigmManager.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include <Peridigm.hpp>
#include <Peridigm_AlbanyDiscretization.hpp>

namespace LCM {
/** \brief Copies partial stress values from Peridigm to Albany for couplied
   simulations.

*/

template <typename EvalT, typename Traits>
class PeridigmPartialStressBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                  public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  PeridigmPartialStressBase(const Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

 protected:
  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  unsigned int numQPs;
  unsigned int numDims;

  // Input:
  PHX::MDField<ScalarT, Cell, QuadPoint>           J;
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> defgrad;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;

  Teuchos::RCP<PeridigmNS::Peridigm> peridigm;
};

// Inherted classes
template <typename EvalT, typename Traits>
class PeridigmPartialStress;

// For all cases except those specialized below, just fall through to base
// class. The base class throws "Not Implemented" for evaluate fields.
template <typename EvalT, typename Traits>
class PeridigmPartialStress : public PeridigmPartialStressBase<EvalT, Traits>
{
 public:
  PeridigmPartialStress(Teuchos::ParameterList& p)
      : PeridigmPartialStressBase<EvalT, Traits>(p){};
};

// Template Specialization: Residual Evaluation (standard force evaluation)
template <typename Traits>
class PeridigmPartialStress<PHAL::AlbanyTraits::Residual, Traits>
    : public PeridigmPartialStressBase<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  PeridigmPartialStress(Teuchos::ParameterList& p)
      : PeridigmPartialStressBase<PHAL::AlbanyTraits::Residual, Traits>(p){};
  void
  evaluateFields(typename Traits::EvalData d);
};

}  // namespace LCM

#endif
