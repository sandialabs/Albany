//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef EFFECTIVE_DIFFUSIVITY_HPP
#define EFFECTIVE_DIFFUSIVITY_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {

  /// \brief Effective Diffusivity Evaluator
  ///
  /// Compute effective diffusivity \f$ D^{*} = 1 + \partial C_{T} / \partial C_{L} \f$
  ///
  template<typename EvalT, typename Traits>
  class EffectiveDiffusivity : public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    EffectiveDiffusivity(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    // Input:
    PHX::MDField<ScalarT,Cell,QuadPoint> Vmolar;
    PHX::MDField<ScalarT,Cell,QuadPoint> Clattice;
    PHX::MDField<ScalarT,Cell,QuadPoint> avogadroNUM;
    PHX::MDField<ScalarT,Cell,QuadPoint> Ntrap;
    PHX::MDField<ScalarT,Cell,QuadPoint> Keq;

    ScalarT Nlattice;

    // Output:
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> effectiveDiffusivity;

    unsigned int numQPs;
    unsigned int numDims;
  };
}

#endif
