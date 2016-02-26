//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Unit_Gradient_hpp)
#define LCM_Unit_Gradient_hpp

#include <Phalanx_config.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>

#include "Albany_Layouts.hpp"

namespace LCM {
  /// \brief
  ///
  /// Compute solution gradient unit vector.
  ///
  template<typename EvalT, typename Traits>
  class UnitGradient : public PHX::EvaluatorWithBaseImpl<Traits>,
                       public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    UnitGradient(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);

    ///
    /// Phalanx method to allocate space
    ///
    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);

    ///
    /// Implementation of physics
    ///
    void evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ///
    /// Input: scalar gradient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> scalar_grad_;

    ///
    /// Output: unit scalar gradient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> unit_grad_;

    ///
    /// Number of integration points
    ///
    int num_pts_;

    ///
    /// Number of spatial dimensions
    ///
    int num_dims_;
  };
}

#endif
