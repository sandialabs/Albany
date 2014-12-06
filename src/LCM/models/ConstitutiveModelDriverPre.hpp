//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ConstitutiveModelDriverPre_hpp)
#define LCM_ConstitutiveModelDriverPre_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LCM {

  /// \brief Constitutive Model Driver
  template<typename EvalT, typename Traits>
  class ConstitutiveModelDriverPre : public PHX::EvaluatorWithBaseImpl<Traits>,
                                     public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    ConstitutiveModelDriverPre(Teuchos::ParameterList& p,
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
    /// solution field
    ///
    PHX::MDField<ScalarT, Cell, Node, Dim, Dim> solution_;

    ///
    /// def grad field
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> def_grad_;

    ///
    /// det def grad field
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint> j_;

    ///
    /// Number of dimensions
    ///
    std::size_t num_dims_;
    
    ///
    /// Number of integration points
    ///
    std::size_t num_pts_;
    
    ///
    /// Number of integration points
    ///
    std::size_t num_nodes_;

  };
}

#endif
