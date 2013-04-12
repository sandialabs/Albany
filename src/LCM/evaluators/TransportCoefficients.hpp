//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_Transport_Coefficients_hpp)
#define LCM_Transport_Coefficients_hpp

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LCM {
  /// \brief
  ///
  /// This evaluator computes the hydrogen concentration at trapped site
  /// through conservation of hydrogen atom
  ///
  template<typename EvalT, typename Traits>
  class TransportCoefficients : public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    TransportCoefficients(const Teuchos::ParameterList& p,
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
    /// Input: lattice concentration
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> c_lattice_;

    ///
    /// Input: number of trap sites
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> n_trap_;

    ///
    /// Input: concentration equilibrium parameter
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> k_eq_;

    ///
    /// Input: Equvalent plastic strain
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> eqps_;

    ///
    /// Output: trapped concentration
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> c_trapped_;

    ///
    /// Output: trapped concentration
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> eff_diff_;

    ///
    /// Output: strain_rate_factor
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> strain_rate_fac_;

    ///
    /// Number of integration points
    ///
    std::size_t num_pts_;

    ///
    /// Number of lattice sites
    ///
    RealType n_lattice_;

    ///
    /// Trapped Solvent Coefficients
    ///
    RealType a_, b_, c_, avogadros_num_;

    ///
    /// bool to check for equivalent plastic strain
    ///
    bool have_eqps_;

  };
}

#endif
