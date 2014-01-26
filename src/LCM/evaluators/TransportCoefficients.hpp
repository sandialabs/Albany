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
  /// This evaluator computes various terms required for the
 ///  hydrogen diffusion-deformation problem
  ///
  template<typename EvalT, typename Traits>
  class TransportCoefficients : public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    TransportCoefficients(Teuchos::ParameterList& p,
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
    /// Input: Equvalent plastic strain
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> eqps_;

    ///
    /// Input: Temperature
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> temperature_;

    ///
    /// Input: deformation gradient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> F_;

    ///
    /// Input: determinant of deformation gradient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> J_;

    ///
    /// Output: concentration equilibrium parameter
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> k_eq_;

    ///
    /// Output: diffusion coefficient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> diffusion_coefficient_;

    ///
    /// Output: convection coefficient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> convection_coefficient_;

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
    /// Output: total concentration
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> total_concentration_;

    ///
    /// Output: Mechanical deformation gradient
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> F_mech_;

    ///
    /// Number of integration points
    ///
    std::size_t num_pts_;

    ///
    /// Number of dimension
    ///
    std::size_t num_dims_;

    ///
    /// Number of cell
    ///
    std::size_t  worksetSize;

    ///
    /// flag to compute the weighted average of J
    ///
    bool weighted_average_;

    ///
    /// stabilization parameter for the weighted average
    ///
    ScalarT alpha_;

    ///
    /// Number of lattice sites
    ///
    RealType n_lattice_;

    ///
    /// Ideal Gas Constant
    ///
    RealType ideal_gas_constant_;

    ///
    /// Trap Binding Energy
    ///
    RealType trap_binding_energy_;

    ///
    /// Trapped Solvent Coefficients
    ///
    RealType a_, b_, c_, avogadros_num_;

    ///
    /// Pre-exponential Factor
    ///
    RealType pre_exponential_factor_;

    ///
    /// Diffusion Activation Enthalpy
    ///
    RealType Q_;

    ///
    /// Partial Molar Volume
    ///
    RealType partial_molar_volume_;

    ///
    /// Partial Molar Volume
    ///
    RealType ref_total_concentration_;

    ///
    /// Lattice Strain Flag
    ///
    bool lattice_strain_flag_;

    ///
    /// bool to check for equivalent plastic strain
    ///
    bool have_eqps_;

  };
}

#endif
